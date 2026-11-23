"""Decision Transformer para recomendación sobre episodios de 20 pasos.

Tokens por paso: (R̂_t, s_t, a_t), intercalados como
R̂_0 s_0 a_0 R̂_1 s_1 a_1 ...

La predicción de a_t se lee de la salida del token s_t (posiciones 1, 4, 7, ...),
que por la máscara causal no ve a_t ni ningún token posterior. s_t combina el
grupo (o el token "desconocido") con el rating del paso anterior.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        n_items: int,
        horizon: int = 20,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        n_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model debe ser divisible por n_heads")
        self.n_items = n_items
        self.horizon = horizon
        self.n_groups = n_groups

        self.rtg_emb = nn.Linear(1, d_model)
        self.group_emb = nn.Embedding(n_groups + 1, d_model)
        self.prev_rating_emb = nn.Linear(1, d_model)
        self.item_emb = nn.Embedding(n_items, d_model)
        self.step_emb = nn.Embedding(horizon, d_model)
        self.type_emb = nn.Embedding(3, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln_out = nn.LayerNorm(d_model)

        for emb in (self.item_emb, self.group_emb, self.step_emb, self.type_emb):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, rtg, items, state, groups) -> torch.Tensor:
        """Devuelve logits (B, T, n_items) para a_t, calculados desde el token s_t."""
        if items.dim() != 2:
            raise ValueError("items debe tener forma (B, T)")
        B, T = items.shape
        if T > self.horizon:
            raise ValueError(f"T={T} supera el horizonte {self.horizon}")

        dtype = self.item_emb.weight.dtype
        positions = torch.arange(T, device=items.device)
        type_ids = torch.arange(3, device=items.device)

        rtg_tokens = self.rtg_emb(rtg.to(dtype).unsqueeze(-1) / 100.0)
        state_tokens = self.group_emb(groups).unsqueeze(1) + self.prev_rating_emb(
            state.to(dtype).unsqueeze(-1) / 5.0
        )
        item_tokens = self.item_emb(items)

        step = self.step_emb(positions).unsqueeze(0)
        tokens = torch.stack(
            [
                rtg_tokens + step + self.type_emb(type_ids[0]),
                state_tokens + step + self.type_emb(type_ids[1]),
                item_tokens + step + self.type_emb(type_ids[2]),
            ],
            dim=2,
        ).reshape(B, 3 * T, -1)

        causal = torch.triu(
            torch.ones(3 * T, 3 * T, dtype=torch.bool, device=items.device), diagonal=1
        )
        hidden = self.ln_out(self.encoder(tokens, mask=causal))
        state_out = hidden[:, 1::3]
        return state_out @ self.item_emb.weight.T


def seen_before_mask(items: torch.Tensor, n_items: int) -> torch.Tensor:
    """(B, T, n_items): True si el ítem ya se mostró antes del paso t en el episodio."""
    one_hot = F.one_hot(items, num_classes=n_items)
    seen = one_hot.cumsum(dim=1) > 0
    return torch.cat([torch.zeros_like(seen[:, :1]), seen[:, :-1]], dim=1)


def masked_action_loss(logits: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    logits = logits.masked_fill(seen_before_mask(items, logits.shape[-1]), float("-inf"))
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), items.reshape(-1))


def dt_loss(model, batch) -> torch.Tensor:
    logits = model(batch["rtg"], batch["items"], batch["state"], batch["groups"])
    return masked_action_loss(logits, batch["items"])


def check_no_leak(model: DecisionTransformer, batch, t: int = 7, atol: float = 1e-5) -> bool:
    """Test de no fuga: la salida del paso t no debe depender de a_t ni del futuro."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        base = model(batch["rtg"], batch["items"], batch["state"], batch["groups"])[:, t]
        items2 = batch["items"].clone()
        items2[:, t:] = torch.randint(
            0, model.n_items, items2[:, t:].shape, device=items2.device
        )
        rtg2 = batch["rtg"].clone()
        rtg2[:, t + 1:] = 0.0
        state2 = batch["state"].clone()
        state2[:, t + 1:] = 0.0
        out = model(rtg2, items2, state2, batch["groups"])[:, t]
    if was_training:
        model.train()
    return bool(torch.allclose(base, out, atol=atol))


class DTPolicy:
    """Política autorregresiva: pide un retorno objetivo y lee las logits de s_t."""

    def __init__(self, model: DecisionTransformer, target_return: float):
        self.model = model.eval()
        self.target = float(target_return)
        self.group = model.n_groups
        self.horizon = model.horizon

    def reset(self, group, horizon):
        self.group = self.model.n_groups if group is None else int(group)
        self.horizon = int(horizon)

    @torch.no_grad()
    def scores(self, items_seen, ratings_seen):
        items_seen = np.asarray(items_seen, dtype=np.int64)
        ratings_seen = np.asarray(ratings_seen, dtype=np.float64)
        t = len(items_seen)

        rtg = np.empty(t + 1, dtype=np.float32)
        rtg[0] = self.target
        if t:
            rtg[1:] = self.target - np.cumsum(ratings_seen)

        items = np.zeros(t + 1, dtype=np.int64)
        items[:t] = items_seen

        prev = np.zeros(t + 1, dtype=np.float32)
        prev[1:] = ratings_seen

        device = next(self.model.parameters()).device
        batch = {
            "rtg": torch.as_tensor(rtg, device=device).unsqueeze(0),
            "items": torch.as_tensor(items, device=device).unsqueeze(0),
            "state": torch.as_tensor(prev, device=device).unsqueeze(0),
            "groups": torch.tensor([self.group], device=device),
        }
        logits = self.model(**batch)[0, t].clone()
        logits[items_seen] = float("-inf")
        return logits.cpu().numpy()
