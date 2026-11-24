"""Implicit Q-Learning para recomendación sobre episodios de 20 pasos.

V, Q y π usan codificadores causales separados sobre el historial de la sesión.
Q se ajusta con un objetivo TD que usa V(s') y una red objetivo de Q; V se
ajusta por regresión de expectil sobre Q̄; π se extrae por AWR. La recompensa se
centra en (r − 3) / 2, igual que en el codificador de estado.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelos.dt import seen_before_mask


class StateEncoder(nn.Module):
    """Transformer causal sobre los pares (ítem, rating); devuelve T+1 estados."""

    def __init__(
        self,
        n_items: int,
        horizon: int = 20,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        n_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model debe ser divisible por n_heads")
        self.horizon = horizon
        self.n_groups = n_groups
        self.item_emb = nn.Embedding(n_items, d_model)
        self.rating_emb = nn.Linear(1, d_model)
        self.group_emb = nn.Embedding(n_groups + 1, d_model)
        self.pos_emb = nn.Embedding(horizon + 1, d_model)
        self.start = nn.Parameter(torch.zeros(d_model))
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
        for emb in (self.item_emb, self.group_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, items: torch.Tensor, ratings: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        B, L = items.shape
        if L > self.horizon:
            raise ValueError(f"L={L} supera el horizonte {self.horizon}")
        dtype = self.item_emb.weight.dtype
        pairs = self.item_emb(items) + self.rating_emb(((ratings.to(dtype) - 3.0) / 2.0).unsqueeze(-1))
        x = torch.cat([self.start.expand(B, 1, -1), pairs], dim=1)
        pos = self.pos_emb(torch.arange(L + 1, device=items.device)).unsqueeze(0)
        x = x + pos + self.group_emb(groups).unsqueeze(1)
        causal = torch.triu(
            torch.ones(L + 1, L + 1, dtype=torch.bool, device=items.device), diagonal=1
        )
        return self.ln_out(self.encoder(x, mask=causal))


class IQLNetwork(nn.Module):
    """Codificador de estado más una cabeza lineal (n_items para Q/π, 1 para V)."""

    def __init__(self, out_dim: int, **encoder_kwargs):
        super().__init__()
        self.encoder = StateEncoder(**encoder_kwargs)
        self.head = nn.Linear(encoder_kwargs.get("d_model", 64), out_dim)

    def forward(self, items: torch.Tensor, ratings: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(items, ratings, groups))


class IQL(nn.Module):
    def __init__(
        self,
        n_items: int,
        horizon: int = 20,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        n_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_items = n_items
        self.horizon = horizon
        self.n_groups = n_groups
        kwargs = dict(
            n_items=n_items,
            horizon=horizon,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_groups=n_groups,
            dropout=dropout,
        )
        self.q = IQLNetwork(n_items, **kwargs)
        self.v = IQLNetwork(1, **kwargs)
        self.pi = IQLNetwork(n_items, **kwargs)
        self.q_target = copy.deepcopy(self.q)
        for parameter in self.q_target.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.q_target.eval()
        return self

    @torch.no_grad()
    def update_target(self, lam: float = 0.005) -> None:
        for target, online in zip(self.q_target.parameters(), self.q.parameters()):
            target.mul_(1.0 - lam).add_(online, alpha=lam)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """diff = Q̄ − V. Con τ > 0,5 penaliza más quedarse por debajo (diff > 0)."""
    weight = tau * (diff > 0) + (1.0 - tau) * (diff <= 0)
    return (weight * diff.pow(2)).mean()


def centered_reward(ratings: torch.Tensor) -> torch.Tensor:
    return (ratings - 3.0) / 2.0


def td_target(r: torch.Tensor, v_all: torch.Tensor, gamma: float) -> torch.Tensor:
    """r: (B, T); v_all: (B, T+1). El último paso es terminal: objetivo r."""
    not_done = torch.ones_like(r)
    not_done[:, -1] = 0.0
    return r + gamma * not_done * v_all[:, 1:].detach()


def iql_losses(
    model: IQL,
    batch: dict[str, torch.Tensor],
    gamma: float,
    tau: float,
    beta: float,
    max_weight: float = 100.0,
) -> dict[str, torch.Tensor]:
    items, ratings, groups = batch["items"], batch["ratings"], batch["groups"]
    B, T = items.shape
    r = centered_reward(ratings)
    actions = items.unsqueeze(-1)

    with torch.no_grad():
        q_obj_all = model.q_target(items, ratings, groups).float()
        q_obj = q_obj_all[:, :T].gather(-1, actions).squeeze(-1)

    v_all = model.v(items, ratings, groups).float().squeeze(-1)
    loss_v = expectile_loss(q_obj - v_all[:, :T], tau)

    q_all = model.q(items, ratings, groups).float()
    q = q_all[:, :T].gather(-1, actions).squeeze(-1)
    loss_q = F.mse_loss(q, td_target(r, v_all, gamma))

    with torch.no_grad():
        weights = torch.exp(beta * (q_obj - v_all[:, :T])).clamp(max=max_weight)
    logits = model.pi(items, ratings, groups).float()[:, :T]
    logits = logits.masked_fill(seen_before_mask(items, model.n_items), float("-inf"))
    ce = F.cross_entropy(
        logits.reshape(-1, model.n_items), items.reshape(-1), reduction="none"
    ).reshape(B, T)
    loss_pi = (weights * ce).mean()

    return {"v": loss_v, "q": loss_q, "pi": loss_pi, "weights": weights}


def episodes_to_iql_tensors(ep, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return {
        "items": torch.as_tensor(ep.items, dtype=torch.long, device=device),
        "ratings": torch.as_tensor(ep.ratings, dtype=torch.float32, device=device),
        "groups": torch.as_tensor(ep.groups, dtype=torch.long, device=device),
    }


@torch.no_grad()
def iql_mean_losses(
    model: IQL, data: dict[str, torch.Tensor], batch_size: int = 512, **loss_kwargs
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    n = data["items"].shape[0]
    total = {"v": 0.0, "q": 0.0, "pi": 0.0, "w": 0.0}
    for start in range(0, n, batch_size):
        batch = {k: v[start : start + batch_size] for k, v in data.items()}
        losses = iql_losses(model, batch, **loss_kwargs)
        nb = batch["items"].shape[0]
        for key in ("v", "q", "pi"):
            total[key] += losses[key].item() * nb
        total["w"] += losses["weights"].mean().item() * nb
    if was_training:
        model.train()
    return {key: value / n for key, value in total.items()}


def train_iql(
    train_ep,
    val_ep,
    n_items: int,
    seed: int = 0,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    gamma: float = 0.9,
    tau: float = 0.9,
    beta: float = 3.0,
    max_weight: float = 100.0,
    target_lambda: float = 0.005,
    group_dropout: float = 0.3,
    grad_clip: float = 1.0,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    dropout: float = 0.1,
    max_batches: int | None = None,
    device: str | torch.device = "cpu",
    amp: bool = False,
    output: str | Path | None = None,
    verbose: bool = True,
):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    horizon = int(train_ep.items.shape[1])
    train_data = episodes_to_iql_tensors(train_ep, device)
    val_data = episodes_to_iql_tensors(val_ep, device)

    model = IQL(
        n_items=n_items,
        horizon=horizon,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    use_amp = bool(amp and torch.cuda.is_available() and torch.device(device).type == "cuda")
    use_bf16 = bool(use_amp and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_amp and not use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    n_episodes = train_data["items"].shape[0]
    steps_per_epoch = max(1, (n_episodes + batch_size - 1) // batch_size)
    if max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(1, epochs * steps_per_epoch)
    )

    loss_kwargs = dict(gamma=gamma, tau=tau, beta=beta, max_weight=max_weight)
    config = {
        "n_items": n_items,
        "horizon": horizon,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "group_dropout": group_dropout,
        "target_lambda": target_lambda,
        "amp": use_amp,
        "amp_dtype": "bf16" if use_bf16 else ("fp16" if use_fp16 else None),
        "device": str(device),
        **loss_kwargs,
    }

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_episodes, generator=generator).to(device)
        accumulated = {"v": 0.0, "q": 0.0, "pi": 0.0, "w": 0.0, "clip": 0.0}
        batches = 0

        for step in range(steps_per_epoch):
            index = order[step * batch_size : (step + 1) * batch_size]
            if index.numel() == 0:
                continue
            batch = {k: v[index] for k, v in train_data.items()}
            if group_dropout > 0:
                drop = (
                    torch.rand(batch["groups"].shape, generator=generator).to(device) < group_dropout
                )
                batch["groups"] = torch.where(
                    drop, torch.full_like(batch["groups"], model.n_groups), batch["groups"]
                )

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    losses = iql_losses(model, batch, **loss_kwargs)
            else:
                losses = iql_losses(model, batch, **loss_kwargs)
            loss = losses["v"] + losses["q"] + losses["pi"]

            optimizer.zero_grad(set_to_none=True)
            if use_fp16:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
            model.update_target(target_lambda)

            weights = losses["weights"]
            for key in ("v", "q", "pi"):
                accumulated[key] += losses[key].item()
            accumulated["w"] += weights.mean().item()
            accumulated["clip"] += (weights >= max_weight).float().mean().item()
            batches += 1

        n_batches = max(1, batches)
        val_losses = iql_mean_losses(model, val_data, **loss_kwargs)
        history.append(
            {
                "epoch": epoch,
                "train_v": accumulated["v"] / n_batches,
                "train_q": accumulated["q"] / n_batches,
                "train_pi": accumulated["pi"] / n_batches,
                "w_medio": accumulated["w"] / n_batches,
                "w_truncado": accumulated["clip"] / n_batches,
                "val_v": val_losses["v"],
                "val_q": val_losses["q"],
                "val_pi": val_losses["pi"],
            }
        )
        if verbose:
            row = history[-1]
            print(
                f"epoch {epoch:3d}/{epochs} | V {row['train_v']:.4f} Q {row['train_q']:.4f} "
                f"π {row['train_pi']:.4f} | peso medio {row['w_medio']:.2f} "
                f"(truncado {row['w_truncado']:.1%}) | val π {row['val_pi']:.4f}"
            )

    model.eval()
    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config}, out / "modelo_iql.pt")
        pd.DataFrame(history).to_csv(out / "historial.csv", index=False)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if verbose:
            print(f"checkpoint e historial en {out}")

    return model, pd.DataFrame(history)


def load_iql(path: str | Path, device: str | torch.device = "cpu") -> IQL:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = IQL(
        n_items=config["n_items"],
        horizon=config["horizon"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


class IQLPolicy:
    """mode="pi": logits de π; mode="q": Q sobre todos los ítems disponibles."""

    def __init__(self, model: IQL, mode: str = "pi"):
        if mode not in ("pi", "q"):
            raise ValueError("mode debe ser 'pi' o 'q'")
        self.model = model.eval()
        self.mode = mode
        self.group = model.n_groups

    def reset(self, group, horizon):
        self.group = self.model.n_groups if group is None else int(group)

    @torch.no_grad()
    def scores(self, items_seen, ratings_seen):
        items_seen = np.asarray(items_seen, dtype=np.int64)
        ratings_seen = np.asarray(ratings_seen, dtype=np.float32)
        t = len(items_seen)
        device = next(self.model.parameters()).device
        items = torch.as_tensor(items_seen, device=device).unsqueeze(0)
        ratings = torch.as_tensor(ratings_seen, device=device).unsqueeze(0)
        groups = torch.tensor([self.group], device=device)
        network = self.model.pi if self.mode == "pi" else self.model.q
        out = network(items, ratings, groups)[0, t].clone()
        out[items_seen] = float("-inf")
        return out.cpu().numpy()
