"""Entrenamiento y carga del Decision Transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from modelos.dt import DecisionTransformer, dt_loss, masked_action_loss
from protocolo.datos import Episodes, load_train, make_episodes, num_items, split_by_user


def episodes_to_tensors(ep: Episodes, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    ratings = torch.as_tensor(ep.ratings, dtype=torch.float32, device=device)
    items = torch.as_tensor(ep.items, dtype=torch.long, device=device)
    groups = torch.as_tensor(ep.groups, dtype=torch.long, device=device)
    state = torch.zeros_like(ratings)
    state[:, 1:] = ratings[:, :-1]
    rtg = torch.as_tensor(
        np.ascontiguousarray(ep.returns_to_go()), dtype=torch.float32, device=device
    )
    return {"rtg": rtg, "items": items, "state": state, "groups": groups, "ratings": ratings}


@torch.no_grad()
def mean_loss(model, data: dict[str, torch.Tensor], batch_size: int = 512) -> float:
    was_training = model.training
    model.eval()
    n = data["items"].shape[0]
    total = 0.0
    for start in range(0, n, batch_size):
        batch = {k: v[start : start + batch_size] for k, v in data.items() if k != "ratings"}
        loss = dt_loss(model, batch)
        total += loss.item() * batch["items"].numel()
    if was_training:
        model.train()
    return total / (n * data["items"].shape[1])


def train_dt(
    train_ep: Episodes,
    val_ep: Episodes,
    n_items: int,
    seed: int = 0,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    group_dropout: float = 0.3,
    grad_clip: float = 1.0,
    d_model: int = 128,
    n_layers: int = 3,
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
    train_data = episodes_to_tensors(train_ep, device)
    val_data = episodes_to_tensors(val_ep, device)

    model = DecisionTransformer(
        n_items=n_items,
        horizon=horizon,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = bool(amp and torch.cuda.is_available() and torch.device(device).type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_episodes = train_data["items"].shape[0]
    steps_per_epoch = max(1, (n_episodes + batch_size - 1) // batch_size)
    if max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(1, epochs * steps_per_epoch)
    )

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
        "amp": use_amp,
        "device": str(device),
    }

    history = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_episodes, generator=generator).to(device)
        running, batches = 0.0, 0

        for step in range(steps_per_epoch):
            index = order[step * batch_size : (step + 1) * batch_size]
            if index.numel() == 0:
                continue
            batch = {k: v[index] for k, v in train_data.items() if k != "ratings"}
            if group_dropout > 0:
                drop = torch.rand(batch["groups"].shape, generator=generator).to(device) < group_dropout
                batch["groups"] = torch.where(
                    drop,
                    torch.full_like(batch["groups"], model.n_groups),
                    batch["groups"],
                )
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch["rtg"], batch["items"], batch["state"], batch["groups"])
                loss = masked_action_loss(logits, batch["items"])
            else:
                loss = dt_loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            batches += 1

        train_loss = running / max(1, batches)
        val_loss = mean_loss(model, val_data)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"epoch {epoch:3d}/{epochs} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | mejor {best_val:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": config,
                "best_val_loss": best_val,
            },
            out / "modelo.pt",
        )
        pd.DataFrame(history).to_csv(out / "historial.csv", index=False)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if verbose:
            print(f"checkpoint e historial en {out}")

    return model, pd.DataFrame(history), best_val


def load_dt(path: str | Path, device: str | torch.device = "cpu") -> DecisionTransformer:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = DecisionTransformer(
        n_items=config["n_items"],
        horizon=config["horizon"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena un Decision Transformer")
    parser.add_argument("--dataset", default="netflix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--group-dropout", type=float, default=0.3)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    train_all = load_train(args.dataset)
    train, val = split_by_user(train_all)
    train_ep = make_episodes(train, length=20)
    val_ep = make_episodes(val, length=20)

    output = args.output or f"outputs/dt/{args.dataset}/seed{args.seed}"
    _, history, best_val = train_dt(
        train_ep,
        val_ep,
        n_items=num_items(args.dataset),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        group_dropout=args.group_dropout,
        max_batches=args.max_batches,
        device=device,
        amp=args.amp,
        output=output,
    )
    print(f"mejor val CE: {best_val:.4f} | referencia uniforme log(n_items) ≈ {np.log(num_items(args.dataset)):.2f}")


if __name__ == "__main__":
    main()
