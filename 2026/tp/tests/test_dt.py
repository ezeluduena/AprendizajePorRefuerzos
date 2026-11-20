"""Tests del Decision Transformer: alineación, no fuga y política."""

import numpy as np
import torch

from modelos.dt import (
    DecisionTransformer,
    DTPolicy,
    check_no_leak,
    dt_loss,
    seen_before_mask,
)


def make_batch(B=4, T=20, n_items=50, n_groups=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = torch.stack([torch.randperm(n_items, generator=g)[:T] for _ in range(B)])
    ratings = torch.randint(1, 6, (B, T), generator=g).float()
    state = torch.zeros_like(ratings)
    state[:, 1:] = ratings[:, :-1]
    rtg = torch.flip(torch.cumsum(torch.flip(ratings, dims=[1]), dim=1), dims=[1])
    groups = torch.randint(0, n_groups, (B,), generator=g)
    return {"items": items, "state": state, "rtg": rtg, "groups": groups}


def make_model(n_items=50, horizon=20, d_model=32, n_layers=1, n_heads=2):
    torch.manual_seed(0)
    return DecisionTransformer(
        n_items=n_items,
        horizon=horizon,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
    )


def test_seen_before_mask():
    items = torch.tensor([[17, 302, 45, 17]])
    mask = seen_before_mask(items, n_items=400)
    assert mask.shape == (1, 4, 400)
    assert not mask[0, 0].any()
    assert mask[0, 1, 17]
    assert mask[0, 2, 17] and mask[0, 2, 302]
    assert not mask[0, 2, 45]
    assert mask[0, 3, 17]


def test_output_shape():
    model = make_model()
    batch = make_batch()
    logits = model(batch["rtg"], batch["items"], batch["state"], batch["groups"])
    assert logits.shape == (4, 20, 50)


def test_loss_is_finite():
    model = make_model()
    assert torch.isfinite(dt_loss(model, make_batch()))


def test_prediction_at_t_ignores_current_and_future_tokens():
    model = make_model()
    batch = make_batch()
    assert check_no_leak(model, batch, t=7)


def test_policy_scores_shape_and_masking():
    model = make_model(n_items=40)
    policy = DTPolicy(model, target_return=80.0)

    policy.reset(group=None, horizon=20)
    scores = policy.scores(np.array([], dtype=np.int64), np.array([], dtype=np.float64))
    assert scores.shape == (40,)
    assert np.isfinite(scores).all()

    items = np.array([1, 2, 3])
    ratings = np.array([4.0, 5.0, 3.0])
    policy.reset(group=3, horizon=20)
    scores = policy.scores(items, ratings)
    assert scores.shape == (40,)
    assert np.isneginf(scores[items]).all()
    assert np.isfinite(scores[~np.isin(np.arange(40), items)]).all()


def test_prediction_uses_current_rtg():
    model = make_model()
    batch = make_batch()
    with torch.no_grad():
        base = model(batch["rtg"], batch["items"], batch["state"], batch["groups"])[:, 5]
        rtg2 = batch["rtg"].clone()
        rtg2[:, 5] += 25.0
        out = model(rtg2, batch["items"], batch["state"], batch["groups"])[:, 5]
    assert not torch.allclose(base, out)


def test_prediction_uses_current_state():
    model = make_model()
    batch = make_batch()
    with torch.no_grad():
        base = model(batch["rtg"], batch["items"], batch["state"], batch["groups"])[:, 5]
        state2 = batch["state"].clone()
        state2[:, 5] += 2.5
        out = model(batch["rtg"], batch["items"], state2, batch["groups"])[:, 5]
    assert not torch.allclose(base, out)
