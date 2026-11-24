"""Tests de IQL: alineación, no fuga, expectil, gradientes y políticas."""

import numpy as np
import torch

from modelos.iql import (
    IQL,
    IQLPolicy,
    expectile_loss,
    iql_losses,
    load_iql,
    td_target,
    train_iql,
)
from protocolo.datos import Episodes


def make_iql_batch(B=4, T=20, n_items=50, n_groups=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    items = torch.stack(
        [torch.randperm(n_items, generator=generator)[:T] for _ in range(B)]
    )
    ratings = torch.randint(1, 6, (B, T), generator=generator).float()
    groups = torch.randint(0, n_groups, (B,), generator=generator)
    return {"items": items, "ratings": ratings, "groups": groups}


def make_small_iql(n_items=50):
    torch.manual_seed(0)
    return IQL(n_items=n_items, horizon=20, d_model=32, n_layers=1, n_heads=2, dropout=0.0)


def make_fake_episodes(n=64, T=20, n_items=50, n_groups=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    items = torch.stack(
        [torch.randperm(n_items, generator=generator)[:T] for _ in range(n)]
    ).numpy()
    ratings = torch.randint(1, 6, (n, T), generator=generator).float().numpy()
    groups = torch.randint(0, n_groups, (n,), generator=generator).numpy()
    return Episodes(items=items, ratings=ratings, groups=groups, user_ids=np.arange(n))


def test_shapes():
    model = make_small_iql()
    batch = make_iql_batch()
    for network, dim in ((model.q, 50), (model.v, 1), (model.pi, 50)):
        out = network(batch["items"], batch["ratings"], batch["groups"])
        assert out.shape == (4, 21, dim)


def test_state_does_not_see_current_or_future():
    model = make_small_iql().eval()
    batch = make_iql_batch()
    items, ratings, groups = batch["items"], batch["ratings"], batch["groups"]
    with torch.no_grad():
        for t in (0, 7, 19):
            base = model.q(items, ratings, groups)[:, t]
            items2 = items.clone()
            items2[:, t:] = torch.randint(0, 50, items2[:, t:].shape)
            ratings2 = ratings.clone()
            ratings2[:, t:] = torch.randint(1, 6, ratings2[:, t:].shape).float()
            out = model.q(items2, ratings2, groups)[:, t]
            assert torch.allclose(base, out, atol=1e-5), f"fuga en t={t}"


def test_state_uses_past_and_group():
    model = make_small_iql().eval()
    batch = make_iql_batch()
    items, ratings, groups = batch["items"], batch["ratings"], batch["groups"]
    with torch.no_grad():
        base = model.pi(items, ratings, groups)[:, 7]
        ratings2 = ratings.clone()
        ratings2[:, 6] += 2.0
        assert not torch.allclose(base, model.pi(items, ratings2, groups)[:, 7])
        items2 = items.clone()
        items2[:, 6] = (items2[:, 6] + 1) % 50
        assert not torch.allclose(base, model.pi(items2, ratings, groups)[:, 7])
        assert not torch.allclose(base, model.pi(items, ratings, (groups + 1) % 8)[:, 7])


def test_expectile_values():
    values = torch.tensor([1.0, 2.0, 3.0, 10.0])
    for tau, expected in ((0.5, 4.0), (0.7, 5.5), (0.9, 8.0)):
        v = torch.tensor(expected, requires_grad=True)
        expectile_loss(values - v, tau).backward()
        assert abs(v.grad.item()) < 1e-4


def test_td_target_terminal_and_gamma_zero():
    r = torch.ones(2, 20)
    v_all = torch.randn(2, 21)
    target = td_target(r, v_all, 0.9)
    v2 = v_all.clone()
    v2[:, -1] += 100.0
    assert torch.allclose(target, td_target(r, v2, 0.9))
    assert torch.allclose(target[:, -1], r[:, -1])
    assert torch.allclose(target[:, 0], r[:, 0] + 0.9 * v_all[:, 1])
    assert torch.allclose(td_target(r, v_all, 0.0), r)


def test_gradient_isolation():
    model = make_small_iql()
    batch = make_iql_batch()
    networks = {"v": model.v, "q": model.q, "pi": model.pi, "q_target": model.q_target}
    model.train()
    for key in ("v", "q", "pi"):
        model.zero_grad(set_to_none=True)
        iql_losses(model, batch, gamma=0.9, tau=0.9, beta=3.0)[key].backward()
        for name, network in networks.items():
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters()
            )
            assert has_grad == (name == key), f"la pérdida {key} llegó a {name}"


def test_polyak_target():
    model = make_small_iql()
    before = next(model.q_target.parameters()).clone()
    with torch.no_grad():
        for parameter in model.q.parameters():
            parameter.add_(1.0)
    model.update_target(0.1)
    assert torch.allclose(next(model.q_target.parameters()), before + 0.1, atol=1e-6)


def test_policy_scores_and_mask():
    model = make_small_iql(n_items=40)
    for mode in ("pi", "q"):
        policy = IQLPolicy(model, mode)
        policy.reset(group=None, horizon=20)
        scores = policy.scores(np.array([], dtype=np.int64), np.array([], dtype=np.float64))
        assert scores.shape == (40,)
        assert np.isfinite(scores).all()
        seen = np.array([1, 2, 3])
        policy.reset(group=3, horizon=20)
        scores = policy.scores(seen, np.array([4.0, 5.0, 3.0]))
        assert np.isneginf(scores[seen]).all()
        assert np.isfinite(scores[~np.isin(np.arange(40), seen)]).all()


def test_train_and_load_smoke(tmp_path):
    train_ep = make_fake_episodes(seed=0)
    val_ep = make_fake_episodes(n=32, seed=1)
    model, history = train_iql(
        train_ep,
        val_ep,
        n_items=50,
        seed=0,
        epochs=1,
        batch_size=16,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_batches=2,
        output=tmp_path,
        verbose=False,
    )
    assert len(history) == 1
    assert (tmp_path / "modelo_iql.pt").exists()
    assert (tmp_path / "historial.csv").exists()
    loaded = load_iql(tmp_path / "modelo_iql.pt")
    batch = make_iql_batch(n_items=50)
    with torch.no_grad():
        assert loaded.q(batch["items"], batch["ratings"], batch["groups"]).shape == (
            4,
            21,
            50,
        )


def test_policy_scores_batch_matches_scores():
    model = make_small_iql()
    batch = make_iql_batch()
    groups = batch["groups"].tolist()
    items, ratings = batch["items"], batch["ratings"]
    for mode in ("pi", "q"):
        policy = IQLPolicy(model, mode)
        for t in (0, 5, 19):
            items_seen = [items[i, :t].numpy() for i in range(items.shape[0])]
            ratings_seen = [ratings[i, :t].numpy() for i in range(ratings.shape[0])]
            batched = policy.scores_batch(items_seen, ratings_seen, groups)
            for i in range(len(items_seen)):
                policy.reset(groups[i], 20)
                single = policy.scores(items_seen[i], ratings_seen[i])
                finite = np.isfinite(single)
                assert np.array_equal(finite, np.isfinite(batched[i]))
                assert np.allclose(single[finite], batched[i][finite], atol=1e-3, rtol=1e-4)
