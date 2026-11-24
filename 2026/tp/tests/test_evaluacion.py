"""Tests del evaluador batcheado: equivalencia con el protocolo oficial."""

import numpy as np
import pandas as pd

from modelos.evaluacion import evaluate_fast
from protocolo.datos import User
from protocolo.evaluar import evaluate
from protocolo.simulador import SessionConfig


class FixedPolicy:
    def __init__(self, n_items, seed=7):
        self.base = np.random.default_rng(seed).random(n_items) + 1.0
        self.group = None

    def reset(self, group, horizon):
        self.group = group

    def _offset(self, group):
        return 0.0 if group is None else 0.1 * (group + 1)

    def scores(self, items_seen, ratings_seen):
        return self.base + self._offset(self.group)

    def scores_batch(self, items_seen, ratings_seen, groups):
        out = np.tile(self.base, (len(items_seen), 1))
        for index, group in enumerate(groups):
            out[index] += self._offset(group)
        return out


class FixedPolicyNoBatch:
    def __init__(self, n_items, seed=7):
        self.inner = FixedPolicy(n_items, seed)

    def reset(self, group, horizon):
        self.inner.reset(group, horizon)

    def scores(self, items_seen, ratings_seen):
        return self.inner.scores(items_seen, ratings_seen)


def make_users(n=12, n_items=40, length=30):
    rng = np.random.default_rng(0)
    return [
        User(
            user_id=user_id,
            group=user_id % 4,
            items=rng.choice(n_items, size=length, replace=False),
            ratings=rng.integers(1, 6, size=length).astype(float),
        )
        for user_id in range(n)
    ]


def compare_results(official, fast):
    official = official.sort_values(["seed", "user_id"]).reset_index(drop=True)
    fast = fast.sort_values(["seed", "user_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(official[["user_id", "group", "seed"]], fast[["user_id", "group", "seed"]])
    for column in [
        "return",
        "full_return",
        "mean_rating",
        "hit_rate",
        "normalized",
        "random_expected",
        "oracle",
        "ndcg@10",
        "precision@10",
    ]:
        assert np.allclose(official[column], fast[column], atol=1e-9), column


def test_fast_equals_official_fixed_policy():
    users, n_items = make_users(), 40
    for warmup in (0, 5):
        for visible in (True, False):
            cfg = SessionConfig(horizon=20, warmup=warmup, group_visible=visible)
            official = evaluate(FixedPolicy(n_items), users, n_items, cfg, seeds=(0, 1))
            fast = evaluate_fast(
                FixedPolicy(n_items), users, n_items, cfg, seeds=(0, 1), batch_size=5
            )
            compare_results(official, fast)


def test_fast_equals_official_fallback_without_batch():
    users, n_items = make_users(), 40
    cfg = SessionConfig(horizon=20, warmup=5, group_visible=False)
    official = evaluate(FixedPolicyNoBatch(n_items), users, n_items, cfg, seeds=(0,))
    fast = evaluate_fast(FixedPolicyNoBatch(n_items), users, n_items, cfg, seeds=(0,), batch_size=4)
    compare_results(official, fast)
