"""Evaluación batcheada del protocolo común.

Agrupa sesiones en lockstep y hace un forward por paso para todas las sesiones
del batch. Está pensada para políticas determinísticas que implementan
`scores_batch` (DT e IQL). Para políticas sin ese método delega en el
`run_session` oficial, así que el resultado es idéntico al de
`protocolo.evaluar.evaluate`.

Las historias de todas las sesiones de un batch tienen el mismo largo porque
comparten horizonte y warm-up. Cada sesión conserva su propio generador
aleatorio, así que los desempates y el warm-up son los mismos que en la
evaluación oficial.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from protocolo.datos import RELEVANCE_THRESHOLD, User
from protocolo.evaluar import run_session
from protocolo.metricas import normalized_score, ranking_metrics, session_bounds
from protocolo.simulador import Session, SessionConfig, choose


def evaluate_fast(
    policy,
    users: list[User],
    n_items: int,
    config: SessionConfig,
    seeds: Iterable[int] = (0,),
    batch_size: int = 512,
) -> pd.DataFrame:
    """Equivalente a `protocolo.evaluar.evaluate`, batcheando los forwards."""
    if batch_size <= 0:
        raise ValueError("batch_size debe ser positivo")
    has_batch = hasattr(policy, "scores_batch")
    rows: list[dict[str, float]] = []

    for seed in seeds:
        for start in range(0, len(users), batch_size):
            chunk = users[start : start + batch_size]
            if not has_batch:
                for index, user in enumerate(chunk):
                    rows.append(run_session(policy, user, n_items, config, seed, start + index))
                continue

            rngs = [np.random.default_rng([seed, start + index]) for index in range(len(chunk))]
            sessions = [Session(user, n_items, config, rng) for user, rng in zip(chunk, rngs)]
            for session in sessions:
                policy.reset(session.visible_group, config.horizon)

            bounds = [
                session_bounds(session._available_hidden_ratings(), config.controlled_steps)
                for session in sessions
            ]
            groups = [session.visible_group for session in sessions]
            controlled: list[list[float]] = [[] for _ in sessions]
            first_ranking: list[dict[str, float] | None] = [None for _ in sessions]

            for _ in range(config.controlled_steps):
                observed = [session.observed() for session in sessions]
                items_seen = [item for item, _ in observed]
                ratings_seen = [rating for _, rating in observed]
                scores = np.asarray(
                    policy.scores_batch(items_seen, ratings_seen, groups), dtype=np.float64
                )

                for index, session in enumerate(sessions):
                    mask = session.available_mask()
                    if first_ranking[index] is None:
                        first_ranking[index] = ranking_metrics(
                            scores[index], session._hidden_ratings(), mask, k=10
                        )
                    controlled[index].append(session.step(choose(scores[index], mask, rngs[index])))

            for index, (session, user) in enumerate(zip(sessions, chunk)):
                total = float(np.sum(controlled[index]))
                random_expected, oracle = bounds[index]
                ranking = first_ranking[index]
                assert ranking is not None
                rows.append(
                    {
                        "user_id": user.user_id,
                        "group": user.group,
                        "seed": seed,
                        "return": total,
                        "full_return": float(np.sum(session.ratings)),
                        "mean_rating": total / config.controlled_steps,
                        "hit_rate": float(
                            np.mean(np.asarray(controlled[index]) >= RELEVANCE_THRESHOLD)
                        ),
                        "normalized": normalized_score(total, random_expected, oracle),
                        "random_expected": random_expected,
                        "oracle": oracle,
                        **ranking,
                    }
                )

    return pd.DataFrame(rows)
