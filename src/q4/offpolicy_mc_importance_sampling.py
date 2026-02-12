"""
Q4: Off-policy Monte Carlo Control with Importance Sampling (Weighted IS)

- Behavior policy b(a|s): uniform random (1/4)
- Target policy pi(a|s): greedy w.r.t Q
- Update: Weighted Importance Sampling (stable incremental average)

Adds:
- Optional weight clipping (numeric stability)
- Optional progress logging every N episodes (checkpoint style)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from .gridworld5x5 import GridWorld5x5, State


@dataclass
class MCControlResult:
    Q: np.ndarray
    V: np.ndarray
    pi: np.ndarray
    episodes: int
    steps_collected: int
    seed: int


def generate_episode(
    env: GridWorld5x5,
    rng: np.random.Generator,
    max_steps_per_episode: int = 200,
) -> List[Tuple[int, int, float]]:
    """Generate one episode using behavior policy b (uniform random). Returns [(s_idx,a,r), ...]."""
    # sample a non-terminal start state
    while True:
        s0 = (int(rng.integers(0, env.size)), int(rng.integers(0, env.size)))
        if not env.is_terminal(s0):
            break

    episode: List[Tuple[int, int, float]] = []
    s: State = s0

    for _ in range(max_steps_per_episode):
        a = int(rng.integers(0, env.n_actions))  # uniform random
        s2, r, done = env.step(s, a)

        episode.append((env.state_to_idx(s), a, float(r)))

        s = s2
        if done:
            break

    return episode


def off_policy_mc_control_weighted_is(
    env: GridWorld5x5,
    num_episodes: int = 100_000,
    seed: int = 7,
    max_steps_per_episode: int = 200,
    weight_clip: float | None = 1e6,  # set None to disable clipping
    log_every: int | None = 10_000,   # set None or 0 to disable logging
    logger: Callable[[str], None] | None = None,
) -> MCControlResult:
    """
    Off-policy MC control using weighted importance sampling.

    log_every/logger:
      - If logger is provided and log_every is a positive int, we write a progress line every log_every episodes.
      - We log: episodes done, steps collected, and current max|deltaV| vs last checkpoint (simple stability signal).
    """
    rng = np.random.default_rng(seed)

    S = env.size * env.size
    A = env.n_actions
    b_prob = 1.0 / A  # uniform behavior

    Q = np.zeros((S, A), dtype=float)
    C = np.zeros((S, A), dtype=float)

    pi = np.zeros(S, dtype=int)
    pi[env.state_to_idx(env.goal)] = 0

    total_steps = 0

    # For checkpoint diagnostics
    prev_V_checkpoint: np.ndarray | None = None

    def should_log(ep_idx_0_based: int) -> bool:
        if logger is None:
            return False
        if log_every is None or log_every <= 0:
            return False
        return (ep_idx_0_based + 1) % log_every == 0

    for ep in range(num_episodes):
        episode = generate_episode(env, rng, max_steps_per_episode=max_steps_per_episode)
        total_steps += len(episode)

        G = 0.0
        W = 1.0

        for (s_idx, a, r) in reversed(episode):
            G = env.gamma * G + r

            C[s_idx, a] += W
            Q[s_idx, a] += (W / C[s_idx, a]) * (G - Q[s_idx, a])

            pi[s_idx] = int(np.argmax(Q[s_idx, :]))

            if a != pi[s_idx]:
                break

            W *= 1.0 / b_prob

            if weight_clip is not None and W > weight_clip:
                W = weight_clip

        # checkpoint logging (compact)
        if should_log(ep):
            V_now = np.max(Q, axis=1)
            V_now[env.state_to_idx(env.goal)] = 0.0

            if prev_V_checkpoint is None:
                max_dV = float(np.max(np.abs(V_now)))
            else:
                max_dV = float(np.max(np.abs(V_now - prev_V_checkpoint)))

            prev_V_checkpoint = V_now.copy()

            logger(
                f"MC checkpoint @ episode {ep+1:,}: steps_so_far={total_steps:,}, "
                f"max|ΔV| since last checkpoint={max_dV:.6f}"
            )

    V = np.max(Q, axis=1)
    V[env.state_to_idx(env.goal)] = 0.0  # terminal displayed as absorbing

    return MCControlResult(Q=Q, V=V, pi=pi, episodes=num_episodes, steps_collected=total_steps, seed=seed)


def format_V_grid(env: GridWorld5x5, V: np.ndarray, decimals: int = 2) -> str:
    grid = V.reshape(env.size, env.size)
    lines = []
    for r in range(env.size):
        row = []
        for c in range(env.size):
            if (r, c) == env.goal:
                row.append("   G   ")
            else:
                row.append(f"{grid[r, c]:6.{decimals}f}")
        lines.append(" ".join(row))
    return "\n".join(lines)


def format_pi_grid(env: GridWorld5x5, pi: np.ndarray) -> str:
    arrows = {0: "→", 1: "←", 2: "↓", 3: "↑"}
    pi_grid = pi.reshape(env.size, env.size)
    lines = []
    for r in range(env.size):
        row = []
        for c in range(env.size):
            if (r, c) == env.goal:
                row.append(" G ")
            else:
                row.append(f" {arrows[int(pi_grid[r, c])]} ")
        lines.append("".join(row))
    return "\n".join(lines)
