"""
Value Iteration baseline for Q3 5x5 gridworld (used in Q4 comparison).

This version matches gridworld5x5.py API:
- States are (row, col) tuples
- env.step(s, a) returns (s_next, r, done)
- env.state_to_idx / env.idx_to_state convert between tuple <-> index
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .gridworld5x5 import GridWorld5x5


@dataclass
class ValueIterationResult:
    V: np.ndarray          # shape (S,)
    pi: np.ndarray         # shape (S,) action index per state
    iterations: int
    delta_history: list


def value_iteration(env: GridWorld5x5, theta: float = 1e-10, max_iters: int = 100_000) -> ValueIterationResult:
    """
    Standard (synchronous) value iteration.

    Complexity per sweep: O(|S| * |A|)
    Total: O(K * |S| * |A|) where K is number of sweeps to converge.
    """
    S = env.size * env.size
    A = env.n_actions

    V = np.zeros(S, dtype=float)
    delta_history = []

    for it in range(1, max_iters + 1):
        delta = 0.0
        V_new = V.copy()

        for s_idx in range(S):
            s = env.idx_to_state(s_idx)

            # Terminal state: absorbing, keep value 0 for stability
            if env.is_terminal(s):
                V_new[s_idx] = 0.0
                continue

            q_sa = np.zeros(A, dtype=float)
            for a in range(A):
                s2, r, done = env.step(s, a)
                s2_idx = env.state_to_idx(s2)
                q_sa[a] = r + env.gamma * (0.0 if done else V[s2_idx])

            V_new[s_idx] = float(np.max(q_sa))
            delta = max(delta, abs(V_new[s_idx] - V[s_idx]))

        V = V_new
        delta_history.append(delta)

        if delta < theta:
            break

    # Derive greedy policy from converged V
    pi = np.zeros(S, dtype=int)
    for s_idx in range(S):
        s = env.idx_to_state(s_idx)
        if env.is_terminal(s):
            pi[s_idx] = 0
            continue

        q_sa = np.zeros(A, dtype=float)
        for a in range(A):
            s2, r, done = env.step(s, a)
            s2_idx = env.state_to_idx(s2)
            q_sa[a] = r + env.gamma * (0.0 if done else V[s2_idx])

        pi[s_idx] = int(np.argmax(q_sa))

    return ValueIterationResult(V=V, pi=pi, iterations=len(delta_history), delta_history=delta_history)


def format_V_grid(env: GridWorld5x5, V: np.ndarray, decimals: int = 2) -> str:
    """Pretty-print values in 5x5 grid layout."""
    grid = np.zeros((env.size, env.size), dtype=float)
    for r in range(env.size):
        for c in range(env.size):
            grid[r, c] = V[env.state_to_idx((r, c))]

    out_lines = []
    for r in range(env.size):
        row_parts = []
        for c in range(env.size):
            if (r, c) == env.goal:
                row_parts.append("   G   ")
            else:
                row_parts.append(f"{grid[r,c]:{6}.{decimals}f}")
        out_lines.append(" ".join(row_parts))
    return "\n".join(out_lines)


def format_pi_grid(env: GridWorld5x5, pi: np.ndarray) -> str:
    """Pretty-print policy arrows in 5x5 grid layout."""
    arrows = {0: "→", 1: "←", 2: "↓", 3: "↑"}
    out_lines = []
    for r in range(env.size):
        row_parts = []
        for c in range(env.size):
            if (r, c) == env.goal:
                row_parts.append(" G ")
            else:
                row_parts.append(f" {arrows[int(pi[env.state_to_idx((r,c))])]} ")
        out_lines.append("".join(row_parts))
    return "\n".join(out_lines)


if __name__ == "__main__":
    env = GridWorld5x5(gamma=0.9)
    res = value_iteration(env)
    print("=== Value Iteration (Q3 baseline) ===")
    print(f"Iterations: {res.iterations}")
    print("\nV*:")
    print(format_V_grid(env, res.V))
    print("\npi*:")
    print(format_pi_grid(env, res.pi))
