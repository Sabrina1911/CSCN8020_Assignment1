"""
Q3: 5x5 Gridworld — Value Iteration (γ = 0.99)
Task 1: reward list + standard (synchronous) value iteration
Task 2: in-place value iteration + performance comparison + confirm same V* and π*

Outputs saved under ./logs/q3/ by default.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

State = Tuple[int, int]  # (i, j)


# ----------------------------
# 1) Configuration
# ----------------------------
@dataclass(frozen=True)
class GridworldConfig:
    N: int = 5
    gamma: float = 0.99
    theta: float = 1e-8
    max_iters: int = 200000

    goal: State = (4, 4)
    grey: Tuple[State, ...] = ((2, 2), (3, 0), (0, 4))

    # a1=Right, a2=Down, a3=Left, a4=Up
    actions: Tuple[int, int, int, int] = (1, 2, 3, 4)


# ----------------------------
# 2) MDP
# ----------------------------
class GridworldMDP:
    """
    Deterministic 5x5 gridworld MDP.

    Reward function R(s):
      +10 if s is goal
      -5  if s is grey
      -1  otherwise

    Reward-on-arrival is used in VI: r = R(s') where s' = δ(s,a)
    """

    def __init__(self, cfg: GridworldConfig):
        self.cfg = cfg

        # action deltas: a -> (di, dj)
        self.A: Dict[int, Tuple[int, int]] = {
            1: (0, +1),   # Right
            2: (+1, 0),   # Down
            3: (0, -1),   # Left
            4: (-1, 0),   # Up
        }

        self.ARROW = {1: "→", 2: "↓", 3: "←", 4: "↑"}

        self.S_goal: State = cfg.goal
        self.S_grey: Set[State] = set(cfg.grey)

        # Task 1 requirement: reward function as a LIST (length N*N)
        self.R_list: List[int] = self._build_reward_list()

    # ---- helpers ----
    def idx(self, s: State) -> int:
        i, j = s
        return self.cfg.N * i + j

    def in_bounds(self, s: State) -> bool:
        i, j = s
        return 0 <= i < self.cfg.N and 0 <= j < self.cfg.N

    # ---- transition δ(s,a) ----
    def delta(self, s: State, a: int) -> State:
        di, dj = self.A[a]
        i, j = s
        s_next = (i + di, j + dj)
        return s_next if self.in_bounds(s_next) else s

    # ---- reward R(s) ----
    def _build_reward_list(self) -> List[int]:
        N = self.cfg.N
        R = [-1] * (N * N)
        for s in self.S_grey:
            R[self.idx(s)] = -5
        R[self.idx(self.S_goal)] = 10
        return R

    def R(self, s: State) -> int:
        return self.R_list[self.idx(s)]

    # ---- formatting ----
    def format_V(self, V: np.ndarray, decimals: int = 2) -> str:
        N = self.cfg.N
        lines: List[str] = []
        for i in range(N):
            row: List[str] = []
            for j in range(N):
                s = (i, j)
                if s == self.S_goal:
                    row.append("   G   ")
                elif s in self.S_grey:
                    row.append(f"{V[i, j]:6.{decimals}f}*")
                else:
                    row.append(f"{V[i, j]:6.{decimals}f} ")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def format_pi(self, pi: np.ndarray) -> str:
        N = self.cfg.N
        lines: List[str] = []
        for i in range(N):
            row: List[str] = []
            for j in range(N):
                s = (i, j)
                if s == self.S_goal:
                    row.append(" G ")
                else:
                    row.append(f" {self.ARROW[int(pi[i, j])]} ")
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ---- greedy policy extraction ----
    def extract_greedy_policy(self, V: np.ndarray) -> np.ndarray:
        N = self.cfg.N
        gamma = self.cfg.gamma
        pi = np.zeros((N, N), dtype=int)

        for i in range(N):
            for j in range(N):
                s = (i, j)
                if s == self.S_goal:
                    pi[i, j] = 0
                    continue

                best_a = self.cfg.actions[0]
                best_q = -1e18
                for a in self.cfg.actions:
                    s_next = self.delta(s, a)
                    q = self.R(s_next) + gamma * V[s_next]
                    if q > best_q:
                        best_q = q
                        best_a = a
                pi[i, j] = best_a
        return pi

    # ----------------------------
    # Standard (synchronous) Value Iteration
    # ----------------------------
    def value_iteration(
        self,
        logger: Optional["VILogger"] = None,
        snapshot_every: int = 50,
        snapshot_iters: Sequence[int] = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
        log_policy: bool = True,
        snapshot_k0: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Returns: (V*, pi*, iters, elapsed_seconds)
        """
        N = self.cfg.N
        gamma = self.cfg.gamma
        theta = self.cfg.theta

        V = np.zeros((N, N), dtype=float)

        def should_snapshot(k: int) -> bool:
            if snapshot_every > 0 and (k % snapshot_every == 0):
                return True
            return k in set(snapshot_iters)

        start = time.perf_counter()

        if logger and snapshot_k0:
            pi_0 = self.extract_greedy_policy(V) if log_policy else None
            logger.snapshot(k=0, delta_max=0.0, V=V, pi=pi_0)

        for k in range(1, self.cfg.max_iters + 1):
            delta_max = 0.0
            V_new = V.copy()

            for i in range(N):
                for j in range(N):
                    s = (i, j)
                    if s == self.S_goal:
                        continue

                    best_q = -1e18
                    for a in self.cfg.actions:
                        s_next = self.delta(s, a)
                        q = self.R(s_next) + gamma * V[s_next]
                        if q > best_q:
                            best_q = q

                    V_new[s] = best_q
                    delta_max = max(delta_max, abs(V_new[s] - V[s]))

            V = V_new

            if logger and should_snapshot(k):
                pi_k = self.extract_greedy_policy(V) if log_policy else None
                logger.snapshot(k=k, delta_max=delta_max, V=V, pi=pi_k)

            if delta_max < theta:
                elapsed = time.perf_counter() - start
                pi_star = self.extract_greedy_policy(V)
                if logger:
                    logger.converged(k=k, delta_max=delta_max, V=V, pi=pi_star)
                return V, pi_star, k, elapsed

        elapsed = time.perf_counter() - start
        pi_star = self.extract_greedy_policy(V)
        if logger:
            logger.converged(k=self.cfg.max_iters, delta_max=float("nan"), V=V, pi=pi_star)
        return V, pi_star, self.cfg.max_iters, elapsed

    # ----------------------------
    # Task 2: In-Place Value Iteration
    # ----------------------------
    def value_iteration_in_place(
        self,
        logger: Optional["VILogger"] = None,
        snapshot_every: int = 50,
        snapshot_iters: Sequence[int] = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
        log_policy: bool = True,
        snapshot_k0: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        In-Place Value Iteration:
        updates V(s) directly and immediately uses updated values in the same sweep.

        Returns: (V*, pi*, iters, elapsed_seconds)
        """
        N = self.cfg.N
        gamma = self.cfg.gamma
        theta = self.cfg.theta

        V = np.zeros((N, N), dtype=float)

        def should_snapshot(k: int) -> bool:
            if snapshot_every > 0 and (k % snapshot_every == 0):
                return True
            return k in set(snapshot_iters)

        start = time.perf_counter()

        if logger and snapshot_k0:
            pi_0 = self.extract_greedy_policy(V) if log_policy else None
            logger.snapshot(k=0, delta_max=0.0, V=V, pi=pi_0)

        for k in range(1, self.cfg.max_iters + 1):
            delta_max = 0.0

            for i in range(N):
                for j in range(N):
                    s = (i, j)
                    if s == self.S_goal:
                        continue

                    old = V[s]

                    best_q = -1e18
                    for a in self.cfg.actions:
                        s_next = self.delta(s, a)
                        q = self.R(s_next) + gamma * V[s_next]  # uses latest V (in-place)
                        if q > best_q:
                            best_q = q

                    V[s] = best_q
                    delta_max = max(delta_max, abs(V[s] - old))

            if logger and should_snapshot(k):
                pi_k = self.extract_greedy_policy(V) if log_policy else None
                logger.snapshot(k=k, delta_max=delta_max, V=V, pi=pi_k)

            if delta_max < theta:
                elapsed = time.perf_counter() - start
                pi_star = self.extract_greedy_policy(V)
                if logger:
                    logger.converged(k=k, delta_max=delta_max, V=V, pi=pi_star)
                return V, pi_star, k, elapsed

        elapsed = time.perf_counter() - start
        pi_star = self.extract_greedy_policy(V)
        if logger:
            logger.converged(k=self.cfg.max_iters, delta_max=float("nan"), V=V, pi=pi_star)
        return V, pi_star, self.cfg.max_iters, elapsed


# ----------------------------
# 3) Logger (logs/q3/)
# ----------------------------
class VILogger:
    """Writes a readable .log and numeric .csv under logs/q3/"""

    def __init__(self, mdp: GridworldMDP, log_dir: str = "logs/q3", prefix: str = "q3"):
        self.mdp = mdp
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{prefix}_value_iteration_{ts}.log"
        self.csv_path = self.log_dir / f"{prefix}_snapshots_{ts}.csv"

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("Q3 Value Iteration Log\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n")
            f.write(f"N={mdp.cfg.N}, gamma={mdp.cfg.gamma}, theta={mdp.cfg.theta}, max_iters={mdp.cfg.max_iters}\n")
            f.write(f"Goal state: s_{{{mdp.cfg.goal[0]},{mdp.cfg.goal[1]}}}\n")
            f.write("Grey states: " + ", ".join([f"s_{{{i},{j}}}" for (i, j) in sorted(mdp.S_grey)]) + "\n")
            f.write("Note: reward-on-arrival is used (r = R(s')).\n\n")

        with open(self.csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            header = ["k", "delta_max"] + [f"V_{i}_{j}" for i in range(mdp.cfg.N) for j in range(mdp.cfg.N)]
            writer.writerow(header)

    def snapshot(self, k: int, delta_max: float, V: np.ndarray, pi: Optional[np.ndarray] = None) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"k={k}  delta_max={delta_max:.10f}\n")
            f.write("V_k:\n")
            f.write(self.mdp.format_V(V, decimals=2))
            f.write("\n")
            if pi is not None:
                f.write("Greedy policy π_k (w.r.t. V_k):\n")
                f.write(self.mdp.format_pi(pi))
                f.write("\n")
            f.write("\n" + ("-" * 60) + "\n\n")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(
                [k, delta_max] + [float(V[i, j]) for i in range(self.mdp.cfg.N) for j in range(self.mdp.cfg.N)]
            )

    def converged(self, k: int, delta_max: float, V: np.ndarray, pi: np.ndarray) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"CONVERGED at k={k} (delta_max={delta_max:.10f} < theta)\n\n")
            f.write("Final V*:\n")
            f.write(self.mdp.format_V(V, decimals=2))
            f.write("\n\nFinal π*:\n")
            f.write(self.mdp.format_pi(pi))
            f.write("\n")


# ----------------------------
# 4) Runners (notebook-friendly)
# ----------------------------
def run_q3_value_iteration(
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iters: int = 200000,
    snapshot_every: int = 50,
    log_policy: bool = True,
    snapshot_k0: bool = True,
    snapshot_iters: Sequence[int] = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000),
    log_dir: str = "logs/q3",
    prefix: str = "q3",
) -> Tuple[GridworldMDP, np.ndarray, np.ndarray, int, float, str, str]:
    cfg = GridworldConfig(gamma=gamma, theta=theta, max_iters=max_iters)
    mdp = GridworldMDP(cfg)
    logger = VILogger(mdp, log_dir=log_dir, prefix=prefix)

    V_star, pi_star, iters, elapsed = mdp.value_iteration(
        logger=logger,
        snapshot_every=snapshot_every,
        snapshot_iters=snapshot_iters,
        log_policy=log_policy,
        snapshot_k0=snapshot_k0,
    )

    return mdp, V_star, pi_star, iters, elapsed, str(logger.log_path), str(logger.csv_path)


def run_q3_variations(
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iters: int = 200000,
    snapshot_every: int = 50,
) -> dict:
    cfg = GridworldConfig(gamma=gamma, theta=theta, max_iters=max_iters)
    mdp = GridworldMDP(cfg)

    # Standard VI
    logger_std = VILogger(mdp, log_dir="logs/q3", prefix="q3_std")
    V_std, pi_std, it_std, t_std = mdp.value_iteration(logger=logger_std, snapshot_every=snapshot_every)

    # In-place VI
    logger_ip = VILogger(mdp, log_dir="logs/q3", prefix="q3_inplace")
    V_ip, pi_ip, it_ip, t_ip = mdp.value_iteration_in_place(logger=logger_ip, snapshot_every=snapshot_every)

    same_V = np.allclose(V_std, V_ip, atol=1e-6)
    same_pi = np.array_equal(pi_std, pi_ip)

    return {
        "mdp": mdp,
        "V_std": V_std, "pi_std": pi_std, "iters_std": it_std, "time_std": t_std,
        "V_ip": V_ip, "pi_ip": pi_ip, "iters_ip": it_ip, "time_ip": t_ip,
        "same_V": same_V, "same_pi": same_pi,
        "log_std": str(logger_std.log_path), "csv_std": str(logger_std.csv_path),
        "log_ip": str(logger_ip.log_path), "csv_ip": str(logger_ip.csv_path),
    }


# ----------------------------
# 5) CLI run
# ----------------------------
if __name__ == "__main__":
    out = run_q3_variations(gamma=0.99, theta=1e-8, max_iters=200000, snapshot_every=50)
    mdp = out["mdp"]

    print("Same V* (allclose)?", out["same_V"])
    print("Same π* (exact)?  ", out["same_pi"])

    print("\n--- Standard VI ---")
    print("Iterations:", out["iters_std"], " Time(s):", round(out["time_std"], 6))
    print(mdp.format_V(out["V_std"], decimals=2))
    print("\nπ*:\n")
    print(mdp.format_pi(out["pi_std"]))

    print("\n--- In-Place VI ---")
    print("Iterations:", out["iters_ip"], " Time(s):", round(out["time_ip"], 6))
    print(mdp.format_V(out["V_ip"], decimals=2))
    print("\nπ*:\n")
    print(mdp.format_pi(out["pi_ip"]))

    print("\nLogs:")
    print("STD log:", out["log_std"])
    print("INP log:", out["log_ip"])
