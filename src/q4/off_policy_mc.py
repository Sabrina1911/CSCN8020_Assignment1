"""
Q4: 5x5 Gridworld — Off-policy Monte Carlo with Importance Sampling (γ = 0.9)

Environment:
- Same 5x5 gridworld as Q3 (same goal, grey states, reward list)
- Reward-on-arrival convention: take action a in state s, land in s', receive r = R(s')

Policies:
- Behavior policy b(a|s): uniform random over actions
- Target policy π(a|s): greedy with respect to the current estimate Q(s,a)

Algorithms implemented:
1) Ordinary Importance Sampling (OIS)
2) Weighted Importance Sampling (WIS)
3) Value Iteration baseline (VI) for comparison

Important note (matches what we observed in the run):
- VI converges quickly for this small deterministic grid (few sweeps over all states).
- WIS is stable and closely matches VI because it normalizes importance weights using C(s,a).
- OIS can show very large spikes (high variance) because raw importance ratios can grow rapidly
  when the behavior policy is random (b = 0.25 per action) and the target policy is greedy.

Logging:
- Logs and CSV snapshots are saved under ./logs/q4/ by default.
- Each MC log includes the VI baseline at the top (V*, π*) for self-contained comparison.
- Snapshots store V_k matrix and greedy policy derived from Q_k, similar to the Q3 formatting style.
- WIS log also includes an evidence line: max |V_WIS - V_VI|.
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
    gamma: float = 0.9

    # Match Q3 grid
    goal: State = (4, 4)
    grey: Tuple[State, ...] = ((2, 2), (3, 0), (0, 4))

    # Action encoding consistent with Q3:
    # a1=Right, a2=Down, a3=Left, a4=Up
    actions: Tuple[int, int, int, int] = (1, 2, 3, 4)

    # MC settings
    episodes: int = 50_000
    max_steps: int = 200

    # Logging snapshots (episode indices)
    snapshot_episodes: Tuple[int, ...] = (0, 1, 2, 3, 10, 100, 1000, 5000, 10000, 20000, 49999)

    # VI baseline settings
    theta: float = 1e-8
    max_iters: int = 200_000


# ----------------------------
# 2) MDP
# ----------------------------
class GridworldMDP:
    """
    Deterministic 5x5 gridworld MDP (same as Q3).

    Reward function R(s):
      +10 if s is goal
      -5  if s is grey
      -1  otherwise

    Reward-on-arrival:
      s_next = δ(s,a)
      r = R(s_next)
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

        # Reward list (Task-style, like Q3)
        self.R_list: List[int] = self._build_reward_list()

    # ---- helpers ----
    def idx(self, s: State) -> int:
        i, j = s
        return self.cfg.N * i + j

    def in_bounds(self, s: State) -> bool:
        i, j = s
        return 0 <= i < self.cfg.N and 0 <= j < self.cfg.N

    def is_goal(self, s: State) -> bool:
        return s == self.S_goal

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

    # ---- formatting (match Q3 style) ----
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
                    a = int(pi[i, j])
                    row.append(f" {self.ARROW[a]} ")
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ---- derive V from Q and greedy policy from Q ----
    def V_from_Q(self, Q: np.ndarray) -> np.ndarray:
        V = np.zeros((self.cfg.N, self.cfg.N), dtype=float)
        for i in range(self.cfg.N):
            for j in range(self.cfg.N):
                s = (i, j)
                if self.is_goal(s):
                    V[i, j] = 0.0
                else:
                    V[i, j] = float(np.max(Q[i, j, :]))
        return V

    def greedy_policy_from_Q(self, Q: np.ndarray) -> np.ndarray:
        pi = np.zeros((self.cfg.N, self.cfg.N), dtype=int)
        for i in range(self.cfg.N):
            for j in range(self.cfg.N):
                s = (i, j)
                if self.is_goal(s):
                    pi[i, j] = 1  # arbitrary; not used
                else:
                    best_idx = int(np.argmax(Q[i, j, :]))
                    pi[i, j] = best_idx + 1
        return pi

    # ---- Value Iteration baseline (like Q3) ----
    def value_iteration(self) -> Tuple[np.ndarray, np.ndarray, int, float]:
        N = self.cfg.N
        gamma = self.cfg.gamma
        theta = self.cfg.theta

        V = np.zeros((N, N), dtype=float)

        start = time.perf_counter()
        for k in range(1, self.cfg.max_iters + 1):
            delta_max = 0.0
            V_new = V.copy()

            for i in range(N):
                for j in range(N):
                    s = (i, j)
                    if self.is_goal(s):
                        continue

                    # Bellman optimality backup
                    best_q = -1e18
                    for a in self.cfg.actions:
                        s_next = self.delta(s, a)
                        q = self.R(s_next) + gamma * V[s_next]
                        if q > best_q:
                            best_q = q

                    V_new[s] = best_q
                    delta_max = max(delta_max, abs(V_new[s] - V[s]))

            V = V_new
            if delta_max < theta:
                elapsed = time.perf_counter() - start
                pi_star = self._extract_greedy_policy_from_V(V)
                return V, pi_star, k, elapsed

        elapsed = time.perf_counter() - start
        pi_star = self._extract_greedy_policy_from_V(V)
        return V, pi_star, self.cfg.max_iters, elapsed

    def _extract_greedy_policy_from_V(self, V: np.ndarray) -> np.ndarray:
        N = self.cfg.N
        gamma = self.cfg.gamma
        pi = np.zeros((N, N), dtype=int)

        for i in range(N):
            for j in range(N):
                s = (i, j)
                if self.is_goal(s):
                    pi[i, j] = 1
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
# 3) Logger (logs/q4/)
# ----------------------------
class MCLogger:
    """Writes a readable .log and numeric .csv under logs/q4/"""

    def __init__(self, mdp: GridworldMDP, log_dir: str = "logs/q4", prefix: str = "q4"):
        self.mdp = mdp
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{prefix}_{ts}.log"
        self.csv_path = self.log_dir / f"{prefix}_{ts}.csv"

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("Q4 Off-policy Monte Carlo (Importance Sampling) Log\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n")
            f.write(f"N={mdp.cfg.N}, gamma={mdp.cfg.gamma}, episodes={mdp.cfg.episodes}, max_steps={mdp.cfg.max_steps}\n")
            f.write(f"Goal state: s_{{{mdp.cfg.goal[0]},{mdp.cfg.goal[1]}}}\n")
            f.write("Grey states: " + ", ".join([f"s_{{{i},{j}}}" for (i, j) in sorted(mdp.S_grey)]) + "\n")
            f.write("Reward-on-arrival: r = R(s_next)\n")
            f.write("Expected behavior: WIS stable (low variance), OIS can be unstable (high variance).\n\n")

        with open(self.csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            header = ["k", "elapsed_s"] + [f"V_{i}_{j}" for i in range(mdp.cfg.N) for j in range(mdp.cfg.N)]
            writer.writerow(header)

    def write_vi_baseline(self, V_star: np.ndarray, pi_star: np.ndarray, iters: int, time_s: float) -> None:
        """Write the Value Iteration baseline once at the top of the log (self-contained log)."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("=== Value Iteration Baseline (VI) ===\n")
            f.write(f"iters={iters}, time_s={time_s:.6f}\n")
            f.write("V*:\n")
            f.write(self.mdp.format_V(V_star, decimals=2))
            f.write("\n\npi*:\n")
            f.write(self.mdp.format_pi(pi_star))
            f.write("\n\n" + ("=" * 60) + "\n\n")

    def append_text(self, text: str) -> None:
        """Append a small text block to the log."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text)

    def snapshot(self, k: int, elapsed_s: float, Q: np.ndarray, tag: str = "") -> None:
        V = self.mdp.V_from_Q(Q)
        pi = self.mdp.greedy_policy_from_Q(Q)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"k={k}  elapsed_s={elapsed_s:.6f} {tag}\n")
            f.write("V_k:\n")
            f.write(self.mdp.format_V(V, decimals=2))
            f.write("\n")
            f.write("Greedy policy (w.r.t. Q_k):\n")
            f.write(self.mdp.format_pi(pi))
            f.write("\n")
            f.write("\n" + ("-" * 60) + "\n\n")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow([k, elapsed_s] + [float(V[i, j]) for i in range(self.mdp.cfg.N) for j in range(self.mdp.cfg.N)])


# ----------------------------
# 4) Off-policy MC Control (OIS + WIS)
# ----------------------------
class OffPolicyMCControl:
    """
    Off-policy Monte Carlo control using importance sampling.

    Behavior policy b(a|s):
      - uniform random (each action prob = 0.25)

    Target policy π(a|s):
      - greedy w.r.t. Q(s,a)

    Key observation:
      - Importance weight contains terms π(a|s) / b(a|s).
      - With a greedy target and random behavior, b(a|s)=0.25, so weights can grow like 4^t.
      - WIS reduces variance by normalizing with cumulative weights C(s,a).
      - OIS can be high variance and produce spikes with finite samples.
    """

    def __init__(self, mdp: GridworldMDP, seed: int = 0):
        self.mdp = mdp
        self.rng = np.random.default_rng(seed)

        # Q arrays: Q[i,j,action_index], action_index 0..3 maps to action id 1..4
        self.Q_ois = np.zeros((mdp.cfg.N, mdp.cfg.N, 4), dtype=float)
        self.Q_wis = np.zeros((mdp.cfg.N, mdp.cfg.N, 4), dtype=float)

        # WIS cumulative weights C(s,a)
        self.C_wis = np.zeros((mdp.cfg.N, mdp.cfg.N, 4), dtype=float)

        # OIS counters N(s,a) (used for incremental averaging)
        self.N_ois = np.zeros((mdp.cfg.N, mdp.cfg.N, 4), dtype=float)

    def _random_start(self) -> State:
        while True:
            i = int(self.rng.integers(0, self.mdp.cfg.N))
            j = int(self.rng.integers(0, self.mdp.cfg.N))
            s = (i, j)
            if not self.mdp.is_goal(s):
                return s

    def _b_action(self) -> int:
        return int(self.rng.integers(1, 5))

    @staticmethod
    def _b_prob() -> float:
        return 0.25

    def _greedy_action_from_Q(self, Q: np.ndarray, s: State) -> int:
        i, j = s
        best_idx = int(np.argmax(Q[i, j, :]))
        return best_idx + 1

    def _generate_episode(self) -> List[Tuple[State, int, int]]:
        """
        Generate one episode under behavior policy b:
          returns list of (s, a, r) where r = R(s_next)
        """
        ep: List[Tuple[State, int, int]] = []
        s = self._random_start()

        for _ in range(self.mdp.cfg.max_steps):
            a = self._b_action()
            s_next = self.mdp.delta(s, a)
            r = self.mdp.R(s_next)
            ep.append((s, a, r))
            s = s_next
            if self.mdp.is_goal(s):
                break

        return ep

    # ----------------------------
    # Weighted Importance Sampling (WIS)
    # ----------------------------
    def run_weighted_is(
        self,
        episodes: int,
        logger: Optional[MCLogger] = None,
        snapshot_episodes: Sequence[int] = (),
        tag: str = "[WIS]",
    ) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        snap_set = set(snapshot_episodes)

        if logger and 0 in snap_set:
            logger.snapshot(k=0, elapsed_s=0.0, Q=self.Q_wis, tag=tag)

        for k in range(episodes):
            ep = self._generate_episode()
            G = 0.0
            W = 1.0

            for t in reversed(range(len(ep))):
                s, a, r = ep[t]
                G = self.mdp.cfg.gamma * G + r

                i, j = s
                a_idx = a - 1

                self.C_wis[i, j, a_idx] += W
                self.Q_wis[i, j, a_idx] += (W / self.C_wis[i, j, a_idx]) * (G - self.Q_wis[i, j, a_idx])

                if a != self._greedy_action_from_Q(self.Q_wis, s):
                    break

                W = W / self._b_prob()

            if logger and k in snap_set:
                elapsed = time.perf_counter() - start
                logger.snapshot(k=k, elapsed_s=elapsed, Q=self.Q_wis, tag=tag)

        elapsed = time.perf_counter() - start
        return self.Q_wis, elapsed

    # ----------------------------
    # Ordinary Importance Sampling (OIS)
    # ----------------------------
    def run_ordinary_is(
        self,
        episodes: int,
        logger: Optional[MCLogger] = None,
        snapshot_episodes: Sequence[int] = (),
        tag: str = "[OIS]",
    ) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        snap_set = set(snapshot_episodes)

        if logger and 0 in snap_set:
            logger.snapshot(k=0, elapsed_s=0.0, Q=self.Q_ois, tag=tag)

        for k in range(episodes):
            ep = self._generate_episode()
            G = 0.0
            W = 1.0

            for t in reversed(range(len(ep))):
                s, a, r = ep[t]
                G = self.mdp.cfg.gamma * G + r

                i, j = s
                a_idx = a - 1

                self.N_ois[i, j, a_idx] += 1.0
                self.Q_ois[i, j, a_idx] += (W * G - self.Q_ois[i, j, a_idx]) / self.N_ois[i, j, a_idx]

                if a != self._greedy_action_from_Q(self.Q_ois, s):
                    break

                W = W / self._b_prob()

            if logger and k in snap_set:
                elapsed = time.perf_counter() - start
                logger.snapshot(k=k, elapsed_s=elapsed, Q=self.Q_ois, tag=tag)

        elapsed = time.perf_counter() - start
        return self.Q_ois, elapsed


# ----------------------------
# 5) Runner (CLI)
# ----------------------------
def run_q4_variations(
    episodes: int = 50_000,
    gamma: float = 0.9,
    max_steps: int = 200,
    snapshot_episodes: Sequence[int] = (0, 1, 2, 3, 10, 100, 1000, 5000, 10000, 20000, 49999),
    log_dir: str = "logs/q4",
) -> dict:
    cfg = GridworldConfig(gamma=gamma, episodes=episodes, max_steps=max_steps)
    mdp = GridworldMDP(cfg)

    # Baseline: Value Iteration
    V_star, pi_star, it_vi, t_vi = mdp.value_iteration()

    # Create loggers first (so we can write baseline into them)
    logger_ois = MCLogger(mdp, log_dir=log_dir, prefix="q4_ordinaryIS")
    logger_wis = MCLogger(mdp, log_dir=log_dir, prefix="q4_weightedIS")

    # Write VI baseline once at top of each MC log (self-contained logs)
    logger_ois.write_vi_baseline(V_star, pi_star, it_vi, t_vi)
    logger_wis.write_vi_baseline(V_star, pi_star, it_vi, t_vi)

    # Off-policy MC Control
    mc = OffPolicyMCControl(mdp, seed=0)

    Q_ois, t_ois = mc.run_ordinary_is(
        episodes=cfg.episodes, logger=logger_ois, snapshot_episodes=snapshot_episodes, tag="[OIS]"
    )
    Q_wis, t_wis = mc.run_weighted_is(
        episodes=cfg.episodes, logger=logger_wis, snapshot_episodes=snapshot_episodes, tag="[WIS]"
    )

    V_ois = mdp.V_from_Q(Q_ois)
    pi_ois = mdp.greedy_policy_from_Q(Q_ois)

    V_wis = mdp.V_from_Q(Q_wis)
    pi_wis = mdp.greedy_policy_from_Q(Q_wis)

    # Evidence line: how close WIS is to VI
    max_diff_wis_vi = float(np.max(np.abs(V_wis - V_star)))
    logger_wis.append_text("=== Evidence (WIS vs VI) ===\n")
    logger_wis.append_text(f"max |V_WIS - V_VI| = {max_diff_wis_vi:.6f}\n\n")

    return {
        "mdp": mdp,
        "VI": {"V": V_star, "pi": pi_star, "iters": it_vi, "time": t_vi},
        "OIS": {
            "Q": Q_ois, "V": V_ois, "pi": pi_ois, "time": t_ois,
            "log": str(logger_ois.log_path), "csv": str(logger_ois.csv_path)
        },
        "WIS": {
            "Q": Q_wis, "V": V_wis, "pi": pi_wis, "time": t_wis,
            "log": str(logger_wis.log_path), "csv": str(logger_wis.csv_path),
            "max_diff_wis_vi": max_diff_wis_vi
        },
    }


if __name__ == "__main__":
    out = run_q4_variations(
        episodes=50_000,
        gamma=0.9,
        max_steps=200,
        snapshot_episodes=(0, 1, 2, 3, 10, 100, 1000, 5000, 10000, 20000, 49999),
        log_dir="logs/q4",
    )

    mdp = out["mdp"]

    print("\n=== Value Iteration Baseline ===")
    print("iters:", out["VI"]["iters"], "time(s):", round(out["VI"]["time"], 4))
    print(mdp.format_V(out["VI"]["V"], decimals=2))
    print("\nPolicy:")
    print(mdp.format_pi(out["VI"]["pi"]))

    print("\n=== Off-policy MC (Ordinary IS) ===")
    print("time(s):", round(out["OIS"]["time"], 4))
    print(mdp.format_V(out["OIS"]["V"], decimals=2))
    print("\nPolicy:")
    print(mdp.format_pi(out["OIS"]["pi"]))
    print("\nlogs:", out["OIS"]["log"])
    print("csv :", out["OIS"]["csv"])

    print("\n=== Off-policy MC (Weighted IS) ===")
    print("time(s):", round(out["WIS"]["time"], 4))
    print(mdp.format_V(out["WIS"]["V"], decimals=2))
    print("\nPolicy:")
    print(mdp.format_pi(out["WIS"]["pi"]))
    print("\nlogs:", out["WIS"]["log"])
    print("csv :", out["WIS"]["csv"])

    # Evidence line: how close WIS is to VI
    print("\n=== Evidence (WIS vs VI) ===")
    print(f"max |V_WIS - V_VI| = {out['WIS']['max_diff_wis_vi']:.4f}")

    print("\n=== Interpretation (what this shows) ===")
    print("- VI is fast because it updates all states directly using a Bellman backup.")
    print("- WIS is stable and matches VI closely because it normalizes weights using C(s,a).")
    print("- OIS can be unstable (high variance) because raw importance weights can grow quickly under random behavior.")
