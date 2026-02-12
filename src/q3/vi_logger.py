#!/usr/bin/python3
"""Q3 Value Iteration Logger (lec3-compatible).

Creates two files under ./logs/q3/ (by default):
  1) <prefix>_value_iteration_<timestamp>.log  (readable snapshots)
  2) <prefix>_snapshots_<timestamp>.csv        (numeric V values)

This logger is adapted from your earlier Q3 logging approach, but trimmed to work
with the lec3 GridWorld/Agent files.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np


class VILogger:
    """Writes snapshots for Value Iteration runs."""

    ARROW = {"Right": "→", "Left": "←", "Down": "↓", "Up": "↑"}

    def __init__(self, env, log_dir: str = "logs/q3", prefix: str = "q3"):
        self.env = env
        self.env_size = env.get_size()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{prefix}_value_iteration_{ts}.log"
        self.csv_path = self.log_dir / f"{prefix}_snapshots_{ts}.csv"

        # Header
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("Q3 Value Iteration Log (lec3)\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n")
            if hasattr(env, "terminal_state"):
                f.write(f"Goal state: s_{{{env.terminal_state[0]},{env.terminal_state[1]}}}\n")
            if hasattr(env, "grey_states"):
                gs = sorted(list(env.grey_states))
                f.write("Grey states: " + ", ".join([f"s_{{{i},{j}}}" for (i, j) in gs]) + "\n")
            f.write("Note: reward-on-arrival is used (r = R(s')).\n\n")

        # CSV header
        with open(self.csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            header = ["k", "delta_max"] + [
                f"V_{i}_{j}" for i in range(self.env_size) for j in range(self.env_size)
            ]
            writer.writerow(header)

    def _format_V(self, V: np.ndarray, decimals: int = 2) -> str:
        N = self.env_size
        lines: List[str] = []
        grey_set = set(getattr(self.env, "grey_states", []))
        goal = getattr(self.env, "terminal_state", None)

        for i in range(N):
            row: List[str] = []
            for j in range(N):
                s = (i, j)
                if goal is not None and s == goal:
                    row.append("   G   ")
                elif s in grey_set:
                    row.append(f"{V[i, j]:6.{decimals}f}*")
                else:
                    row.append(f"{V[i, j]:6.{decimals}f} ")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def _format_pi(self, pi_str_grid: List[List[str]]) -> str:
        lines: List[str] = []
        for row in pi_str_grid:
            out: List[str] = []
            for cell in row:
                if cell == "X":
                    out.append("  X ")
                else:
                    parts = cell.split("|")
                    arrows = "".join(self.ARROW.get(p.strip(), "?") for p in parts)
                    out.append(f"{arrows:^3}")
            lines.append(" ".join(out))
        return "\n".join(lines)

    def snapshot(
        self,
        k: int,
        delta_max: float,
        V: np.ndarray,
        pi_str: Optional[List[List[str]]] = None,
        decimals: int = 2,
    ) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"k={k}  delta_max={delta_max:.10f}\n")
            f.write("V_k:\n")
            f.write(self._format_V(V, decimals=decimals))
            f.write("\n")
            if pi_str is not None:
                f.write("Greedy policy π_k (w.r.t. V_k):\n")
                f.write(self._format_pi(pi_str))
                f.write("\n")
            f.write("\n" + ("-" * 60) + "\n\n")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(
                [k, delta_max] + [float(V[i, j]) for i in range(self.env_size) for j in range(self.env_size)]
            )

    def converged(
        self,
        k: int,
        delta_max: float,
        V: np.ndarray,
        pi_str: Optional[List[List[str]]] = None,
        decimals: int = 2,
    ) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"CONVERGED at k={k} (delta_max={delta_max:.10f})\n\n")
            f.write("Final V*:\n")
            f.write(self._format_V(V, decimals=decimals))
            f.write("\n")
            if pi_str is not None:
                f.write("\nFinal π*:\n")
                f.write(self._format_pi(pi_str))
                f.write("\n")
