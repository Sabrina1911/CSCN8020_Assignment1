"""
5x5 Gridworld (same as Q3) for CSCN8020 Assignment 1.

Environment definition (from Q3 statement):
- Grid: 5x5
- Terminal/goal state: s_goal = (4,4). Episode ends if agent reaches this state.
- Grey (unfavourable) states: S_grey = {(2,2), (3,0), (0,4)}
- Actions: Right, Left, Down, Up (deterministic transitions)
- Transitions: If action is valid, move; otherwise s' = s
- Rewards R(s):
    +10  if s == s_goal
     -5  if s in S_grey
     -1  otherwise

Note:
- Reward is associated with the NEXT state (s'), i.e., r = R(s_next).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

State = Tuple[int, int]


@dataclass(frozen=True)
class GridWorld5x5:
    size: int = 5
    gamma: float = 0.9
    goal: State = (4, 4)
    grey_states: Tuple[State, ...] = ((2, 2), (3, 0), (0, 4))

    def __post_init__(self):
        if self.size != 5:
            raise ValueError("This assignment environment is fixed to 5x5.")

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def actions(self) -> List[State]:
        # (di, dj)
        return [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    def is_terminal(self, s: State) -> bool:
        return s == self.goal

    def reward_of(self, s: State) -> float:
        if s == self.goal:
            return 10.0
        if s in self.grey_states:
            return -5.0
        return -1.0

    def is_valid(self, s: State) -> bool:
        i, j = s
        return 0 <= i < self.size and 0 <= j < self.size

    def step(self, s: State, a: int) -> Tuple[State, float, bool]:
        """Deterministic step: returns (s_next, r, done). Reward is R(s_next)."""
        if not (0 <= a < self.n_actions):
            raise ValueError(f"Invalid action index {a}")

        if self.is_terminal(s):
            return s, 0.0, True

        di, dj = self.actions[a]
        s_next = (s[0] + di, s[1] + dj)
        if not self.is_valid(s_next):
            s_next = s

        r = self.reward_of(s_next)
        done = self.is_terminal(s_next)
        return s_next, r, done

    def all_states(self) -> List[State]:
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def state_to_idx(self, s: State) -> int:
        return s[0] * self.size + s[1]

    def idx_to_state(self, idx: int) -> State:
        return (idx // self.size, idx % self.size)

    def pretty_V(self, V: np.ndarray, decimals: int = 2) -> str:
        grid = np.array(V, dtype=float).reshape(self.size, self.size)
        out_lines: List[str] = []
        for i in range(self.size):
            row_cells = []
            for j in range(self.size):
                s = (i, j)
                if s == self.goal:
                    row_cells.append("  G   ")
                else:
                    row_cells.append(f"{grid[i, j]:6.{decimals}f}")
            out_lines.append(" ".join(row_cells))
        return "\n".join(out_lines)

    def pretty_pi(self, pi: np.ndarray) -> str:
        arrows = {0: "→", 1: "←", 2: "↓", 3: "↑"}
        pi_grid = np.array(pi, dtype=int).reshape(self.size, self.size)
        out_lines: List[str] = []
        for i in range(self.size):
            row_cells = []
            for j in range(self.size):
                s = (i, j)
                if s == self.goal:
                    row_cells.append(" G ")
                else:
                    row_cells.append(f" {arrows[int(pi_grid[i, j])]} ")
            out_lines.append("".join(row_cells))
        return "\n".join(out_lines)
