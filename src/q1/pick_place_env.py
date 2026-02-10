from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, Tuple

from src.common.rollout_recorder import RolloutRecorder


class SimplePickPlaceEnv:
    """
    Toy pick-and-place environment for CSCN8020 Assignment 1 (Q1).
    """

    def __init__(
        self,
        gamma: float = 0.9,
        max_steps: int = 200,
        logger: Optional[RolloutRecorder] = None,
    ):
        self.gamma = gamma
        self.max_steps = max_steps
        self.logger = logger
        self.reset()

    def reset(self, log_init: bool = False) -> np.ndarray:
        self.q = np.zeros(3, dtype=float)
        self.qd = np.zeros(3, dtype=float)

        self.obj = np.array([0.50, 0.20, 0.00], dtype=float)
        self.goal = np.array([0.80, 0.80, 0.00], dtype=float)

        self.grip = 0
        self.t = 0

        self.prev_action = np.zeros(3, dtype=float)
        self.prev_dist = float(np.linalg.norm(self.obj - self.goal))

        if log_init and self.logger:
            self.logger.header(
                "Q1 Pick-and-Place Rollout Log",
                meta={
                    "gamma": self.gamma,
                    "max_steps": self.max_steps,
                    "state_dim": 13,
                    "action_dim": 3,
                },
            )

            # Optional: log initial state as t=0
            self.logger.log_step(
                t=0,
                state=self._get_state(),
                action=np.zeros(3, dtype=float),
                reward=0.0,
                next_state=self._get_state(),
                done=False,
                info={"note": "initial state"},
            )

        return self._get_state()

    def toggle_gripper(self, closed: bool) -> None:
        self.grip = 1 if closed else 0

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.q, self.qd, self.obj, self.goal, [float(self.grip)]])

    def _end_effector_pos(self) -> np.ndarray:
        return np.array(
            [0.50 + 0.20 * self.q[0], 0.50 + 0.20 * self.q[1], 0.00],
            dtype=float,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        One MDP transition:
        (s_t, a_t) -> (r_{t+1}, s_{t+1}, done, info)
        """
        self.t += 1

        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (3,):
            raise ValueError(f"Q1 expects action shape (3,), got {action.shape}")

        action = np.clip(action, -1.0, 1.0)

        state = self._get_state()

        # Toy dynamics
        self.qd = 0.8 * self.qd + 0.2 * action
        self.q = self.q + self.qd * 0.05

        ee = self._end_effector_pos()

        if self.grip == 1 and np.linalg.norm(ee - self.obj) < 0.08:
            self.obj = ee.copy()

        # Reward
        dist = float(np.linalg.norm(self.obj - self.goal))
        r_progress = self.prev_dist - dist
        r_smooth = -0.05 * float(np.sum((action - self.prev_action) ** 2))
        r_energy = -0.01 * float(np.sum(action ** 2))
        r_time = -0.01

        reward = float(r_progress + r_smooth + r_energy + r_time)

        self.prev_action = action.copy()
        self.prev_dist = dist

        success = dist < 0.05
        done = False

        if success:
            reward = float(reward + 10.0)
            done = True

        if self.t >= self.max_steps:
            done = True

        info: Dict[str, Any] = {
            "success": bool(success),
            "dist_to_goal": float(dist),
        }

        next_state = self._get_state()

        if self.logger:
            self.logger.log_step(
                t=self.t,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info,
            )

        return next_state, reward, done, info
