# src/common/rollout_recorder.py

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import json


@dataclass
class RolloutRecorder:
    """
    Records state–action–reward transitions during an RL rollout.
    Designed to be reused across Q1–Q4.
    """
    filepath: str

    def _write(self, line: str) -> None:
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def header(self, title: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._write("=" * 72)
        self._write(title)
        self._write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
        if meta:
            self._write("meta: " + json.dumps(meta))
        self._write("=" * 72)

    def log_step(
        self,
        t: int,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "t": int(t),
            "state": self._to_list(state),
            "action": self._to_list(action),
            "reward": float(reward),
            "next_state": self._to_list(next_state),
            "done": bool(done),
            "info": info or {},
        }

        # human-readable line
        self._write(f"[t={t:03d}] r={reward:+.4f} done={done} info={payload['info']}")

        # machine-readable JSON line (indented)
        self._write("  " + json.dumps(payload))

        # optional spacing every 25 steps (makes log readable)
        if int(t) % 25 == 0:
            self._write("-" * 72)

    def episode_summary(
        self,
        steps: int,
        total_reward: float,
        success: bool,
        final_dist: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Professor-friendly conclusion block."""
        self._write("")
        self._write("=" * 72)
        self._write("EPISODE SUMMARY")
        self._write(f"steps: {int(steps)}")
        self._write(f"success: {bool(success)}")
        self._write(f"final_dist_to_goal: {float(final_dist):.6f}")
        self._write(f"total_reward: {float(total_reward):.6f}")
        if extra:
            self._write("extra: " + json.dumps(extra))
        self._write("=" * 72)

    # Alias so your demo_rollout.py can call logger.footer(...)
    def footer(self, summary: Dict[str, Any]) -> None:
        self.episode_summary(
            steps=int(summary.get("steps", -1)),
            total_reward=float(summary.get("total_reward_recomputed", 0.0)),
            success=bool(summary.get("success", False)),
            final_dist=float(summary.get("final_dist_to_goal", float("nan"))),
            extra={k: v for k, v in summary.items()
                   if k not in {"steps", "total_reward_recomputed", "success", "final_dist_to_goal"}},
        )

    @staticmethod
    def _to_list(x: Any):
        try:
            return x.tolist()
        except Exception:
            return x
