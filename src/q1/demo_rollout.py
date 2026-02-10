# src/q1/demo_rollout.py

import os
import numpy as np

from src.common.rollout_recorder import RolloutRecorder
from src.q1.pick_place_env import SimplePickPlaceEnv


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _scripted_action(env: SimplePickPlaceEnv) -> np.ndarray:
    """
    Simple scripted controller:
      - Move end-effector toward object
      - Once close enough, carry object toward goal
    """
    ee = env._end_effector_pos()

    # If close to object, we assume it's being carried
    carrying = np.linalg.norm(ee - env.obj) < 0.08
    target = env.goal if carrying else env.obj

    # Inverse of toy EE model:
    # ee_x = 0.50 + 0.20*q0
    # ee_y = 0.50 + 0.20*q1
    q_des = np.array([
        (target[0] - 0.50) / 0.20,
        (target[1] - 0.50) / 0.20,
        0.0,
    ], dtype=float)

    # Proportional controller in joint space
    k = 1.0
    action = k * (q_des - env.q)

    return np.clip(action, -1.0, 1.0)


def main():
    np.random.seed(42)

    # ✅ Q1-specific logs folder
    logs_dir = os.path.join(_project_root(), "logs", "q1")
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, "q1_pick_place_log.txt")

    # Clear old log
    with open(log_path, "w", encoding="utf-8"):
        pass

    logger = RolloutRecorder(log_path)
    env = SimplePickPlaceEnv(gamma=0.9, max_steps=200, logger=logger)

    env.reset(log_init=True)
    env.toggle_gripper(True)  # closed for this toy task

    total_reward = 0.0
    info = {"success": False, "dist_to_goal": None}

    for _ in range(env.max_steps):
        action = _scripted_action(env)
        _, reward, done, info = env.step(action)
        total_reward += float(reward)
        if done:
            break

    # ✅ Write final, professor-friendly summary
    logger.episode_summary(
        steps=env.t,
        total_reward=total_reward,
        success=bool(info.get("success", False)),
        final_dist=float(info.get("dist_to_goal", float("nan"))),
    )

    print("Episode finished")
    print("Steps:", env.t)
    print("Success:", info.get("success", False))
    print("Final dist_to_goal:", info.get("dist_to_goal"))
    print("Total reward:", round(total_reward, 4))
    print("Log saved at:", log_path)


if __name__ == "__main__":
    main()
