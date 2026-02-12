#!/usr/bin/python3
"""Q3 Task 2: In-Place Value Iteration using lec3 files.

- Uses a single value array (in-place / Gauss–Seidel style updates).
- Logs snapshots under ./logs/q3/.
- Prints V*, π*, iterations, and runtime.
"""

import time
import numpy as np

from gridworld import GridWorld
from value_iteration_agent import Agent
from vi_logger import VILogger


def main():
    ENV_SIZE = 5
    GAMMA = 0.99
    THETA_THRESHOLD = 1e-8
    MAX_ITERATIONS = 200000

    SNAPSHOT_EVERY = 1  # write a snapshot every N sweeps

    env = GridWorld(ENV_SIZE)
    agent = Agent(env, theta_threshold=THETA_THRESHOLD, gamma=GAMMA)

    reward_list = env.get_reward_list()
    print("Reward list (row-major, length = %d):" % len(reward_list))
    print(reward_list)
    print()

    # Logger
    logger = VILogger(env, log_dir="../../logs/q3", prefix="q3_inplace")


    # Initial snapshot (k=0)
    agent.update_greedy_policy()
    logger.snapshot(k=0, delta_max=0.0, V=agent.get_value_function(), pi_str=agent.pi_str)

    # --- In-place Value Iteration ---
    t0 = time.perf_counter()
    iters = 0
    delta_max = 0.0

    for k in range(1, MAX_ITERATIONS + 1):
        V_old = np.copy(agent.get_value_function())

        # in-place sweep: write directly into agent.V
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if not env.is_terminal_state(i, j):
                    new_val, _, _ = agent.calculate_max_value(i, j)
                    agent.V[i, j] = new_val

        delta_max = float(np.max(np.abs(agent.V - V_old)))
        iters = k

        if (k % SNAPSHOT_EVERY) == 0:
            agent.update_greedy_policy()
            logger.snapshot(k=k, delta_max=delta_max, V=agent.get_value_function(), pi_str=agent.pi_str)

        if delta_max <= THETA_THRESHOLD:
            break

    t1 = time.perf_counter()

    V_star = agent.get_value_function()
    agent.update_greedy_policy()
    logger.converged(k=iters, delta_max=delta_max, V=V_star, pi_str=agent.pi_str)

    print("=== In-Place Value Iteration ===")
    print(f"Iterations (sweeps): {iters}")
    print(f"Runtime (seconds):   {t1 - t0:.6f}")
    print("\nV*:\n")
    print(logger._format_V(V_star, decimals=2))
    print("\nπ* (arrows):\n")
    print(logger._format_pi(agent.pi_str))

    print("\nLogs written:")
    print("  ", logger.log_path)
    print("  ", logger.csv_path)


if __name__ == "__main__":
    main()
