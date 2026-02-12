from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from gridworld5x5 import GridWorld5x5
from offpolicy_mc_importance_sampling import (
    format_pi_grid as format_pi_grid_mc,
    format_V_grid as format_V_grid_mc,
    off_policy_mc_control_weighted_is,
)
from value_iteration_q3 import (
    format_pi_grid as format_pi_grid_vi,
    format_V_grid as format_V_grid_vi,
    value_iteration,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100_000, help="Number of behavior episodes to sample")
    p.add_argument("--seed", type=int, default=7, help="RNG seed")
    p.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    p.add_argument("--max-steps", type=int, default=200, help="Max steps per episode (safety cap)")

    # Value Iteration logging controls
    p.add_argument(
        "--log-vi",
        choices=["deltas", "tables", "none"],
        default="deltas",
        help="Log VI progress: deltas (compact), tables (requires V_history in VI), none.",
    )

    # NEW: Monte Carlo progress logging controls (checkpoint-style, not every episode)
    p.add_argument(
        "--log-mc-every",
        type=int,
        default=10_000,
        help="Log MC progress every N episodes (set 0 to disable).",
    )

    return p.parse_args()


def get_logs_dir() -> Path:
    """
    Expected project structure:
      CSCN8020_Assignment1/
        logs/q4/
        src/q4/run_q4.py

    run_q4.py is in src/q4, so project root is parents[2].
    """
    project_root = Path(__file__).resolve().parents[2]
    logs_dir = project_root / "logs" / "q4"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def main() -> None:
    args = parse_args()
    env = GridWorld5x5(gamma=args.gamma)

    logs_dir = get_logs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"q4_run_{ts}.txt"

    def log_print(fh, msg: str = "") -> None:
        print(msg)
        fh.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        log_print(f, "==============================")
        log_print(f, "Q4 - Off-policy Monte Carlo (Weighted IS)")
        log_print(f, "Timestamp: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        log_print(f, "==============================\n")

        log_print(f, f"Environment: 5x5, goal={env.goal}, grey={list(env.grey_states)}, gamma={env.gamma}")
        log_print(f, f"Run args: episodes={args.episodes}, seed={args.seed}, max_steps={args.max_steps}")
        log_print(f, f"VI per-iteration logging: {args.log_vi}")
        log_print(f, f"MC checkpoint logging every: {args.log_mc_every if args.log_mc_every > 0 else 'DISABLED'} episodes\n")

        # ---------------------------
        # Value Iteration baseline
        # ---------------------------
        t0 = time.perf_counter()
        vi_res = value_iteration(env)
        t_vi = time.perf_counter() - t0

        # Log VI convergence trace
        if args.log_vi != "none":
            log_print(f, "--- Value Iteration convergence trace ---")
            for i, d in enumerate(vi_res.delta_history, start=1):
                log_print(f, f"VI iter {i:02d}: delta={d:.12f}")
            log_print(f, "")

            if args.log_vi == "tables":
                # This requires value_iteration_q3.py to return V_history (list of V tables)
                # If not implemented, we just print a helpful note.
                if not hasattr(vi_res, "V_history") or vi_res.V_history is None:
                    log_print(
                        f,
                        "NOTE: Full VI tables per-iteration require modifying value_iteration_q3.py "
                        "to store and return V_history. Currently only delta_history is available.\n",
                    )
                else:
                    log_print(f, "--- Value Iteration V tables (per iteration) ---")
                    for i, V_i in enumerate(vi_res.V_history, start=1):
                        log_print(f, f"\nVI V after iter {i:02d}:")
                        log_print(f, format_V_grid_vi(env, V_i, decimals=2))
                    log_print(f, "")

        # ---------------------------
        # Off-policy MC (with checkpoints)
        # ---------------------------
        def mc_logger(msg: str) -> None:
            log_print(f, msg)

        log_every = None if args.log_mc_every <= 0 else args.log_mc_every

        t1 = time.perf_counter()
        # IMPORTANT: This requires off_policy_mc_control_weighted_is to accept:
        #   log_every=..., logger=...
        mc_res = off_policy_mc_control_weighted_is(
            env,
            num_episodes=args.episodes,
            seed=args.seed,
            max_steps_per_episode=args.max_steps,
            log_every=log_every,
            logger=mc_logger,
        )
        t_mc = time.perf_counter() - t1

        # ---------------------------
        # Print + log final results
        # ---------------------------
        log_print(f, "\n--- Monte Carlo (estimated - final) ---")
        log_print(f, f"Episodes sampled: {mc_res.episodes}")
        log_print(f, f"Total transitions used: {mc_res.steps_collected}")
        log_print(f, f"Runtime (MC): {t_mc:.4f} seconds\n")
        log_print(f, "Estimated V (from Q):")
        log_print(f, format_V_grid_mc(env, mc_res.V, decimals=2))
        log_print(f, "\nGreedy target policy (from Q):")
        log_print(f, format_pi_grid_mc(env, mc_res.pi))

        log_print(f, "\n--- Value Iteration (baseline - final) ---")
        log_print(f, f"Iterations: {vi_res.iterations}")
        log_print(f, f"Runtime (VI): {t_vi:.4f} seconds\n")
        log_print(f, "Optimal V*:")
        log_print(f, format_V_grid_vi(env, vi_res.V, decimals=2))
        log_print(f, "\nOptimal policy pi*:")
        log_print(f, format_pi_grid_vi(env, vi_res.pi))

        v_diff = float(np.max(np.abs(mc_res.V - vi_res.V)))
        log_print(f, "\n--- Quick comparison ---")
        log_print(f, f"Max |V_MC - V*|: {v_diff:.4f}\n")

        log_print(f, "Optimization time & computation (high level):")
        log_print(
            f,
            "- Value Iteration: model-based sweeps all states each iteration. "
            "One sweep ≈ O(|S|·|A|). Total ≈ O(K·|S|·|A|)."
        )
        log_print(
            f,
            "- Off-policy MC (IS): model-free, uses sampled episodes. "
            "Total ≈ O(E·L) where E=episodes and L=avg episode length (capped)."
        )
        log_print(
            f,
            "- MC typically needs many episodes for stable estimates; importance sampling can have high variance, "
            "so checkpoint logs help visualize progress."
        )

    # Save numeric outputs as CSV for reproducibility
    np.savetxt(logs_dir / f"V_mc_{ts}.csv", mc_res.V, delimiter=",")
    np.savetxt(logs_dir / f"pi_mc_{ts}.csv", mc_res.pi.astype(int), delimiter=",", fmt="%d")
    np.savetxt(logs_dir / f"V_vi_{ts}.csv", vi_res.V, delimiter=",")
    np.savetxt(logs_dir / f"pi_vi_{ts}.csv", vi_res.pi.astype(int), delimiter=",", fmt="%d")

    print(f"\n✅ Log saved to: {log_path}")
    print(f"✅ CSVs saved to: {logs_dir}")


if __name__ == "__main__":
    main()
