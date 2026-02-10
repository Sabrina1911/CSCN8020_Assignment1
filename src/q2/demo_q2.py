# src/q2/demo_q2.py
from src.q2.mdp_analysis import value_iteration_two_iters_with_log

if __name__ == "__main__":
    V1, V2, pi2 = value_iteration_two_iters_with_log(
        out_log_path="logs/q2/q2_mdp_analysis_log.txt",
        gamma=0.98,
    )
    print("V1:", V1)
    print("V2:", V2)
    print("pi2:", pi2)
    print("Log written to: logs/q2/q2_mdp_analysis_log.txt")
