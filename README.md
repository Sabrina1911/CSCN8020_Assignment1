## CSCN8020 – Assignment 1
### Reinforcement Learning Programming

**Course:** CSCN8020 – Reinforcement Learning Programming  
**Assignment:** Assignment 1  
**Student:** Sabrina Ronnie George Karippatt  

---

#### Assignment Overview

This assignment explores foundational reinforcement learning concepts through **Markov Decision Processes (MDPs)** and **Monte Carlo (MC)** methods. The focus is on modeling environments, computing value functions, and comparing **model-based** and **model-free** approaches using Gridworld examples.

The assignment consists of four problems:

1. **Problem 1:** MDP design for a Pick-and-Place Robot  
2. **Problem 2:** Manual Value Iteration on a 2×2 Gridworld  
3. **Problem 3:** Value Iteration on a 5×5 Gridworld (MDP-based)  
4. **Problem 4:** Off-policy Monte Carlo with Importance Sampling  

---

#### Folder Structure

```
CSCN8020_Assignment1/
│
├── src/
│   ├── q1/
│   │   ├── demo_rollout.py          # Q1 – Pick-and-Place rollout driver
│   │   ├── plot_rollout.py          # Q1 – Plot generation
│   │   └── pick_place_env.py        # Q1 – Pick-and-Place environment (MDP)
│   │
│   ├── q2/
│   │   ├── demo_q2.py               # Q2 – Driver for 2×2 Gridworld value iteration
│   │   └── mdp_analysis.py          # Q2 – Bellman backup logic + detailed logging
│   │
│   ├── q3/
│   │   ├── gridworld.py             # Q3 – 5×5 Gridworld environment (δ, R, γ)
│   │   ├── value_iteration_agent.py # Q3 – Shared VI agent utilities
│   │   ├── value_iteration_solved.py    # Q3 – Standard (synchronous) Value Iteration
│   │   ├── value_iteration_inplace.py   # Q3 – In-place (Gauss–Seidel) Value Iteration
│   │   └── vi_logger.py             # Q3 – Logging + CSV snapshot utilities
│   │
│   └── q4/
│       ├── offpolicy_mc_importance_sampling.py  # Q4 – Off-policy MC (OIS & WIS)
│       └── run_q4.py                            # Q4 – Runner that executes + saves outputs
│
├── logs/
│   ├── q1/
│   │   ├── q1_pick_place_log.txt    # Q1 rollout log
│   │   ├── q1_rollout_report.pdf    # Q1 rollout summary report
│   │   └── figures/                 # Q1 plots (reward, distance, actions)
│   │
│   ├── q2/
│   │   └── q2_mdp_analysis_log.txt  # Q2 step-by-step value iteration log
│   │
│   ├── q3/                          # Q3 value iteration logs + CSV snapshots
│   │   ├── q3_std_value_iteration_*.log
│   │   ├── q3_inplace_value_iteration_*.log
│   │   ├── q3_std_snapshots_*.csv
│   │   └── q3_inplace_snapshots_*.csv
│   │
│   └── q4/                          # Q4 Monte Carlo logs + CSV outputs
│       ├── q4_run_*.txt
│       ├── V_mc_*.csv
│       ├── V_vi_*.csv
│       ├── pi_mc_*.csv
│       └── pi_vi_*.csv
│
├── notebooks/
│   ├── Q1_PickAndPlace_MDP.ipynb
│   ├── Q2_2x2_ValueIteration_NoCode.ipynb
│   ├── Q3_5x5_ValueIteration.ipynb
│   └── Q4_OffPolicy_MC_ImportanceSampling.ipynb
│
└── README.md

```

**Note:** Virtual environment folders (e.g., `.venv/`) are intentionally excluded from submission to keep the project portable and reproducible.

---

#### Problem Details

**Problem 1 – Pick-and-Place Robot (MDP Design & Rollout Analysis)**
- The task is formulated as a **finite-horizon Markov Decision Process**
- **States:** Robot arm positions, velocities, and object status  
- **Actions:** Joint motor controls  
- **Rewards:** Encourage smooth motion, progress toward the goal, and successful task completion  
- A scripted policy rollout is executed to **validate environment dynamics and reward design**
- No learning is performed; analysis is based on **logged trajectories, plots, and a final rollout report**

**Problem 2 – 2×2 Gridworld (Manual Value Iteration)**
- Performs **two iterations** of Value Iteration derived manually
- Demonstrates:
  - Value function initialization
  - Bellman backups
  - Greedy policy extraction after iteration 2
- A small Python script is used **only to verify calculations and generate a step-by-step log**

**Problem 3 – 5×5 Gridworld (MDP Value Iteration, γ = 0.99)**
- Deterministic transition dynamics with reward-on-arrival
- Terminal goal state with positive reward (+10)
- Grey states with negative rewards (−5)
- Step cost of −1 for all other states
- Implemented:
  - **Standard (synchronous) Value Iteration**
  - **In-place Value Iteration**
- Both implementations converge to the same optimal state-value function V∗ and optimal greedy policy 𝜋∗
- Demonstrates deterministic contraction behavior of Dynamic Programming when the full MDP model is known

**Problem 4 – Off-policy Monte Carlo with Importance Sampling**
- Uses the same 5×5 Gridworld as Problem 3
- **Behavior policy:** Uniform random  
- **Target policy:** Greedy with respect to learned 𝑄(𝑠,𝑎)  
- Implements:
  - Ordinary Importance Sampling (OIS)
  - Weighted Importance Sampling (WIS)
- Estimates the action-value function Q(s,a),then derives:
  * V(s)=maxa​Q(s,a)
  * Greedy policy 𝜋(𝑠)
- Monte Carlo estimates are compared against Value Iteration (Q3) as the optimal baseline
- Results show:
  * High variance and instability for OIS
  * Improved stability and convergence for WIS
  * Convergence toward the optimal solution despite being model-free

---

#### Comparison: Value Iteration vs Monte Carlo (Q3 vs Q4)

| Aspect | Value Iteration (Q3, γ = 0.99) | Monte Carlo (Q4, γ = 0.9) |
|---|---|---|
| Environment Knowledge | Full MDP model required (transition δ and reward R known) | Model-free (no transition model required) |
| Update Method | Bellman optimality backup applied to **all states each sweep** | Returns estimated from **sampled episodes** |
| Data Requirement | No episodes required | Requires many episodes for stable estimates |
| Iteration Unit | Sweeps over entire state space | Episodes (complete trajectories) |
| Convergence Behavior | Deterministic and fast contraction | Stochastic and sample-dependent |
| Variance | Very low (deterministic updates) | High for Ordinary IS; reduced using Weighted IS |
| Importance Sampling | Not required | Required to correct behavior–target policy mismatch |
| Computational Cost | O(K · \|S\| · \|A\|) | O(E · T) (episodes × trajectory length) |
| Accuracy | Exact optimal solution after convergence | Approximate; approaches optimal solution asymptotically |
| Sensitivity to γ | Higher γ (0.99) emphasizes long-term reward | Lower γ (0.9) stabilizes Monte Carlo estimates |
| Best Use Case | Small or fully known environments | Large-scale or unknown environments |


---

#### How to Run

```bash
# Q1
python -m src.q1.demo_rollout
python -m src.q1.plot_rollout

# Q2
python -m src.q2.demo_q2

# Q3 - Standard Value Iteration
python -m src.q3.value_iteration_solved

# Q3 - In-Place Value Iteration
python -m src.q3.value_iteration_inplace

# Q4 - Off-Policy Monte Carlo with Importance Sampling
python -m src.q4.run_q4
```

**Python version:** Python 3.9+ (tested with Python 3.14)

---

## Key Takeaways

- Value Iteration efficiently computes the optimal solution when the full MDP model (states, transitions, and rewards) is known. In Q3, using γ = 0.99, the 5×5 Gridworld converged in a small number of sweeps, illustrating the fast contraction property of Dynamic Programming methods.
- Off-policy Monte Carlo with Weighted Importance Sampling (Q4, γ = 0.9) does not require knowledge of the transition model, but relies on sampled episodes and therefore requires significantly more data and computation to approximate the optimal value function.
-Although different discount factors were used across problems (Q1: γ = 0.9 Q2: γ = 0.98, Q3: γ = 0.99, Q4: γ = 0.9), this was intentional and aligned with the experimental setup of each task. The choice of γ influences how strongly future rewards are weighted but does not affect the correctness of the algorithms.
- Despite differences in γ and computational effort, both Dynamic Programming (model-based) and Monte Carlo (model-free) approaches converged to highly similar value functions and greedy policies, confirming theoretical expectations.
- In-place and synchronous Value Iteration produced identical convergence behavior in our implementation due to sweep ordering. This demonstrates that update order can influence intermediate propagation speed, but does not change the final optimal solution.