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
│   │   ├── demo_rollout.py        # Q1 – Pick-and-Place rollout driver
│   │   ├── plot_rollout.py        # Q1 – Plot generation
│   │   └── pick_place_env.py      # Q1 – Pick-and-Place environment (MDP)
│   │
│   ├── q2/
│   │   ├── demo_q2.py             # Q2 – Driver for 2×2 Gridworld value iteration
│   │   └── mdp_analysis.py        # Q2 – Bellman backup logic + detailed logging
│   │
│   ├── q3/
│   │   └── gridworld_vi_oop.py    # Q3 – 5×5 Gridworld (standard vs in-place VI)
│   │
│   └── q4/
│       └── off_policy_mc.py       # Q4 – Off-policy Monte Carlo (OIS & WIS)
│
├── logs/
│   ├── q1/
│   │   ├── q1_pick_place_log.txt  # Q1 rollout log
│   │   ├── q1_rollout_report.pdf  # Q1 rollout summary report
│   │   └── figures/               # Q1 plots (reward, distance, actions)
│   │
│   ├── q2/
│   │   └── q2_mdp_analysis_log.txt # Q2 step-by-step value iteration log
│   │
│   ├── q3/                        # Q3 value iteration logs + CSV snapshots
│   └── q4/                        # Q4 Monte Carlo logs + CSV snapshots
│
├── notebooks/                     # (Optional) Jupyter notebooks for exploration/debugging
│
├── README.md                      # Project documentation
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

**Problem 3 – 5×5 Gridworld (MDP Value Iteration)**
- Deterministic transition dynamics
- Terminal goal state with positive reward
- Grey states with negative rewards
- Implemented:
  - **Standard (synchronous) Value Iteration**
  - **In-place Value Iteration**
- Confirms both methods converge to the **same optimal value function (V\*) and policy (π\*)**

**Problem 4 – Off-policy Monte Carlo with Importance Sampling**
- Uses the same 5×5 Gridworld as Problem 3
- **Behavior policy:** Uniform random  
- **Target policy:** Greedy  
- Implements:
  - Ordinary Importance Sampling (OIS)
  - Weighted Importance Sampling (WIS)
- Monte Carlo estimates are compared against **Value Iteration as a baseline**
- Results show **high variance for OIS** and **improved stability for WIS**

---

#### Comparison: Value Iteration vs Monte Carlo (Q3 vs Q4)

| Aspect | Value Iteration (MDP) | Monte Carlo (Off-policy) |
|------|-----------------------|--------------------------|
| Environment Knowledge | Full MDP model required (P, R known) | Model-free (no transition model needed) |
| Update Method | Bellman optimality backups over all states | Sampled episode returns |
| Data Requirement | No episodes required | Many episodes required |
| Convergence | Fast and deterministic | Slow and stochastic |
| Variance | Low | High for OIS, reduced for WIS |
| Importance Sampling | Not required | Required to correct behavior–target mismatch |
| Computational Cost | O(K · \|S\| · \|A\|) | O(E · T) (episodes × trajectory length) |
| Accuracy | Exact optimal solution (after convergence) | Approximate; approaches VI asymptotically |
| Best Use Case | Small or known environments | Large or unknown environments |

---

#### How to Run

```bash
# Q1
python -m src.q1.demo_rollout
python -m src.q1.plot_rollout

# Q2
python -m src.q2.demo_q2

# Q3
python -m src.q3.gridworld_vi_oop

# Q4
python -m src.q4.off_policy_mc
```

**Python version:** 3.9 or higher recommended

---

## Key Takeaways

- Value Iteration provides an exact solution when the MDP model is known
- Monte Carlo methods are flexible but require more data and careful variance control
- Both methods converge toward similar value functions, validating theoretical expectations

