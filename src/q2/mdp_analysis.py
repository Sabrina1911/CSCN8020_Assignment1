# src/q2/mdp_analysis.py
"""
Q2 — 2x2 Gridworld (Value Iteration, 2 iterations)

Important:
- The assignment asks for the step-by-step process WITHOUT code.
- This module is only a *verifier* + log generator to support your written math.

Notation matches class style:
- V_k(s) is the state value estimate at iteration k
- q_k(s,a) = sum_{s'} P(s'|s,a) [ R(s) + gamma * V_k(s') ]
- V_{k+1}(s) = max_a q_k(s,a)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

State = str
Action = str


@dataclass
class Gridworld2x2:
    gamma: float = 0.98

    # Rewards are state-based (same for all actions), as per the question
    rewards: Dict[State, float] = None

    # Deterministic transition function f(s,a) -> s'
    transitions: Dict[Tuple[State, Action], State] = None

    def __post_init__(self):
        if self.rewards is None:
            self.rewards = {"s1": 5.0, "s2": 10.0, "s3": 1.0, "s4": 2.0}

        if self.transitions is None:
            # Layout:
            # s1 s2
            # s3 s4
            self.transitions = {
                # s1
                ("s1", "up"): "s1",     # wall
                ("s1", "left"): "s1",   # wall
                ("s1", "right"): "s2",
                ("s1", "down"): "s3",
                # s2
                ("s2", "up"): "s2",     # wall
                ("s2", "right"): "s2",  # wall
                ("s2", "left"): "s1",
                ("s2", "down"): "s4",
                # s3
                ("s3", "down"): "s3",   # wall
                ("s3", "left"): "s3",   # wall
                ("s3", "up"): "s1",
                ("s3", "right"): "s4",
                # s4
                ("s4", "down"): "s4",   # wall
                ("s4", "right"): "s4",  # wall
                ("s4", "up"): "s2",
                ("s4", "left"): "s3",
            }

    @property
    def states(self) -> List[State]:
        return ["s1", "s2", "s3", "s4"]

    @property
    def actions(self) -> List[Action]:
        return ["up", "down", "left", "right"]

    def next_state(self, s: State, a: Action) -> State:
        return self.transitions[(s, a)]

    def r(self, s: State) -> float:
        return self.rewards[s]


def value_iteration_two_iters_with_log(
    out_log_path: str = "logs/q2/q2_mdp_analysis_log.txt",
    gamma: float = 0.98,
) -> Tuple[Dict[State, float], Dict[State, float], Dict[State, Action]]:
    """
    Runs exactly 2 iterations:
      V0 -> V1 -> V2
    Writes a step-by-step, professor-friendly log showing each q_k(s,a) computation.
    Returns (V1, V2, greedy_policy_after_iter2_wrt_V1).
    """
    env = Gridworld2x2(gamma=gamma)

    # Ensure logs/q2 exists
    log_path = Path(out_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def w(line: str = ""):
        log_path.open("a", encoding="utf-8").write(line + "\n")

    # Overwrite log each run
    log_path.write_text("", encoding="utf-8")

    # ---------------------- helpers for matrix / grid output ----------------------
    def fmt2(x: float) -> str:
        return f"{x:6.2f}"

    def write_value_matrix(title: str, V: Dict[State, float]):
        # Layout:
        # [ s1  s2
        #   s3  s4 ]
        w(title)
        w("Layout: [ s1  s2 ; s3  s4 ]")
        w(f"[ {fmt2(V['s1'])}  {fmt2(V['s2'])} ]")
        w(f"[ {fmt2(V['s3'])}  {fmt2(V['s4'])} ]")
        w()

    def write_policy_grid(title: str, pi: Dict[State, Action]):
        arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
        w(title)
        w("Policy grid (arrows):")
        w(f"[ s1 {arrow.get(pi['s1'], '?')}   s2 {arrow.get(pi['s2'], '?')} ]")
        w(f"[ s3 {arrow.get(pi['s3'], '?')}   s4 {arrow.get(pi['s4'], '?')} ]")
        w()
    # ------------------------------------------------------------------------------

    w("=" * 72)
    w("Q2 — 2x2 Gridworld — Value Iteration (2 iterations)")
    w(f"Assumption: gamma = {gamma}")
    w("States: s1,s2,s3,s4 | Actions: up,down,left,right")
    w("Update: q_k(s,a)=R(s)+gamma*V_k(s')  (deterministic transitions)")
    w("Then:   V_{k+1}(s)=max_a q_k(s,a)")
    w("=" * 72)
    w()

    # V0 initialization (standard)
    V0 = {s: 0.0 for s in env.states}
    w("INITIAL VALUES (V0)")
    for s in env.states:
        w(f"  V0({s}) = {V0[s]:.4f}")
    w()

    # Helper to compute q_k(s,a)
    def q_of(Vk: Dict[State, float], s: State, a: Action) -> float:
        s_prime = env.next_state(s, a)
        return env.r(s) + env.gamma * Vk[s_prime]

    # -------- Iteration 1: V0 -> V1 --------
    w("-" * 72)
    w("ITERATION 1 (compute V1 from V0)")
    w("-" * 72)

    V1: Dict[State, float] = {}
    for s in env.states:
        w(f"\nState {s}: R({s})={env.r(s):.4f}")
        q_vals = {}
        for a in env.actions:
            sp = env.next_state(s, a)
            q = q_of(V0, s, a)
            q_vals[a] = q
            w(f"  a={a:5s} -> s'={sp} | q0({s},{a}) = R({s}) + γ V0({sp})")
            w(f"              = {env.r(s):.4f} + {env.gamma:.2f}*{V0[sp]:.4f} = {q:.4f}")
        best_a = max(q_vals, key=q_vals.get)
        V1[s] = q_vals[best_a]
        w(f"  => V1({s}) = max_a q0({s},a) = {V1[s]:.4f}  (greedy a*: {best_a})")

    w("\nUPDATED VALUES (V1)")
    for s in env.states:
        w(f"  V1({s}) = {V1[s]:.4f}")

    # -------- Iteration 2: V1 -> V2 --------
    w("\n" + "-" * 72)
    w("ITERATION 2 (compute V2 from V1)")
    w("-" * 72)

    V2: Dict[State, float] = {}
    greedy_pi2: Dict[State, Action] = {}
    for s in env.states:
        w(f"\nState {s}: R({s})={env.r(s):.4f}")
        q_vals = {}
        for a in env.actions:
            sp = env.next_state(s, a)
            q = q_of(V1, s, a)
            q_vals[a] = q
            w(f"  a={a:5s} -> s'={sp} | q1({s},{a}) = R({s}) + γ V1({sp})")
            w(f"              = {env.r(s):.4f} + {env.gamma:.2f}*{V1[sp]:.4f} = {q:.4f}")
        best_a = max(q_vals, key=q_vals.get)
        V2[s] = q_vals[best_a]
        greedy_pi2[s] = best_a
        w(f"  => V2({s}) = max_a q1({s},a) = {V2[s]:.4f}  (greedy a*: {best_a})")

    w("\nUPDATED VALUES (V2)")
    for s in env.states:
        w(f"  V2({s}) = {V2[s]:.4f}")

    w("\nGREEDY POLICY after Iteration 2 (greedy w.r.t V1)")
    for s in env.states:
        w(f"  π2({s}) = {greedy_pi2[s]}")

    # ---------------------- Q-table matrix output (Iteration 2) ----------------------
    w("\nQ1(s,a) table (using V1, gamma = {:.2f})".format(env.gamma))
    Q1 = {s: {} for s in env.states}
    for s in env.states:
        for a in env.actions:
            sp = env.next_state(s, a)
            Q1[s][a] = env.r(s) + env.gamma * V1[sp]

    w(f"{'':6s} {'up':>8s} {'down':>8s} {'left':>8s} {'right':>8s}")
    for s in env.states:
        w(
            f"{s:6s}"
            f"{Q1[s]['up']:8.2f}"
            f"{Q1[s]['down']:8.2f}"
            f"{Q1[s]['left']:8.2f}"
            f"{Q1[s]['right']:8.2f}"
        )
    w()
    # ------------------------------------------------------------------------------

    # ---------------------- Matrix / grid representation section ----------------------
    w("=" * 72)
    w("MATRIX / GRID REPRESENTATION (for report-style viewing)")
    w("=" * 72)
    write_value_matrix("V0 matrix:", V0)
    write_value_matrix("V1 matrix:", V1)
    write_value_matrix("V2 matrix:", V2)
    write_policy_grid("Greedy policy grid after Iteration 2 (π2):", greedy_pi2)
    # ------------------------------------------------------------------------------

    w("CONCLUSION")
    w("  - Iteration 1 collapses to V1(s)=R(s) because V0 is initialized to 0.")
    w("  - Iteration 2 propagates value through γ*V1(s'), producing higher values near s2 (reward 10).")
    w("=" * 72)

    return V1, V2, greedy_pi2
