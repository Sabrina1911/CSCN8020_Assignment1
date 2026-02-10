# src/q1/plot_rollout.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# Paths
# -----------------------------
def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _q1_logs_dir() -> str:
    return os.path.join(_project_root(), "logs", "q1")


def _default_log_path() -> str:
    # ✅ updated: log is now inside logs/q1/
    return os.path.join(_q1_logs_dir(), "q1_pick_place_log.txt")


# -----------------------------
# Read + parse transitions
# -----------------------------
def read_transitions(log_path: str) -> List[Dict[str, Any]]:
    """
    Read transitions from a rollout log.

    Supports:
      A) Lines that start with:   JSON: {...}
      B) Lines that are raw JSON: {...}
      C) Lines that are indented JSON (common if logger writes "  " + json.dumps(payload))

    Ignores:
      - headers, separators, summary text, blank lines
      - malformed json lines safely

    Additionally filters objects to ensure they're real transitions.
    """
    transitions: List[Dict[str, Any]] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            payload_str: Optional[str] = None

            # Format A: JSON: {...}
            if s.startswith("JSON:"):
                payload_str = s[len("JSON:") :].strip()

            # Format B/C: raw JSON object line (including indented JSON once stripped)
            elif s.startswith("{") and s.endswith("}"):
                payload_str = s

            if payload_str is None:
                continue

            try:
                obj = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            # Only keep real transition objects
            if (
                isinstance(obj, dict)
                and ("t" in obj)
                and ("reward" in obj)
                and ("action" in obj)
                and ("state" in obj or "next_state" in obj)
            ):
                transitions.append(obj)

    return transitions


def extract_series(
    transitions: List[Dict[str, Any]]
) -> Tuple[List[int], List[float], List[float], List[float], List[bool], List[bool]]:
    """
    Extract time-series from transitions.
    Returns:
      t, reward, dist_to_goal, action_magnitude, done_flags, success_flags
    """
    t: List[int] = []
    reward: List[float] = []
    dist: List[float] = []
    action_mag: List[float] = []
    done_flags: List[bool] = []
    success_flags: List[bool] = []

    for i, tr in enumerate(transitions):
        step = int(tr.get("t", i + 1))
        t.append(step)

        reward.append(float(tr.get("reward", 0.0)))

        info = tr.get("info", {}) or {}
        dist.append(float(info.get("dist_to_goal", float("nan"))))

        a = tr.get("action", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        a0 = float(a[0]) if len(a) > 0 else 0.0
        a1 = float(a[1]) if len(a) > 1 else 0.0
        a2 = float(a[2]) if len(a) > 2 else 0.0
        mag = (a0 * a0 + a1 * a1 + a2 * a2) ** 0.5
        action_mag.append(float(mag))

        done_flags.append(bool(tr.get("done", False)))
        success_flags.append(bool(info.get("success", False)))

    return t, reward, dist, action_mag, done_flags, success_flags


def cumulative(values: List[float]) -> List[float]:
    out: List[float] = []
    s = 0.0
    for v in values:
        s += float(v)
        out.append(s)
    return out


# -----------------------------
# Plot helpers (human-readable)
# -----------------------------
def _add_markers(
    ax: plt.Axes,
    t: List[int],
    done_step: Optional[int],
    success_step: Optional[int],
    chunk: int = 25,
) -> None:
    """Add vertical markers for readability."""
    if not t:
        return

    t_min, t_max = min(t), max(t)

    # chunk separators
    first = ((t_min // chunk) + 1) * chunk
    for k in range(first, t_max + 1, chunk):
        ax.axvline(k, linewidth=0.8, linestyle="--")

    # episode end marker
    if done_step is not None:
        ax.axvline(done_step, linewidth=2.0, linestyle="-")
        ax.text(
            done_step,
            0.98,
            "END",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
        )

    # success marker
    if success_step is not None:
        ax.axvline(success_step, linewidth=2.0, linestyle=":")
        ax.text(
            success_step,
            0.90,
            "SUCCESS",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
        )


def save_line_plot(
    x: List[int],
    y: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    done_step: Optional[int] = None,
    success_step: Optional[int] = None,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    _add_markers(ax, x, done_step=done_step, success_step=success_step, chunk=25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_pdf_report(
    pdf_path: str,
    t: List[int],
    r: List[float],
    r_cum: List[float],
    d: List[float],
    a_mag: List[float],
    done_step: Optional[int],
    success_step: Optional[int],
    summary: Dict[str, Any],
) -> None:
    """Create a single PDF with a summary page + 4 plots."""
    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Summary text ---
        fig = plt.figure(figsize=(8.27, 11.69))  # A4-ish
        fig.suptitle("Q1 Pick-and-Place Rollout Report", fontsize=16, y=0.98)

        lines = [
            f"Log file: {summary.get('log_path')}",
            f"Transitions parsed: {summary.get('n_steps')}",
            f"Episode end step: {summary.get('end_step')}",
            f"Success: {summary.get('success')}",
            f"Final distance to goal: {summary.get('final_dist')}",
            f"Total reward (sum): {summary.get('total_reward')}",
            "",
            "How to read the plots:",
            "- Dashed vertical lines = every 25 steps (time chunks)",
            "- Solid END line = episode termination step",
            "- Dotted SUCCESS line = success step (if achieved)",
        ]
        fig.text(0.07, 0.92, "\n".join(lines), va="top", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        def add_plot_page(title: str, y: List[float], ylabel: str):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(t, y)
            ax.set_title(title)
            ax.set_xlabel("t (step)")
            ax.set_ylabel(ylabel)
            _add_markers(ax, t, done_step=done_step, success_step=success_step, chunk=25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        add_plot_page("Reward per Step", r, "reward")
        add_plot_page("Cumulative Reward", r_cum, "cumulative reward")
        add_plot_page("Distance to Goal", d, "dist_to_goal")
        add_plot_page("Action Magnitude", a_mag, "||action||")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    log_path = _default_log_path()
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    transitions = read_transitions(log_path)
    if not transitions:
        raise RuntimeError(
            "No JSON transitions found. Your log must contain either:\n"
            " - lines starting with 'JSON: {...}' OR\n"
            " - raw JSON object lines like '{...}' (including indented JSON).\n"
            "Tip: re-run: python -m src.q1.demo_rollout"
        )

    t, r, d, a_mag, done_flags, success_flags = extract_series(transitions)
    r_cum = cumulative(r)

    # done_step = first done=True, else last step
    done_step: Optional[int] = None
    for i, is_done in enumerate(done_flags):
        if is_done:
            done_step = t[i]
            break
    if done_step is None and t:
        done_step = t[-1]

    # success_step = first success True (if any)
    success_step: Optional[int] = None
    for i, is_succ in enumerate(success_flags):
        if is_succ:
            success_step = t[i]
            break

    total_reward = float(sum(r))
    final_dist = d[-1] if d else float("nan")

    # ✅ updated: write outputs into logs/q1/
    logs_dir = _q1_logs_dir()
    os.makedirs(logs_dir, exist_ok=True)

    # PNG outputs
    save_line_plot(
        t, r,
        "Q1 Reward per Step", "t (step)", "reward",
        os.path.join(logs_dir, "q1_reward.png"),
        done_step=done_step, success_step=success_step,
    )
    save_line_plot(
        t, r_cum,
        "Q1 Cumulative Reward", "t (step)", "cumulative reward",
        os.path.join(logs_dir, "q1_cum_reward.png"),
        done_step=done_step, success_step=success_step,
    )
    save_line_plot(
        t, d,
        "Q1 Distance to Goal", "t (step)", "dist_to_goal",
        os.path.join(logs_dir, "q1_dist_to_goal.png"),
        done_step=done_step, success_step=success_step,
    )
    save_line_plot(
        t, a_mag,
        "Q1 Action Magnitude", "t (step)", "||action||",
        os.path.join(logs_dir, "q1_action_mag.png"),
        done_step=done_step, success_step=success_step,
    )

    # PDF report
    pdf_path = os.path.join(logs_dir, "q1_rollout_report.pdf")
    summary = {
        "log_path": log_path,
        "n_steps": len(transitions),
        "end_step": done_step,
        "success": bool(success_step is not None),
        "final_dist": f"{final_dist:.4f}" if final_dist == final_dist else "nan",
        "total_reward": f"{total_reward:.4f}",
    }
    save_pdf_report(
        pdf_path=pdf_path,
        t=t,
        r=r,
        r_cum=r_cum,
        d=d,
        a_mag=a_mag,
        done_step=done_step,
        success_step=success_step,
        summary=summary,
    )

    print("Saved outputs to:", logs_dir)
    print(" - q1_reward.png")
    print(" - q1_cum_reward.png")
    print(" - q1_dist_to_goal.png")
    print(" - q1_action_mag.png")
    print(" - q1_rollout_report.pdf")


if __name__ == "__main__":
    main()
