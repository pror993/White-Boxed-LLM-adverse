"""Plotting utilities for discrete optimization dynamics, loss curvature, and sensitivity.

Generates comparative visualization plots for Standard Cross-Entropy vs. Mellowmax
and Draft-Filtered candidate search, exporting publication-quality figures and terminal tables.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

logger = logging.getLogger("plot_dynamics")
console = Console()


def plot_loss_convergence(
    benchmark_records: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Optimization Loss Trajectories",
) -> None:
    """Plot comparative loss convergence curves across methods and tasks."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    color_map = {
        "standard_gcg": "#1f77b4",
        "mellowmax": "#ff7f0e",
        "draft_filtered": "#2ca02c",
    }
    style_map = {
        "standard_gcg": "-",
        "mellowmax": "--",
        "draft_filtered": "-.",
    }

    for record in benchmark_records:
        method = record.get("method", "unknown")
        task_id = record.get("task_id", "task")
        curve = record.get("loss_curve", [])
        if not curve:
            continue

        steps = list(range(len(curve)))
        label = f"{method} ({task_id})"
        color = color_map.get(method, "#333333")
        linestyle = style_map.get(method, "-")

        ax.plot(steps, curve, label=label, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.85)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Discrete Optimization Step", fontsize=12)
    ax.set_ylabel("Conditional Autoregressive Loss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    console.print(f"[bold green]Saved loss convergence figure to:[/bold green] {output_path}")


def plot_gradient_norms(
    benchmark_records: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Embedding Gradient Norm Dynamics",
) -> None:
    """Plot first-order gradient magnitude progression across steps."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    for record in benchmark_records:
        method = record.get("method", "unknown")
        task_id = record.get("task_id", "task")
        grads = record.get("gradient_norms", [])
        if not grads:
            continue

        steps = list(range(len(grads)))
        label = f"{method} ({task_id})"
        ax.plot(steps, grads, label=label, linewidth=1.8, alpha=0.85)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Discrete Optimization Step", fontsize=12)
    ax.set_ylabel(r"Gradient Norm $||\nabla_{X_{\mathrm{opt}}} \mathcal{L}||_2$", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    console.print(f"[bold green]Saved gradient norms figure to:[/bold green] {output_path}")


def plot_method_tradeoffs(
    benchmark_records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot Pareto bar chart comparing loss reduction vs. computational wall-time."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    methods = [r.get("method", "") for r in benchmark_records]
    reductions = [r.get("loss_reduction", 0.0) for r in benchmark_records]
    times = [r.get("total_wall_time", 0.0) for r in benchmark_records]

    x = np.arange(len(methods))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(methods)]

    ax1.bar(x, reductions, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax1.set_title("Empirical Loss Reduction (Higher is Better)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"$\Delta \mathcal{L} = \mathcal{L}_{\mathrm{init}} - \mathcal{L}_{\mathrm{best}}$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(x, times, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.set_title("Total Wall Time Seconds (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Seconds (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    console.print(f"[bold green]Saved trade-off comparison figure to:[/bold green] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate discrete optimization dynamics plots.")
    parser.add_argument(
        "--input-json",
        type=str,
        default="experiments/results/benchmark/benchmark_summary.json",
        help="Path to benchmark summary JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/plots",
        help="Directory to save generated figures",
    )
    args = parser.parse_args()

    json_path = Path(args.input_json)
    if not json_path.exists():
        console.print(f"[yellow]Input file '{json_path}' does not exist yet. Run benchmark_suite.py first.[/yellow]")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_convergence(records, out_dir / "loss_convergence.png")
    plot_gradient_norms(records, out_dir / "gradient_norms.png")
    plot_method_tradeoffs(records, out_dir / "method_tradeoffs.png")


if __name__ == "__main__":
    main()
