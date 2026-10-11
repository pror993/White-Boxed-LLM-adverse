"""Automated benchmark harness for discrete optimization dynamics and sensitivity profiling.

Executes comparative benchmarks across optimization methods (Standard Cross-Entropy,
Mellowmax, and Multi-Fidelity Draft Filtering) on Apple Silicon MPS, logging convergence
trajectories, empirical sharpness, and compute metrics to JSON and CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from src.eval.sensitivity import DiscreteSensitivityProfiler, PerturbationType
from src.models.causal_lm import CausalLMModelWrapper
from src.optimizers.base import OptimizerConfig
from src.optimizers.coordinate_gradient import CoordinateGradientOptimizer
from src.optimizers.draft_filter import DraftFilterConfig, DraftModelFilter
from src.utils.device import clear_device_cache, get_best_device, get_mps_memory_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_suite")
console = Console()

# Standard benchmark evaluation targets for sensitivity & convergence profiling
BENCHMARK_TASKS = [
    {
        "id": "task_calculus",
        "prompt": "Explain the fundamental theorem of calculus",
        "target": "The fundamental theorem connects differentiation and integration.",
    },
    {
        "id": "task_transformer",
        "prompt": "Describe the architecture of causal transformers",
        "target": "Causal transformers utilize masked self-attention to generate tokens autoregressively.",
    },
    {
        "id": "task_convex_opt",
        "prompt": "Summarize the foundations of convex optimization",
        "target": "Convex optimization studies minimizing convex functions over convex sets.",
    },
]


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    synthetic: bool = False,
):
    """Load primary target model and tokenizer."""
    if synthetic:
        config = GPT2Config(vocab_size=1000, n_positions=256, n_embd=64, n_layer=2, n_head=2, use_cache=False)
        model = GPT2LMHeadModel(config).to(device=device, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
        device_map=str(device),
        trust_remote_code=True,
    )
    return model, tokenizer


def run_benchmark_experiment(
    model_wrapper: CausalLMModelWrapper,
    task: Dict[str, str],
    method: str,
    num_steps: int = 15,
    batch_size: int = 64,
    top_k: int = 32,
    micro_batch_size: int = 16,
    draft_filter: Optional[DraftModelFilter] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute single benchmark configuration and capture complete telemetry."""
    dev = device or model_wrapper.device

    # 1. Build prompt and slices
    decomp = model_wrapper.build_prompt_slices(
        context=task["prompt"],
        optim_str="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        target=task["target"],
        use_chat_template=False,
    )

    # 2. Configure optimizer based on method
    use_draft = method == "draft_filtered"
    objective_type = "mellowmax" if method == "mellowmax" else "cross_entropy"

    cfg = OptimizerConfig(
        num_steps=num_steps,
        batch_size=batch_size,
        top_k=top_k,
        micro_batch_size=micro_batch_size,
        device=str(dev),
        enable_draft_filtering=use_draft,
        draft_filtering_ratio=4,
        objective_type=objective_type,
        mellowmax_alpha=1.0,
        early_stopping_loss=0.01,
        seed=42,
    )

    optimizer = CoordinateGradientOptimizer(
        model_wrapper=model_wrapper,
        config=cfg,
        draft_filter=draft_filter if use_draft else None,
    )

    start_mem = get_mps_memory_snapshot(dev)
    t0 = time.time()

    # 3. Execute optimization trajectory
    trajectory = optimizer.optimize_sequence(
        current_ids=decomp.input_ids,
        slices=decomp.slices,
        log_interval=max(1, num_steps // 2),
    )
    wall_time = time.time() - t0
    end_mem = get_mps_memory_snapshot(dev)

    # 4. Post-optimization discrete sensitivity profiling
    profiler = DiscreteSensitivityProfiler(model_wrapper=model_wrapper, device=dev)
    if len(trajectory.best_tokens) == decomp.input_ids.size(1):
        best_full_ids = torch.tensor([trajectory.best_tokens], dtype=torch.long, device=dev)
    else:
        best_full_ids = decomp.input_ids.clone()
        best_full_ids[0, decomp.slices.opt_slice] = torch.tensor(trajectory.best_tokens, dtype=torch.long, device=dev)

    sensitivity_res = profiler.profile_sequence(
        input_ids=best_full_ids,
        slices=decomp.slices,
        target_text=task["target"],
        noise_levels=[0.05, 0.10, 0.15],
        samples_per_noise_level=4,
        strategy=PerturbationType.TOKEN_SUBSTITUTION,
        micro_batch_size=micro_batch_size,
    )

    # 5. Greedy generation check
    with torch.no_grad():
        generated_suffix = model_wrapper.generate(
            input_ids=best_full_ids[:, : decomp.slices.opt_slice.stop],
            max_new_tokens=len(model_wrapper.tokenizer(task["target"]).input_ids) + 5,
        )
        target_matched = task["target"].strip().lower() in generated_suffix.lower()

    step_times = [s.step_time_seconds for s in trajectory.history]
    avg_step_time = float(sum(step_times) / max(1, len(step_times)))

    return {
        "task_id": task["id"],
        "method": method,
        "initial_loss": round(trajectory.initial_loss, 4),
        "best_loss": round(trajectory.best_loss, 4),
        "loss_reduction": round(trajectory.initial_loss - trajectory.best_loss, 4),
        "num_steps": len(trajectory.history),
        "total_wall_time": round(wall_time, 3),
        "avg_step_time": round(avg_step_time, 3),
        "candidates_evaluated": trajectory.total_candidates_evaluated,
        "sharpness_index": round(sensitivity_res.sharpness_index, 4),
        "empirical_curvature": round(sensitivity_res.empirical_curvature, 4),
        "target_matched": target_matched,
        "loss_curve": trajectory.get_loss_curve(),
        "gradient_norms": trajectory.get_gradient_norms(),
        "optimized_suffix": trajectory.best_text,
        "memory_peak_delta_mb": round(end_mem.allocated_mb - start_mem.allocated_mb, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run discrete optimization research benchmark suite.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--synthetic", action="store_true", help="Use local synthetic causal LM for fast verification")
    parser.add_argument("--num-steps", type=int, default=10, help="Optimization steps per benchmark run")
    parser.add_argument("--batch-size", type=int, default=32, help="Candidate pool size B")
    parser.add_argument("--top-k", type=int, default=16, help="Top-k tokens considered")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Forward chunk size")
    parser.add_argument("--output-dir", type=str, default="experiments/results/benchmark")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    device = get_best_device(args.device)
    console.print(
        Panel.fit(
            f"[bold magenta]Discrete Optimization Benchmark Suite[/bold magenta]\n"
            f"Device: {device} | Synthetic: {args.synthetic} | Steps: {args.num_steps} | Candidates: {args.batch_size}",
            title="Benchmark Harness",
            border_style="magenta",
        )
    )

    # Load target model
    model, tokenizer = load_model_and_tokenizer(args.model, device=device, synthetic=args.synthetic)
    wrapper = CausalLMModelWrapper(model=model, tokenizer=tokenizer, device=device)

    # Initialize draft model screener if draft method is tested
    draft_filter = None
    if args.synthetic:
        draft_model_cfg = GPT2Config(vocab_size=1000, n_positions=256, n_embd=48, n_layer=1, n_head=2, use_cache=False)
        draft_model = GPT2LMHeadModel(draft_model_cfg).to(device=device, dtype=torch.float32)
        draft_filter = DraftModelFilter(
            draft_model=draft_model,
            draft_tokenizer=tokenizer,
            config=DraftFilterConfig(filtering_ratio=4, draft_micro_batch_size=args.micro_batch_size),
            device=device,
        )

    methods = ["standard_gcg", "mellowmax", "draft_filtered"] if draft_filter else ["standard_gcg", "mellowmax"]
    results_records: List[Dict[str, Any]] = []

    for task in BENCHMARK_TASKS[:2]:  # Evaluate on first 2 tasks for fast profiling
        for method in methods:
            console.print(f"\n[bold cyan]Benchmarking:[/bold cyan] Task={task['id']} | Method={method}")
            record = run_benchmark_experiment(
                model_wrapper=wrapper,
                task=task,
                method=method,
                num_steps=args.num_steps,
                batch_size=args.batch_size,
                top_k=args.top_k,
                micro_batch_size=args.micro_batch_size,
                draft_filter=draft_filter,
                device=device,
            )
            results_records.append(record)
            clear_device_cache(device)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON summary
    json_path = out_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_records, f, indent=2)

    # Save CSV table
    csv_path = out_dir / "benchmark_summary.csv"
    if results_records:
        csv_fields = [
            "task_id",
            "method",
            "initial_loss",
            "best_loss",
            "loss_reduction",
            "total_wall_time",
            "avg_step_time",
            "candidates_evaluated",
            "sharpness_index",
            "target_matched",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for r in results_records:
                writer.writerow(r)

    # Display comparison table
    table = Table(title="Benchmark Evaluation Results", header_style="bold green")
    table.add_column("Task ID", style="dim")
    table.add_column("Method", style="bold cyan")
    table.add_column("Init Loss", justify="right")
    table.add_column("Best Loss", justify="right", style="yellow")
    table.add_column("Reduction", justify="right", style="green")
    table.add_column("Wall Time (s)", justify="right")
    table.add_column("Avg Step (s)", justify="right")
    table.add_column("Sharpness", justify="right")

    for r in results_records:
        table.add_row(
            r["task_id"],
            r["method"],
            f"{r['initial_loss']:.4f}",
            f"{r['best_loss']:.4f}",
            f"{r['loss_reduction']:.4f}",
            f"{r['total_wall_time']:.2f}",
            f"{r['avg_step_time']:.2f}",
            f"{r['sharpness_index']:.2f}",
        )
    console.print(table)
    console.print(f"[bold green]Benchmark artifacts saved to:[/bold green] {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
