"""Standalone experiment runner for Greedy Coordinate Gradient (GCG) discrete optimization.

Executes coordinate gradient search on target instruction models or local synthetic
architectures with real-time terminal telemetry and unified memory budgeting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from src.models.causal_lm import CausalLMModelWrapper
from src.optimizers.base import OptimizerConfig
from src.optimizers.coordinate_gradient import CoordinateGradientOptimizer
from src.utils.device import get_best_device, get_mps_memory_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_optimization")
console = Console()


def build_synthetic_model_and_tokenizer(device: torch.device):
    """Instantiate a local lightweight causal LM for instant testing without HF weight downloads."""
    console.print("[dim]Instantiating local synthetic causal LM for fast verification...[/dim]")
    vocab_size = 2000
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_embd=128,
        n_layer=4,
        n_head=4,
        use_cache=False,
    )
    model = GPT2LMHeadModel(config).to(device=device, dtype=torch.float32)

    # Use standard GPT2 tokenizer or a mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run discrete GCG token optimization on Causal LM.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HF model name or path")
    parser.add_argument("--prompt", type=str, default="Explain the fundamental theorem of calculus", help="Context prompt")
    parser.add_argument("--target", type=str, default="The fundamental theorem connects differentiation and integration.", help="Target completion")
    parser.add_argument("--optim-init", type=str, default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", help="Initial optimizable token string")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of discrete optimization steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Candidate pool batch size B")
    parser.add_argument("--top-k", type=int, default=128, help="Top-k replacement tokens per coordinate")
    parser.add_argument("--micro-batch-size", type=int, default=32, help="Micro-batch chunk size for memory safety")
    parser.add_argument("--device", type=str, default="mps", help="Compute device ('mps' or 'cpu')")
    parser.add_argument("--dtype", type=str, default="float16", help="Model precision ('float16', 'bfloat16', 'float32')")
    parser.add_argument("--synthetic", action="store_true", help="Run with local lightweight synthetic causal LM")
    parser.add_argument("--output-json", type=str, default="experiments/results/last_run.json", help="Path to save trajectory")

    args = parser.parse_args()

    device = get_best_device(args.device)
    snapshot = get_mps_memory_snapshot(device)

    console.print(
        Panel.fit(
            f"[bold green]Discrete Coordinate Gradient (GCG) Optimizer[/bold green]\n"
            f"[cyan]Device:[/cyan] {device} | [cyan]Total System RAM:[/cyan] {snapshot.system_total_mb:.1f} MB\n"
            f"[cyan]Model Target:[/cyan] {'Synthetic (Local GPT2)' if args.synthetic else args.model}\n"
            f"[cyan]Steps:[/cyan] {args.num_steps} | [cyan]Candidates B:[/cyan] {args.batch_size} | "
            f"[cyan]Micro-Batch:[/cyan] {args.micro_batch_size}",
            title="Experiment Configuration",
            border_style="bright_blue",
        )
    )

    # 1. Load Model and Tokenizer
    if args.synthetic:
        model, tokenizer = build_synthetic_model_and_tokenizer(device)
    else:
        torch_dtype = getattr(torch, args.dtype, torch.float16)
        console.print(f"[cyan]Loading Hugging Face model:[/cyan] {args.model} ({torch_dtype})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                device_map=str(device),
                trust_remote_code=True,
            )
        except Exception as e:
            console.print(f"[yellow]Could not load '{args.model}' from HuggingFace hub ({e}). Falling back to synthetic model.[/yellow]")
            model, tokenizer = build_synthetic_model_and_tokenizer(device)

    # 2. Wrap Model
    wrapper = CausalLMModelWrapper(model=model, tokenizer=tokenizer, device=device)

    # 3. Build and verify token slices
    decomp = wrapper.build_prompt_slices(
        context=args.prompt,
        optim_str=args.optim_init,
        target=args.target,
        use_chat_template=not args.synthetic,
    )

    console.print(
        f"[bold]Prompt Slices:[/bold] Total Tokens={decomp.input_ids.size(1)} | "
        f"Opt Slice={decomp.slices.opt_slice} ({decomp.slices.opt_len} tokens) | "
        f"Target Slice={decomp.slices.target_slice} ({decomp.slices.target_len} tokens)"
    )

    # 4. Configure Optimizer
    opt_config = OptimizerConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        top_k=args.top_k,
        micro_batch_size=args.micro_batch_size,
        device=str(device),
        early_stopping_loss=0.01,
        seed=42,
    )
    optimizer = CoordinateGradientOptimizer(model_wrapper=wrapper, config=opt_config)

    # 5. Run Optimization Loop
    trajectory = optimizer.optimize_sequence(
        current_ids=decomp.input_ids,
        slices=decomp.slices,
        log_interval=5,
    )

    # 6. Save Trajectory
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trajectory.to_dict(), f, indent=2)

    console.print(
        Panel.fit(
            f"[bold green]Optimization Finished![/bold green]\n"
            f"[cyan]Initial Loss:[/cyan] {trajectory.initial_loss:.4f} -> [cyan]Best Loss:[/cyan] {trajectory.best_loss:.4f}\n"
            f"[cyan]Best Token Suffix:[/cyan] {trajectory.best_text}\n"
            f"[cyan]Wall Time:[/cyan] {trajectory.total_wall_time:.2f}s | "
            f"[cyan]Total Candidates Evaluated:[/cyan] {trajectory.total_candidates_evaluated}\n"
            f"[cyan]Saved to:[/cyan] {out_path}",
            title="Results Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
