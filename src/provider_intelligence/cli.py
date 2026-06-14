"""Command-line interface for the provider intelligence pipeline."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from provider_intelligence.config import ensure_outputs_dir
from provider_intelligence.cost_model import write_cost_estimate
from provider_intelligence.data_generation import generate_demo_data
from provider_intelligence.evaluation import run_evaluation
from provider_intelligence.pipeline import run_pipeline

app = typer.Typer(
    name="provider-intelligence",
    help="Offline-first provider directory validation pipeline.",
    no_args_is_help=True,
)
console = Console()


def _do_generate_demo_data(output_dir: Path | None = None) -> dict[str, Path]:
    outputs = output_dir or ensure_outputs_dir()
    paths = generate_demo_data(outputs)
    console.print("[green]Demo data generated:[/green]")
    for name, path in paths.items():
        console.print(f"  {name}: {path}")
    return paths


def _do_run_pipeline(
    mode: str = "demo",
    generate_data: bool = False,
    output_dir: Path | None = None,
) -> dict:
    outputs = output_dir or ensure_outputs_dir()
    result = run_pipeline(mode=mode, generate_data=generate_data, output_dir=outputs)
    console.print("[green]Pipeline complete[/green]")
    console.print(f"  Records processed: {result['records_processed']}")
    console.print(f"  Action counts: {result['action_counts']}")
    console.print(f"  Recommendations: {result['recommendations_path']}")
    return result


def _do_evaluate(
    output_dir: Path | None = None,
    compare_llm: bool = False,
    rule_based_dir: Path | None = None,
    adaptive_dir: Path | None = None,
) -> dict:
    outputs = output_dir or ensure_outputs_dir()
    result = run_evaluation(
        output_dir=outputs,
        compare_llm=compare_llm,
        rule_based_dir=rule_based_dir,
        adaptive_dir=adaptive_dir,
    )
    metrics = result["metrics"]
    console.print("[green]Evaluation complete[/green]")
    console.print(f"  false_auto_update_rate: {metrics['false_auto_update_rate']}")
    console.print(f"  action_accuracy: {metrics['action_accuracy']}")
    console.print(f"  Output: {result['output_path']}")
    if "llm_comparison" in result:
        comp = result["llm_comparison"]
        console.print(f"  Safety preserved: {comp['safety_constraint_met']}")
    return result


def _do_estimate_cost(
    record_count: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    outputs = output_dir or ensure_outputs_dir()
    result = write_cost_estimate(output_dir=outputs, record_count=record_count)
    console.print("[green]Cost estimate written[/green]")
    for mode, data in result["by_mode"].items():
        console.print(f"  {mode}: ${data['total_estimated_cost_usd']} total")
    console.print(f"  Output: {result['output_path']}")
    return result


@app.command("generate-demo-data")
def generate_demo_data_cmd(
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Outputs directory."),
) -> None:
    """Generate synthetic directory data and ground truth labels."""
    _do_generate_demo_data(output_dir)


@app.command("run-pipeline")
def run_pipeline_cmd(
    mode: str = typer.Option("demo", "--mode", help="Pipeline mode: demo or real."),
    generate_data: bool = typer.Option(False, "--generate-data", help="Regenerate demo data first."),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Outputs directory."),
    llm_mode: str | None = typer.Option(None, "--llm-mode", help="Override LLM_MODE (off|auto|force)."),
) -> None:
    """Run the full provider intelligence pipeline."""
    if llm_mode:
        import os

        os.environ["LLM_MODE"] = llm_mode
    _do_run_pipeline(mode=mode, generate_data=generate_data, output_dir=output_dir)


@app.command("evaluate")
def evaluate_cmd(
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Outputs directory."),
    compare_llm: bool = typer.Option(False, "--compare-llm", help="Compare rule-only vs adaptive LLM."),
    rule_based_dir: Path | None = typer.Option(None, "--rule-based-dir", help="Rule-only outputs dir."),
    adaptive_dir: Path | None = typer.Option(None, "--adaptive-dir", help="Adaptive LLM outputs dir."),
) -> None:
    """Run synthetic benchmark evaluation."""
    _do_evaluate(
        output_dir=output_dir,
        compare_llm=compare_llm,
        rule_based_dir=rule_based_dir,
        adaptive_dir=adaptive_dir,
    )


@app.command("estimate-cost")
def estimate_cost_cmd(
    record_count: int | None = typer.Option(None, "--records", help="Number of records to estimate."),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Outputs directory."),
) -> None:
    """Estimate operational costs per LLM mode."""
    _do_estimate_cost(record_count=record_count, output_dir=output_dir)


@app.command("run-all")
def run_all_cmd(
    compare_llm: bool = typer.Option(False, "--compare-llm", help="Include LLM comparison in evaluation."),
    llm_mode: str | None = typer.Option(None, "--llm-mode", help="Override LLM_MODE (off|auto|force)."),
) -> None:
    """Run full demo workflow: data, pipeline, evaluate, cost."""
    if llm_mode:
        import os

        os.environ["LLM_MODE"] = llm_mode
    _do_generate_demo_data()
    pipeline_result = _do_run_pipeline(generate_data=False)
    _do_evaluate(compare_llm=compare_llm)
    _do_estimate_cost(record_count=pipeline_result["records_processed"])
    console.print("[bold green]run-all complete[/bold green]")


def main() -> None:
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    main()
