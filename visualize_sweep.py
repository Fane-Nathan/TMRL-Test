"""
Visualization Dashboard for Hyperparameter Optimization Results

Creates interactive visualizations to analyze:
- Optimization history
- Parameter importance
- Hyperparameter interactions
- Stability score distributions

Usage:
    python visualize_sweep.py
    python visualize_sweep.py --results-path C:/Users/felix/TmrlData/hyperparameter_results/optimization_results.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")


def load_results(results_path: Path) -> Dict:
    """Load optimization results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def create_optimization_history_plot(results: Dict, output_dir: Path):
    """Create optimization history visualization."""
    if not PLOTLY_AVAILABLE:
        return

    trials = results["all_trials"]

    trial_numbers = [t["number"] for t in trials if t["value"] is not None]
    scores = [t["value"] for t in trials if t["value"] is not None]

    if not scores:
        print("No completed trials to visualize")
        return

    best_scores = []
    current_best = float("-inf")
    for score in scores:
        if score > current_best:
            current_best = score
        best_scores.append(current_best)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=scores,
            mode="markers",
            name="Trial Score",
            marker=dict(
                size=8,
                color=scores,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Stability Score"),
            ),
            text=[
                f"Trial {n}<br>Score: {s:.4f}" for n, s in zip(trial_numbers, scores)
            ],
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=best_scores,
            mode="lines",
            name="Best Score",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Optimization History: Stability Score Over Trials",
        xaxis_title="Trial Number",
        yaxis_title="Stability Score",
        hovermode="closest",
        template="plotly_white",
        height=600,
    )

    output_path = output_dir / "dashboard_optimization_history.html"
    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


def create_parameter_importance_plot(results: Dict, output_dir: Path):
    """Analyze and visualize parameter importance."""
    if not PLOTLY_AVAILABLE:
        return

    trials = [t for t in results["all_trials"] if t["value"] is not None]

    if len(trials) < 10:
        print("Not enough completed trials for importance analysis (need >= 10)")
        return

    params_data = []
    for trial in trials:
        row = {"score": trial["value"]}
        row.update(trial["params"])
        params_data.append(row)

    df = pd.DataFrame(params_data)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("score")

    if not numeric_cols:
        print("No numeric parameters to analyze")
        return

    correlations = {}
    for col in numeric_cols:
        corr = df["score"].corr(df[col])
        correlations[col] = abs(corr) if not np.isnan(corr) else 0

    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    fig = go.Figure(
        data=[
            go.Bar(
                x=[p[1] for p in sorted_params],
                y=[p[0] for p in sorted_params],
                orientation="h",
                marker=dict(
                    color=[p[1] for p in sorted_params],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="|Correlation|"),
                ),
            )
        ]
    )

    fig.update_layout(
        title="Parameter Importance (Correlation with Stability Score)",
        xaxis_title="Absolute Correlation",
        yaxis_title="Hyperparameter",
        template="plotly_white",
        height=max(400, len(sorted_params) * 30),
        showlegend=False,
    )

    output_path = output_dir / "dashboard_parameter_importance.html"
    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


def create_parallel_coordinates_plot(results: Dict, output_dir: Path):
    """Create parallel coordinates plot showing hyperparameter combinations."""
    if not PLOTLY_AVAILABLE:
        return

    trials = [t for t in results["all_trials"] if t["value"] is not None]

    if len(trials) < 5:
        print("Not enough trials for parallel coordinates plot (need >= 5)")
        return

    params_data = []
    for trial in trials:
        row = {"trial": trial["number"], "stability_score": trial["value"]}
        row.update(trial["params"])
        params_data.append(row)

    df = pd.DataFrame(params_data)

    key_params = [
        "lr_actor",
        "lr_critic",
        "lr_entropy",
        "alpha",
        "alpha_floor",
        "gamma",
        "polyak",
        "batch_size",
        "l2_actor",
    ]
    available_params = [p for p in key_params if p in df.columns]

    if len(available_params) < 3:
        print("Not enough parameters for parallel coordinates")
        return

    dimensions = []
    for param in available_params[:6]:
        dimensions.append(
            dict(
                label=param, values=df[param], range=[df[param].min(), df[param].max()]
            )
        )

    dimensions.append(
        dict(
            label="stability_score",
            values=df["stability_score"],
            range=[df["stability_score"].min(), df["stability_score"].max()],
        )
    )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df["stability_score"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Stability Score"),
            ),
            dimensions=dimensions,
        )
    )

    fig.update_layout(
        title="Parallel Coordinates: Hyperparameter Combinations",
        template="plotly_white",
        height=600,
    )

    output_path = output_dir / "dashboard_parallel_coordinates.html"
    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


def create_score_distribution_plot(results: Dict, output_dir: Path):
    """Visualize distribution of stability scores."""
    if not PLOTLY_AVAILABLE:
        return

    trials = [t for t in results["all_trials"] if t["value"] is not None]
    scores = [t["value"] for t in trials]

    if not scores:
        print("No scores to visualize")
        return

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Score Distribution", "Score by Trial State")
    )

    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=30,
            name="Score Distribution",
            marker_color="blue",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    completed_scores = [
        t["value"] for t in results["all_trials"] if t["state"] == "TrialState.COMPLETE"
    ]
    pruned_count = len(
        [t for t in results["all_trials"] if t["state"] == "TrialState.PRUNED"]
    )
    failed_count = len(
        [t for t in results["all_trials"] if t["state"] == "TrialState.FAIL"]
    )

    states = ["Completed", "Pruned", "Failed"]
    counts = [len(completed_scores), pruned_count, failed_count]
    colors = ["green", "orange", "red"]

    fig.add_trace(
        go.Bar(
            x=states, y=counts, name="Trial States", marker_color=colors, opacity=0.7
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Stability Score", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Trial State", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(
        title_text=f"Score Distribution Summary (Total: {len(results['all_trials'])} trials)",
        showlegend=False,
        template="plotly_white",
        height=500,
    )

    output_path = output_dir / "dashboard_score_distribution.html"
    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


def create_top_trials_comparison(results: Dict, output_dir: Path, top_n: int = 10):
    """Compare top N best trials."""
    if not PLOTLY_AVAILABLE:
        return

    trials = [t for t in results["all_trials"] if t["value"] is not None]

    if len(trials) < top_n:
        top_n = len(trials)

    sorted_trials = sorted(trials, key=lambda x: x["value"], reverse=True)[:top_n]

    params_data = []
    for trial in sorted_trials:
        row = {
            "Trial": f"#{trial['number']}",
            "Score": trial["value"],
            **trial["params"],
        }
        params_data.append(row)

    df = pd.DataFrame(params_data)

    key_params = [
        "lr_actor",
        "lr_critic",
        "alpha",
        "alpha_floor",
        "gamma",
        "batch_size",
    ]
    available_params = [p for p in key_params if p in df.columns]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Trial", "Score"] + available_params,
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=12, weight="bold"),
                ),
                cells=dict(
                    values=[df["Trial"], df["Score"].round(4)]
                    + [
                        df[p].round(6)
                        if df[p].dtype in [np.float64, np.float32]
                        else df[p]
                        for p in available_params
                    ],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Top {top_n} Hyperparameter Configurations",
        template="plotly_white",
        height=max(400, top_n * 35 + 100),
    )

    output_path = output_dir / "dashboard_top_trials.html"
    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


def create_parameter_scatter_matrix(results: Dict, output_dir: Path):
    """Create scatter matrix for key parameters."""
    if not PLOTLY_AVAILABLE:
        return

    trials = [t for t in results["all_trials"] if t["value"] is not None]

    if len(trials) < 10:
        print("Not enough trials for scatter matrix (need >= 10)")
        return

    params_data = []
    for trial in trials:
        row = {"stability_score": trial["value"]}
        row.update(trial["params"])
        params_data.append(row)

    df = pd.DataFrame(params_data)

    key_params = [
        "lr_actor",
        "lr_critic",
        "alpha",
        "gamma",
        "batch_size",
        "stability_score",
    ]
    available_params = [p for p in key_params if p in df.columns]

    if len(available_params) < 3:
        print("Not enough parameters for scatter matrix")
        return

    fig = px.scatter_matrix(
        df,
        dimensions=available_params[:5],
        color="stability_score",
        title="Parameter Relationships Scatter Matrix",
        labels={col: col for col in df.columns},
        color_continuous_scale="Viridis",
    )

    fig.update_layout(template="plotly_white", height=800, width=800)

    output_path = output_dir / "dashboard_scatter_matrix.html"
    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


def create_summary_report(results: Dict, output_dir: Path):
    """Create a text summary report."""
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("HYPERPARAMETER OPTIMIZATION SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append(f"Study Name: {results.get('study_name', 'N/A')}")
    report_lines.append(f"Timestamp: {results.get('timestamp', 'N/A')}")
    report_lines.append(f"Total Trials: {results.get('n_trials', 'N/A')}")
    report_lines.append("")

    best = results.get("best_params", {})
    best_score = results.get("best_stability_score", "N/A")
    best_trial = results.get("best_trial_number", "N/A")

    report_lines.append("-" * 80)
    report_lines.append("BEST CONFIGURATION")
    report_lines.append("-" * 80)
    report_lines.append(f"Trial Number: {best_trial}")
    report_lines.append(
        f"Stability Score: {best_score:.4f}"
        if isinstance(best_score, float)
        else f"Stability Score: {best_score}"
    )
    report_lines.append("")
    report_lines.append("Hyperparameters:")
    for key, value in sorted(best.items()):
        if isinstance(value, float):
            report_lines.append(f"  {key:30s}: {value:.6e}")
        else:
            report_lines.append(f"  {key:30s}: {value}")
    report_lines.append("")

    trials = results.get("all_trials", [])
    completed = len([t for t in trials if t.get("state") == "TrialState.COMPLETE"])
    pruned = len([t for t in trials if t.get("state") == "TrialState.PRUNED"])
    failed = len([t for t in trials if t.get("state") == "TrialState.FAIL"])

    report_lines.append("-" * 80)
    report_lines.append("TRIAL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Completed: {completed}")
    report_lines.append(f"Pruned: {pruned}")
    report_lines.append(f"Failed: {failed}")
    report_lines.append(f"Success Rate: {100 * completed / max(len(trials), 1):.1f}%")
    report_lines.append("")

    scores = [t["value"] for t in trials if t["value"] is not None]
    if scores:
        report_lines.append("Score Statistics:")
        report_lines.append(f"  Mean:   {np.mean(scores):.4f}")
        report_lines.append(f"  Median: {np.median(scores):.4f}")
        report_lines.append(f"  Std:    {np.std(scores):.4f}")
        report_lines.append(f"  Min:    {np.min(scores):.4f}")
        report_lines.append(f"  Max:    {np.max(scores):.4f}")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("")

    report_text = "\n".join(report_lines)

    output_path = output_dir / "summary_report.txt"
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"  Saved: {output_path}")
    print("\n" + report_text)


def create_all_visualizations(results_path: Path, output_dir: Optional[Path] = None):
    """Create all visualization dashboards."""

    if output_dir is None:
        output_dir = results_path.parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from: {results_path}")
    results = load_results(results_path)

    print(f"\nCreating visualizations in: {output_dir}")
    print("-" * 60)

    create_optimization_history_plot(results, output_dir)
    create_parameter_importance_plot(results, output_dir)
    create_parallel_coordinates_plot(results, output_dir)
    create_score_distribution_plot(results, output_dir)
    create_top_trials_comparison(results, output_dir, top_n=10)
    create_parameter_scatter_matrix(results, output_dir)
    create_summary_report(results, output_dir)

    print("-" * 60)
    print("\nVisualization complete! Open the HTML files in your browser.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hyperparameter optimization results"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Path to optimization_results.json (default: ~/TmrlData/hyperparameter_results/optimization_results.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: same as results)",
    )

    args = parser.parse_args()

    if args.results_path:
        results_path = Path(args.results_path)
    else:
        results_path = (
            Path.home()
            / "TmrlData"
            / "hyperparameter_results"
            / "optimization_results.json"
        )

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("\nRun hyperparameter_sweep_optuna.py first to generate results.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None

    create_all_visualizations(results_path, output_dir)


if __name__ == "__main__":
    main()
