#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_train_log_plot_module():
    module_path = Path(__file__).resolve().parent / "train_log_plot.py"
    if not module_path.exists():
        raise RuntimeError(f"missing module file: {module_path}")
    spec = importlib.util.spec_from_file_location("train_log_plot", module_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_log_plot"] = module
    spec.loader.exec_module(module)
    return module


def _normalize_metric(metric: str) -> str:
    alias_map = {
        "memory": "memory_gib",
        "mfu": "mfu_pct",
        "elapsed": "elapsed_time_per_step",
    }
    return alias_map.get(metric, metric)


def _parse_metric_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [_normalize_metric(item.strip()) for item in raw.split(",") if item.strip()]


PR_IMAGE_WIDTH = 1024
PR_IMAGE_HEIGHT = 768
PR_IMAGE_DPI = 100


def _default_output_path(log_a: str, log_b: str | None, output_format: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_a = Path(log_a).stem
    if log_b:
        base_b = Path(log_b).stem
        filename = f"{base_a}_vs_{base_b}_{stamp}.{output_format}"
    else:
        filename = f"{base_a}_{stamp}.{output_format}"
    return Path.cwd() / filename


def _default_pr_image_output_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_pr_1024x768.png")


def _resolve_pr_image_output(
    output: Path,
    *,
    generate_pr_image: bool,
    pr_image_output: str | None,
) -> Path | None:
    if not generate_pr_image and not pr_image_output:
        return None
    pr_output = (
        Path(pr_image_output)
        if pr_image_output
        else _default_pr_image_output_path(output)
    )
    if pr_output.suffix.lower() != ".png":
        pr_output = pr_output.with_suffix(".png")
    return pr_output


def _require_matplotlib(no_show: bool):
    try:
        import matplotlib

        if no_show:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required for plotting. Please install it first: pip install matplotlib"
        ) from exc


def _metric_label(metric_key: str) -> str:
    labels = {
        "loss": "loss",
        "grad_norm": "grad_norm",
        "memory_gib": "memory (GB)",
        "memory_pct": "memory (%)",
        "tps": "tps",
        "tflops": "tflops",
        "mfu_pct": "mfu (%)",
        "elapsed_time_per_step": "elapsed_time_per_step (s)",
        "loss_abs_error": "loss abs error",
        "loss_rel_error": "loss relative error",
        "grad_norm_abs_error": "grad_norm abs error",
        "grad_norm_rel_error": "grad_norm relative error",
    }
    return labels.get(metric_key, metric_key)


def _compute_error_stats(errors: list[float]) -> dict[str, float]:
    """Compute error statistics: mean, MSE, min, max."""
    if not errors:
        return {"mean": 0.0, "mse": 0.0, "min": 0.0, "max": 0.0}

    mean_err = sum(errors) / len(errors)
    mse = sum(e * e for e in errors) / len(errors)
    min_err = min(errors)
    max_err = max(errors)

    return {
        "mean": mean_err,
        "mse": mse,
        "min": min_err,
        "max": max_err,
    }


def _plot_single_log(
    plt,
    module,
    *,
    records,
    metrics: list[str],
    title: str,
    output: Path,
    pr_image_output: Path | None,
    no_show: bool,
):
    """Plot single log with 2-column layout."""
    selected = ["loss", "grad_norm"] + [
        metric for metric in metrics if metric not in {"loss", "grad_norm"}
    ]
    n_metrics = len(selected)

    # Calculate grid: 2 columns, enough rows
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 3.5 * n_rows), sharex=False
    )
    if n_metrics == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    # Flatten axes for easier iteration
    if n_rows == 1:
        axes_flat = list(axes[0])
    else:
        axes_flat = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]

    warnings: list[str] = []
    for idx, metric in enumerate(selected):
        axis = axes_flat[idx]
        steps, values = module.extract_metric_series(records, metric)
        if not steps:
            warnings.append(f"metric '{metric}' not found in log")
            axis.text(
                0.5,
                0.5,
                f"{metric} not available",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_ylabel(_metric_label(metric))
            axis.grid(True, linestyle="--", alpha=0.4)
            axis.set_xticks([])
            continue
        axis.plot(steps, values, label=metric, linewidth=1.2)
        axis.set_ylabel(_metric_label(metric))
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="best", fontsize=8)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    try:
        figure.savefig(output, dpi=150, bbox_inches="tight")
    except OSError as error:
        plt.close(figure)
        raise RuntimeError(f"failed to save plot to '{output}': {error}") from error

    if pr_image_output is not None:
        original_width, original_height = figure.get_size_inches()
        try:
            figure.set_size_inches(
                PR_IMAGE_WIDTH / PR_IMAGE_DPI,
                PR_IMAGE_HEIGHT / PR_IMAGE_DPI,
            )
            figure.savefig(pr_image_output, dpi=PR_IMAGE_DPI)
        except OSError as error:
            plt.close(figure)
            raise RuntimeError(
                f"failed to save PR image to '{pr_image_output}': {error}"
            ) from error
        finally:
            figure.set_size_inches(original_width, original_height)

    if not no_show:
        plt.show()
    plt.close(figure)
    return warnings


def _plot_compare(
    plt,
    module,
    *,
    records_a,
    records_b,
    metrics: list[str],
    baseline: str,
    title: str,
    output: Path,
    pr_image_output: Path | None,
    no_show: bool,
    log_a_name: str = "a",
    log_b_name: str = "b",
):
    """Plot comparison with 2-column layout and error curves under each metric."""
    aligned = module.align_by_common_steps(records_a, records_b)
    warnings: list[str] = []
    if aligned.missing_in_a:
        warnings.append(f"steps only in log-b: {len(aligned.missing_in_a)}")
    if aligned.missing_in_b:
        warnings.append(f"steps only in log-a: {len(aligned.missing_in_b)}")
    if not aligned.steps:
        warnings.append("no common steps between log-a and log-b")
        return warnings

    # Required metrics with error tracking
    required_metrics = ["loss", "grad_norm"]
    optional_metrics = [m for m in metrics if m not in required_metrics]

    # Compute signed errors for loss and grad_norm
    losses_a = [float(r["loss"]) for r in aligned.records_a]
    losses_b = [float(r["loss"]) for r in aligned.records_b]
    loss_abs_err, loss_rel_err = module.compute_signed_errors(
        losses_a, losses_b, baseline=baseline
    )

    grad_norms_a = [float(r.get("grad_norm", 0)) for r in aligned.records_a]
    grad_norms_b = [float(r.get("grad_norm", 0)) for r in aligned.records_b]
    grad_abs_err, grad_rel_err = module.compute_signed_errors(
        grad_norms_a, grad_norms_b, baseline=baseline
    )

    # Compute error statistics
    loss_abs_stats = _compute_error_stats(loss_abs_err)
    loss_rel_stats = _compute_error_stats(loss_rel_err)
    grad_abs_stats = _compute_error_stats(grad_abs_err)
    grad_rel_stats = _compute_error_stats(grad_rel_err)

    # Build subplot list: each main metric followed by its error curves
    # Layout: 2 columns
    # Row 0: loss, grad_norm
    # Row 1: loss_abs_error, grad_norm_abs_error
    # Row 2: loss_rel_error, grad_norm_rel_error
    # Then optional metrics in subsequent rows

    n_optional_rows = (len(optional_metrics) + 1) // 2
    n_rows = 3 + n_optional_rows  # 3 rows for loss+grad_norm with their errors

    figure, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Row 0: loss and grad_norm
    # Loss subplot
    ax_loss = axes[0][0]
    loss_a_vals = [float(r.get("loss", float("nan"))) for r in aligned.records_a]
    loss_b_vals = [float(r.get("loss", float("nan"))) for r in aligned.records_b]
    ax_loss.plot(aligned.steps, loss_a_vals, label=log_a_name, linewidth=1.2)
    ax_loss.plot(aligned.steps, loss_b_vals, label=log_b_name, linewidth=1.2)
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend(loc="best", fontsize=8)

    # Grad norm subplot
    ax_grad = axes[0][1]
    grad_a_vals = [float(r.get("grad_norm", float("nan"))) for r in aligned.records_a]
    grad_b_vals = [float(r.get("grad_norm", float("nan"))) for r in aligned.records_b]
    ax_grad.plot(aligned.steps, grad_a_vals, label=log_a_name, linewidth=1.2)
    ax_grad.plot(aligned.steps, grad_b_vals, label=log_b_name, linewidth=1.2)
    ax_grad.set_ylabel("grad_norm")
    ax_grad.grid(True, linestyle="--", alpha=0.4)
    ax_grad.legend(loc="best", fontsize=8)

    # Row 1: absolute errors (no threshold lines)
    # Loss abs error
    ax_loss_abs = axes[1][0]
    ax_loss_abs.plot(
        aligned.steps, loss_abs_err, label="loss abs error", color="blue", linewidth=1.0
    )
    ax_loss_abs.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_loss_abs.set_ylabel("loss abs error")
    ax_loss_abs.grid(True, linestyle="--", alpha=0.4)
    ax_loss_abs.legend(loc="best", fontsize=8)

    # Add statistics text below the plot (larger font)
    stats_text = (
        f"mean={loss_abs_stats['mean']:.5f}, mse={loss_abs_stats['mse']:.5f}, "
        f"min={loss_abs_stats['min']:.5f}, max={loss_abs_stats['max']:.5f}"
    )
    ax_loss_abs.annotate(
        stats_text,
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="navy",
    )

    # Grad norm abs error
    ax_grad_abs = axes[1][1]
    ax_grad_abs.plot(
        aligned.steps,
        grad_abs_err,
        label="grad_norm abs error",
        color="blue",
        linewidth=1.0,
    )
    ax_grad_abs.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_grad_abs.set_ylabel("grad_norm abs error")
    ax_grad_abs.grid(True, linestyle="--", alpha=0.4)
    ax_grad_abs.legend(loc="best", fontsize=8)

    # Add statistics text (larger font)
    stats_text = (
        f"mean={grad_abs_stats['mean']:.5f}, mse={grad_abs_stats['mse']:.5f}, "
        f"min={grad_abs_stats['min']:.5f}, max={grad_abs_stats['max']:.5f}"
    )
    ax_grad_abs.annotate(
        stats_text,
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="navy",
    )

    # Row 2: relative errors (with threshold lines)
    # Loss rel error with threshold lines
    ax_loss_rel = axes[2][0]
    baseline_name = log_a_name if baseline == "a" else log_b_name
    ax_loss_rel.plot(
        aligned.steps,
        loss_rel_err,
        label=f"loss relative error (baseline={baseline_name})",
        color="green",
        linewidth=1.0,
    )
    ax_loss_rel.axhline(
        y=0.02, color="red", linestyle="--", linewidth=1.5, label="threshold ±0.02"
    )
    ax_loss_rel.axhline(y=-0.02, color="red", linestyle="--", linewidth=1.5)
    ax_loss_rel.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_loss_rel.set_ylabel("loss relative error")
    ax_loss_rel.grid(True, linestyle="--", alpha=0.4)
    ax_loss_rel.legend(loc="best", fontsize=8)

    # Add statistics text (larger font)
    stats_text = (
        f"mean={loss_rel_stats['mean']:.5f}, mse={loss_rel_stats['mse']:.5f}, "
        f"min={loss_rel_stats['min']:.5f}, max={loss_rel_stats['max']:.5f}"
    )
    ax_loss_rel.annotate(
        stats_text,
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="navy",
    )

    # Grad norm rel error with threshold lines
    ax_grad_rel = axes[2][1]
    ax_grad_rel.plot(
        aligned.steps,
        grad_rel_err,
        label=f"grad_norm relative error (baseline={baseline_name})",
        color="green",
        linewidth=1.0,
    )
    ax_grad_rel.axhline(
        y=0.02, color="red", linestyle="--", linewidth=1.5, label="threshold ±0.02"
    )
    ax_grad_rel.axhline(y=-0.02, color="red", linestyle="--", linewidth=1.5)
    ax_grad_rel.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_grad_rel.set_ylabel("grad_norm relative error")
    ax_grad_rel.grid(True, linestyle="--", alpha=0.4)
    ax_grad_rel.legend(loc="best", fontsize=8)

    # Add statistics text (larger font)
    stats_text = (
        f"mean={grad_rel_stats['mean']:.5f}, mse={grad_rel_stats['mse']:.5f}, "
        f"min={grad_rel_stats['min']:.5f}, max={grad_rel_stats['max']:.5f}"
    )
    ax_grad_rel.annotate(
        stats_text,
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="navy",
    )

    # Remaining rows: optional metrics
    for idx, metric in enumerate(optional_metrics):
        row = 3 + idx // 2
        col = idx % 2
        axis = axes[row][col]

        values_a = []
        values_b = []
        for record_a, record_b in zip(
            aligned.records_a, aligned.records_b, strict=True
        ):
            value_a = record_a.get(metric)
            value_b = record_b.get(metric)
            values_a.append(float(value_a) if value_a is not None else float("nan"))
            values_b.append(float(value_b) if value_b is not None else float("nan"))

        has_valid_a = any(not math.isnan(value) for value in values_a)
        has_valid_b = any(not math.isnan(value) for value in values_b)
        if not has_valid_a and not has_valid_b:
            warnings.append(f"metric '{metric}' not found in either log")
            axis.text(
                0.5,
                0.5,
                f"{metric} not available",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_ylabel(_metric_label(metric))
            axis.grid(True, linestyle="--", alpha=0.4)
            axis.set_xticks([])
            continue

        axis.plot(aligned.steps, values_a, label=log_a_name, linewidth=1.2)
        axis.plot(aligned.steps, values_b, label=log_b_name, linewidth=1.2)
        axis.set_ylabel(_metric_label(metric))
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="best", fontsize=8)

    # Hide unused subplots in optional metrics section
    n_optional_plotted = len(optional_metrics)
    for idx in range(n_optional_plotted, n_optional_rows * 2):
        row = 3 + idx // 2
        col = idx % 2
        if row < n_rows:
            axes[row][col].axis("off")

    # Set xlabel for bottom row
    for col in range(2):
        axes[-1][col].set_xlabel("step")

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    try:
        figure.savefig(output, dpi=150, bbox_inches="tight")
        if pr_image_output is not None:
            figure.set_size_inches(
                PR_IMAGE_WIDTH / PR_IMAGE_DPI,
                PR_IMAGE_HEIGHT / PR_IMAGE_DPI,
            )
            figure.savefig(pr_image_output, dpi=PR_IMAGE_DPI)
    except OSError as error:
        plt.close(figure)
        raise RuntimeError(f"failed to save plot to '{output}': {error}") from error

    if not no_show:
        plt.show()
    plt.close(figure)
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot training metrics from torchtitan-npu logs"
    )
    parser.add_argument("--log-a", required=True, help="path to primary log")
    parser.add_argument("--log-b", default=None, help="path to comparison log")
    parser.add_argument(
        "--metrics",
        default="",
        help="optional metrics: memory_gib,memory_pct,tps,tflops,mfu_pct,elapsed_time_per_step",
    )
    parser.add_argument("--output", default=None, help="output image path")
    parser.add_argument(
        "--generate-pr-image",
        action="store_true",
        help="also export an extra 1024x768 PNG for PR embedding",
    )
    parser.add_argument(
        "--pr-image-output",
        default=None,
        help="optional path for the 1024x768 PR image (.png)",
    )
    parser.add_argument("--title", default=None, help="figure title")
    parser.add_argument(
        "--baseline",
        choices=["a", "b"],
        default="a",
        help="baseline for relative error (default: a, i.e., log-a)",
    )
    parser.add_argument(
        "--format", choices=["png", "pdf"], default="png", help="output format"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="disable interactive display"
    )
    args = parser.parse_args()

    module = _load_train_log_plot_module()

    try:
        records_a, warnings_a = module.read_training_metrics(args.log_a)
    except OSError as error:
        logger.info(f"[error] failed to read log-a '{args.log_a}': {error}")
        return 1
    if not records_a:
        logger.info(f"[error] no valid training metrics found in log-a: {args.log_a}")
        return 1

    records_b = None
    warnings_b: list[str] = []
    if args.log_b:
        try:
            records_b, warnings_b = module.read_training_metrics(args.log_b)
        except OSError as error:
            logger.info(f"[error] failed to read log-b '{args.log_b}': {error}")
            return 1
        if not records_b:
            logger.info(
                f"[error] no valid training metrics found in log-b: {args.log_b}"
            )
            return 1

    output = (
        Path(args.output)
        if args.output
        else _default_output_path(args.log_a, args.log_b, args.format)
    )
    if output.suffix.lower() != f".{args.format}":
        output = output.with_suffix(f".{args.format}")

    pr_image_output = _resolve_pr_image_output(
        output,
        generate_pr_image=args.generate_pr_image,
        pr_image_output=args.pr_image_output,
    )

    metrics = _parse_metric_list(args.metrics)

    # Get log file names for legend
    log_a_name = Path(args.log_a).stem
    log_b_name = Path(args.log_b).stem if args.log_b else None

    title = args.title or (
        f"Training Metrics: {log_a_name} vs {log_b_name}"
        if args.log_b
        else f"Training Metrics: {log_a_name}"
    )

    try:
        plt = _require_matplotlib(args.no_show)
    except RuntimeError as error:
        logger.info(f"[error] {error}")
        return 1

    warnings = [*warnings_a, *warnings_b]
    try:
        if records_b is None:
            warnings.extend(
                _plot_single_log(
                    plt,
                    module,
                    records=records_a,
                    metrics=metrics,
                    title=title,
                    output=output,
                    pr_image_output=pr_image_output,
                    no_show=args.no_show,
                )
            )
        else:
            warnings.extend(
                _plot_compare(
                    plt,
                    module,
                    records_a=records_a,
                    records_b=records_b,
                    metrics=metrics,
                    baseline=args.baseline,
                    title=title,
                    output=output,
                    pr_image_output=pr_image_output,
                    no_show=args.no_show,
                    log_a_name=log_a_name,
                    log_b_name=log_b_name or "b",
                )
            )
    except RuntimeError as error:
        logger.info(f"[error] {error}")
        return 1

    if records_b is not None and "no common steps between log-a and log-b" in warnings:
        logger.info("[error] no common steps between log-a and log-b")
        return 1

    logger.info(f"[ok] plot saved to: {output}")
    if pr_image_output is not None:
        logger.info(f"[ok] pr image saved to: {pr_image_output}")
    summary_a = module.summarize_records(records_a)
    logger.info(
        f"[summary-a] steps={summary_a['num_steps']} "
        f"range=[{summary_a['first_step']},{summary_a['last_step']}] "
        f"loss={summary_a['first_loss']:.5f}->{summary_a['last_loss']:.5f} "
        f"grad_norm={summary_a['first_grad_norm']:.4f}->{summary_a['last_grad_norm']:.4f}"
    )
    if records_b is not None:
        summary_b = module.summarize_records(records_b)
        logger.info(
            f"[summary-b] steps={summary_b['num_steps']} "
            f"range=[{summary_b['first_step']},{summary_b['last_step']}] "
            f"loss={summary_b['first_loss']:.5f}->{summary_b['last_loss']:.5f} "
            f"grad_norm={summary_b['first_grad_norm']:.4f}->{summary_b['last_grad_norm']:.4f}"
        )

    for warning in warnings:
        logger.info(f"[warning] {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
