from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_dir


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def available_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """
    Return candidate columns that exist and contain at least one non-null value.
    """
    return [
        col
        for col in candidates
        if col in df.columns and df[col].notna().any()
    ]


def _available_metric_pairs(
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
) -> list[tuple[str, str]]:
    """
    Return metric/label pairs where the metric column exists and is not empty.
    """
    return [
        (col, label)
        for col, label in zip(columns, labels)
        if col in df.columns and df[col].notna().any()
    ]


def _finalize_and_save(
    fig: plt.Figure,
    ax: plt.Axes,
    output_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    dpi: int,
    y_min: float | None = None,
    y_max: float | None = None,
    show_legend: bool = True,
    xtick_positions: list[int] | None = None,
    xtick_labels: list[str] | None = None,
) -> None:
    """
    Apply common plot formatting and save the figure.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)

    if xtick_positions is not None and xtick_labels is not None:
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right")

    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"[INFO] Wrote {output_path}")


# ---------------------------------------------------------------------
# Single-run / direct metric plots
# ---------------------------------------------------------------------


def plot_metric_lines(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: str | Path,
    dpi: int = 300,
    y_min: float | None = None,
    y_max: float | None = None,
    marker: str | None = "o",
    linewidth: float = 2.0,
) -> None:
    """
    Plot one or more metric curves from a DataFrame.
    """
    if x_col not in df.columns:
        raise KeyError(f"DataFrame must contain {x_col!r} column.")

    available = _available_metric_pairs(df, y_cols, labels)

    if not available:
        print(f"[WARN] No available columns for plot: {title}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for col, label in available:
        ax.plot(
            df[x_col],
            df[col],
            marker=marker,
            linewidth=linewidth,
            label=label,
        )

    _finalize_and_save(
        fig=fig,
        ax=ax,
        output_path=output_path,
        title=title,
        xlabel=x_col,
        ylabel=ylabel,
        dpi=dpi,
        y_min=y_min,
        y_max=y_max,
        show_legend=True,
    )


def plot_loss_curves(
    train_loss: pd.DataFrame,
    eval_loss: pd.DataFrame,
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Plot train loss and eval loss curves.
    """
    if train_loss.empty and eval_loss.empty:
        print("[WARN] No train/eval loss data found. Skipping loss plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    if not train_loss.empty:
        ax.plot(
            train_loss["epoch"],
            train_loss["loss"],
            label="train loss",
        )

    if not eval_loss.empty:
        ax.plot(
            eval_loss["epoch"],
            eval_loss["eval_loss"],
            marker="o",
            label="eval loss",
        )

    _finalize_and_save(
        fig=fig,
        ax=ax,
        output_path=output_path,
        title="Training and validation loss",
        xlabel="epoch",
        ylabel="loss",
        dpi=dpi,
        show_legend=True,
    )


def plot_checkpoint_comparison_lines(
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: str | Path,
    label_col: str = "epoch",
    xlabel: str = "epoch",
    dpi: int = 300,
    y_min: float | None = 0.0,
    y_max: float | None = 1.05,
    marker_size: float = 5.0,
    linewidth: float = 2.0,
) -> None:
    """
    Plot several metrics across selected checkpoints.

    The x-axis is categorical/equally spaced, but the displayed tick labels
    should usually be selected epoch numbers.
    """
    if label_col not in df.columns:
        raise KeyError(f"DataFrame must contain {label_col!r} column.")

    available = _available_metric_pairs(df, columns, labels)

    if not available:
        print(f"[WARN] No available columns for plot: {title}")
        return

    x_positions = list(range(len(df)))
    x_labels = df[label_col].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(12, 6))

    for col, label in available:
        ax.plot(
            x_positions,
            df[col],
            marker="o",
            markersize=marker_size,
            linewidth=linewidth,
            label=label,
        )

    _finalize_and_save(
        fig=fig,
        ax=ax,
        output_path=output_path,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        dpi=dpi,
        y_min=y_min,
        y_max=y_max,
        show_legend=True,
        xtick_positions=x_positions,
        xtick_labels=x_labels,
    )


def plot_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: str | Path,
    label_col: str = "epoch",
    xlabel: str = "epoch",
    dpi: int = 300,
) -> None:
    """
    Plot a simple categorical bar chart.
    """
    if value_col not in df.columns or not df[value_col].notna().any():
        print(f"[WARN] Missing/empty column {value_col}; skipping {title}")
        return

    if label_col not in df.columns:
        raise KeyError(f"DataFrame must contain {label_col!r} column.")

    x_positions = list(range(len(df)))
    x_labels = df[label_col].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_positions, df[value_col])

    _finalize_and_save(
        fig=fig,
        ax=ax,
        output_path=output_path,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        dpi=dpi,
        show_legend=False,
        xtick_positions=x_positions,
        xtick_labels=x_labels,
    )


# ---------------------------------------------------------------------
# Multi-seed mean ± std-band plots
# ---------------------------------------------------------------------


def _draw_mean_std_bands(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_values: pd.Series,
    columns: list[str],
    labels: list[str],
    linewidth: float = 2.0,
    marker: str | None = None,
    marker_size: float = 5.0,
    shade_alpha: float = 0.18,
) -> bool:
    """
    Draw mean ± std metric curves on an existing axis.

    Expected metric columns:
        <metric>_mean
        <metric>_std
    """
    drew_any = False
    x = pd.to_numeric(x_values, errors="coerce")

    for col, label in zip(columns, labels):
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"

        if mean_col not in df.columns or not df[mean_col].notna().any():
            continue

        y_mean = pd.to_numeric(df[mean_col], errors="coerce")

        if std_col in df.columns:
            y_std = pd.to_numeric(df[std_col], errors="coerce").fillna(0.0)
        else:
            y_std = pd.Series(0.0, index=df.index)

        valid = x.notna() & y_mean.notna()

        if not valid.any():
            continue

        x_valid = x[valid].to_numpy(dtype=float)
        y_mean_valid = y_mean[valid].to_numpy(dtype=float)
        y_std_valid = y_std[valid].to_numpy(dtype=float)

        line, = ax.plot(
            x_valid,
            y_mean_valid,
            linewidth=linewidth,
            marker=marker,
            markersize=marker_size,
            label=label,
            zorder=3,
        )

        ax.fill_between(
            x_valid,
            y_mean_valid - y_std_valid,
            y_mean_valid + y_std_valid,
            color=line.get_color(),
            alpha=shade_alpha,
            linewidth=0,
            zorder=2,
        )

        drew_any = True

    return drew_any


def plot_metric_lines_with_std_band(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: str | Path,
    dpi: int = 300,
    y_min: float | None = None,
    y_max: float | None = None,
    linewidth: float = 2.0,
    shade_alpha: float = 0.18,
) -> None:
    """
    Plot several multi-seed metric curves in one figure.

    Expected columns:
        <metric>_mean
        <metric>_std

    The x-axis is numeric and preserves real epoch spacing.
    """
    if x_col not in df.columns:
        raise KeyError(f"DataFrame must contain {x_col!r} column.")

    plot_df = df.sort_values(x_col).copy()
    x_values = pd.to_numeric(plot_df[x_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 6))

    drew_any = _draw_mean_std_bands(
        ax=ax,
        df=plot_df,
        x_values=x_values,
        columns=y_cols,
        labels=labels,
        linewidth=linewidth,
        marker=None,
        shade_alpha=shade_alpha,
    )

    if not drew_any:
        plt.close(fig)
        print(f"[WARN] No available mean/std columns for plot: {title}")
        return

    _finalize_and_save(
        fig=fig,
        ax=ax,
        output_path=output_path,
        title=title,
        xlabel=x_col,
        ylabel=ylabel,
        dpi=dpi,
        y_min=y_min,
        y_max=y_max,
        show_legend=True,
    )


def plot_training_curves_multiseed(
    df: pd.DataFrame,
    metric_groups: dict[str, dict[str, Any]],
    output_dir: str | Path,
    x_col: str = "epoch",
    dpi: int = 300,
) -> None:
    """
    Plot grouped multi-seed training/evaluation curves using mean ± std bands.
    """
    output_dir = ensure_dir(output_dir)

    for group_name, group in metric_groups.items():
        plot_metric_lines_with_std_band(
            df=df,
            x_col=x_col,
            y_cols=group["columns"],
            labels=group["labels"],
            title=group["title"],
            ylabel=group["ylabel"],
            output_path=output_dir / group.get("output", f"{group_name}_multiseed.png"),
            dpi=dpi,
            y_min=group.get("y_min", 0.0),
            y_max=group.get("y_max", 1.05),
            linewidth=group.get("linewidth", 2.0),
            shade_alpha=group.get("shade_alpha", 0.18),
        )


# ---------------------------------------------------------------------
# Multi-seed individual-seed-lines + mean plots
# ---------------------------------------------------------------------


def _draw_seed_lines_with_mean(
    ax: plt.Axes,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    x_col: str,
    seed_col: str,
    columns: list[str],
    labels: list[str],
    seed_linewidth: float = 0.8,
    mean_linewidth: float = 2.8,
    seed_alpha: float = 0.25,
) -> bool:
    """
    Draw individual seed trajectories plus a bold mean line.
    """
    if x_col not in raw_df.columns:
        raise KeyError(f"raw_df must contain {x_col!r} column.")

    if seed_col not in raw_df.columns:
        raise KeyError(f"raw_df must contain {seed_col!r} column.")

    if x_col not in summary_df.columns:
        raise KeyError(f"summary_df must contain {x_col!r} column.")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    drew_any = False

    for idx, (col, label) in enumerate(zip(columns, labels)):
        mean_col = f"{col}_mean"

        if col not in raw_df.columns and mean_col not in summary_df.columns:
            continue

        color = color_cycle[idx % len(color_cycle)] if color_cycle else None

        if col in raw_df.columns:
            work = raw_df[[seed_col, x_col, col]].copy()
            work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
            work[col] = pd.to_numeric(work[col], errors="coerce")
            work = work.dropna(subset=[seed_col, x_col, col])

            for _, seed_df in work.groupby(seed_col):
                seed_df = seed_df.sort_values(x_col)

                ax.plot(
                    seed_df[x_col],
                    seed_df[col],
                    color=color,
                    linewidth=seed_linewidth,
                    alpha=seed_alpha,
                    label=None,
                    zorder=1,
                )

        if mean_col in summary_df.columns:
            mean_df = summary_df[[x_col, mean_col]].copy()
            mean_df[x_col] = pd.to_numeric(mean_df[x_col], errors="coerce")
            mean_df[mean_col] = pd.to_numeric(mean_df[mean_col], errors="coerce")
            mean_df = mean_df.dropna(subset=[x_col, mean_col])
            mean_df = mean_df.sort_values(x_col)

            if not mean_df.empty:
                ax.plot(
                    mean_df[x_col],
                    mean_df[mean_col],
                    color=color,
                    linewidth=mean_linewidth,
                    label=label,
                    zorder=3,
                )

                drew_any = True

    return drew_any


def plot_metric_lines_with_seed_lines(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    x_col: str,
    seed_col: str,
    y_cols: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: str | Path,
    dpi: int = 300,
    y_min: float | None = None,
    y_max: float | None = None,
    seed_linewidth: float = 0.8,
    mean_linewidth: float = 2.8,
    seed_alpha: float = 0.25,
) -> None:
    """
    Plot multi-seed metric curves.

    Visual meaning:
        thin transparent lines = individual seed trajectories
        bold line              = mean across seeds

    The x-axis is numeric and preserves real epoch spacing.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    drew_any = _draw_seed_lines_with_mean(
        ax=ax,
        raw_df=raw_df,
        summary_df=summary_df,
        x_col=x_col,
        seed_col=seed_col,
        columns=y_cols,
        labels=labels,
        seed_linewidth=seed_linewidth,
        mean_linewidth=mean_linewidth,
        seed_alpha=seed_alpha,
    )

    if not drew_any:
        plt.close(fig)
        print(f"[WARN] No available seed/mean columns for plot: {title}")
        return

    _finalize_and_save(
        fig=fig,
        ax=ax,
        output_path=output_path,
        title=title,
        xlabel=x_col,
        ylabel=ylabel,
        dpi=dpi,
        y_min=y_min,
        y_max=y_max,
        show_legend=True,
    )


def plot_training_curves_multiseed_with_seed_lines(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metric_groups: dict[str, dict[str, Any]],
    output_dir: str | Path,
    x_col: str = "epoch",
    seed_col: str = "seed",
    dpi: int = 300,
) -> None:
    """
    Plot grouped multi-seed training curves using seed lines plus mean lines.
    """
    output_dir = ensure_dir(output_dir)

    for group_name, group in metric_groups.items():
        plot_metric_lines_with_seed_lines(
            raw_df=raw_df,
            summary_df=summary_df,
            x_col=x_col,
            seed_col=seed_col,
            y_cols=group["columns"],
            labels=group["labels"],
            title=group["title"],
            ylabel=group["ylabel"],
            output_path=output_dir / group.get("output", f"{group_name}_multiseed.png"),
            dpi=dpi,
            y_min=group.get("y_min", 0.0),
            y_max=group.get("y_max", 1.05),
            seed_linewidth=group.get("seed_linewidth", 0.8),
            mean_linewidth=group.get("mean_linewidth", 2.8),
            seed_alpha=group.get("seed_alpha", 0.25),
        )


# ---------------------------------------------------------------------
# Selected-checkpoint multi-seed comparison plots
# ---------------------------------------------------------------------


def plot_checkpoint_comparison_with_std_band(
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: str | Path,
    sort_col: str = "epoch",
    label_col: str = "epoch",
    xlabel: str = "epoch",
    dpi: int = 300,
    y_min: float | None = 0.0,
    y_max: float | None = 1.05,
    linewidth: float = 2.8,
    marker_size: float = 5.0,
    shade_alpha: float = 0.18,
) -> None:
    """
    Plot selected-checkpoint comparison metrics as mean ± std across seeds.

    Expected input columns:
        <metric>_mean
        <metric>_std

    The x-axis is categorical/equally spaced, but the tick labels should
    usually be selected epoch numbers.
    """
    if label_col not in df.columns:
        raise KeyError(f"DataFrame must contain {label_col!r} column.")

    if sort_col in df.columns:
        plot_df = df.sort_values(sort_col).copy()
    else:
        plot_df = df.copy()

    x_positions = pd.Series(
        range(len(plot_df)),
        index=plot_df.index,
        dtype=float,
    )
    x_tick_positions = list(range(len(plot_df)))
    x_tick_labels = plot_df[label_col].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(12, 6))

    drew_any = _draw_mean_std_bands(
        ax=ax,
        df=plot_df,
        x_values=x_positions,
        columns=columns,
        labels=labels,
        linewidth=linewidth,
        marker="o",
        marker_size=marker_size,
        shade_alpha=shade_alpha,
    )

    if not drew_any:
        plt.close(fig)
        print(f"[WARN] No available mean/std columns for plot: {title}")
        return

    _finalize_and_save(
        fig=fig,
        ax=ax,
        output_path=output_path,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        dpi=dpi,
        y_min=y_min,
        y_max=y_max,
        show_legend=True,
        xtick_positions=x_tick_positions,
        xtick_labels=x_tick_labels,
    )


def plot_checkpoint_comparison_with_seed_lines(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: str | Path,
    group_col: str = "epoch",
    seed_col: str = "seed",
    label_col: str = "epoch",
    sort_col: str = "epoch",
    xlabel: str = "epoch",
    dpi: int = 300,
    y_min: float | None = 0.0,
    y_max: float | None = 1.05,
    seed_linewidth: float = 0.8,
    mean_linewidth: float = 2.8,
    seed_alpha: float = 0.25,
) -> None:
    """
    Plot selected-checkpoint comparison metrics with individual seed lines
    plus a bold mean line.

    Visual meaning:
        thin transparent lines = individual seed values
        bold line              = mean across seeds

    The x-axis is categorical/equally spaced, but the tick labels should
    usually be selected epoch numbers.
    """
    if group_col not in raw_df.columns:
        raise KeyError(f"raw_df must contain {group_col!r} column.")

    if seed_col not in raw_df.columns:
        raise KeyError(f"raw_df must contain {seed_col!r} column.")

    if group_col not in summary_df.columns:
        raise KeyError(f"summary_df must contain {group_col!r} column.")

    if label_col not in summary_df.columns:
        raise KeyError(f"summary_df must contain {label_col!r} column.")

    if sort_col in summary_df.columns:
        summary_plot = summary_df.sort_values(sort_col).copy()
    else:
        summary_plot = summary_df.copy()

    x_labels = summary_plot[label_col].astype(str).tolist()
    group_values = summary_plot[group_col].tolist()
    x_positions = list(range(len(group_values)))

    group_to_x = {
        group_value: x_position
        for group_value, x_position in zip(group_values, x_positions)
    }

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    fig, ax = plt.subplots(figsize=(12, 6))
    drew_any = False

    for idx, (col, label) in enumerate(zip(columns, labels)):
        mean_col = f"{col}_mean"

        if col not in raw_df.columns and mean_col not in summary_plot.columns:
            continue

        color = color_cycle[idx % len(color_cycle)] if color_cycle else None

        if col in raw_df.columns:
            work = raw_df[[seed_col, group_col, col]].copy()
            work[col] = pd.to_numeric(work[col], errors="coerce")
            work = work.dropna(subset=[seed_col, group_col, col])
            work = work[work[group_col].isin(group_to_x)]

            for _, seed_df in work.groupby(seed_col):
                seed_df = seed_df.copy()
                seed_df["_x"] = seed_df[group_col].map(group_to_x)
                seed_df = seed_df.dropna(subset=["_x"]).sort_values("_x")

                ax.plot(
                    seed_df["_x"],
                    seed_df[col],
                    color=color,
                    linewidth=seed_linewidth,
                    alpha=seed_alpha,
                    label=None,
                    zorder=1,
                )

        if mean_col in summary_plot.columns:
            mean_df = summary_plot[[group_col, mean_col]].copy()
            mean_df[mean_col] = pd.to_numeric(mean_df[mean_col], errors="coerce")
            mean_df["_x"] = mean_df[group_col].map(group_to_x)
            mean_df = mean_df.dropna(subset=["_x", mean_col]).sort_values("_x")

            if not mean_df.empty:
                ax.plot(
                    mean_df["_x"],
                    mean_df[mean_col],
                    color=color,
                    linewidth=mean_linewidth,
                    marker="o",
                    label=label,
                    zorder=3,
                )

                drew_any = True

    if not drew_any:
        plt.close(fig)
        print(f"[WARN] No available seed/mean columns for plot: {title}")
        return

    _finalize_and_save(
        fig=fig,
        ax=ax,
        output_path=output_path,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        dpi=dpi,
        y_min=y_min,
        y_max=y_max,
        show_legend=True,
        xtick_positions=x_positions,
        xtick_labels=x_labels,
    )


# ---------------------------------------------------------------------
# Backward-compatible wrappers
# ---------------------------------------------------------------------


def plot_grouped_metric_lines(
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Backward-compatible wrapper for plot_checkpoint_comparison_lines.
    """
    plot_checkpoint_comparison_lines(*args, **kwargs)


def plot_metric_groups_with_std_band(
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Backward-compatible wrapper for plot_training_curves_multiseed.
    """
    plot_training_curves_multiseed(*args, **kwargs)


def plot_grouped_metric_lines_with_std_band(
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Backward-compatible wrapper for plot_checkpoint_comparison_with_std_band.
    """
    plot_checkpoint_comparison_with_std_band(*args, **kwargs)