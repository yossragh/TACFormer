from __future__ import annotations

import math
import re
from numbers import Integral, Number
from pathlib import Path
from typing import Any

import pandas as pd

from utils import read_json, read_jsonl


# ---------------------------------------------------------------------
# Generic metric-table helpers
# ---------------------------------------------------------------------


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to numeric only when all originally non-null values
    can be converted.

    This avoids partially converting mixed columns such as labels or paths,
    while still converting numeric columns that contain missing values.
    """
    df = df.copy()

    for col in df.columns:
        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")

        original_non_null = original.notna()
        converted_non_null = converted.notna()

        if converted_non_null[original_non_null].all():
            df[col] = converted

    return df


def extract_scalar_numeric_metrics(metrics: dict[str, Any]) -> dict[str, int | float]:
    """
    Extract scalar numeric metrics from a metrics dictionary.

    Kept:
        int
        float
        numpy numeric scalars

    Ignored:
        bool
        strings
        lists
        dictionaries
        non-finite floats
    """
    out: dict[str, int | float] = {}

    for key, value in metrics.items():
        if isinstance(value, bool):
            continue

        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass

        if isinstance(value, Number):
            if isinstance(value, Integral):
                out[key] = int(value)
            else:
                value = float(value)

                if math.isfinite(value):
                    out[key] = value

    return out


def build_mean_std_summary(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_col: str = "epoch",
    add_epoch_label: bool = True,
) -> pd.DataFrame:
    """
    Aggregate metric columns by a grouping column.

    For every metric, produces:
        <metric>_mean
        <metric>_std
        <metric>_n

    Typical use:
        group_col="epoch" for multi-seed experiments.
    """
    if df.empty:
        raise ValueError("df is empty; cannot build mean/std summary.")

    if group_col not in df.columns:
        raise KeyError(f"df must contain {group_col!r} column.")

    work = df.copy()
    available_metrics: list[str] = []

    for col in metric_cols:
        if col not in work.columns:
            continue

        converted = pd.to_numeric(work[col], errors="coerce")

        if converted.notna().any():
            work[col] = converted
            available_metrics.append(col)

    if not available_metrics:
        raise ValueError("None of the requested metric columns are available.")

    grouped = work.groupby(group_col)[available_metrics].agg(["mean", "std", "count"])

    grouped.columns = [
        f"{metric}_{stat if stat != 'count' else 'n'}"
        for metric, stat in grouped.columns
    ]

    summary = grouped.reset_index()

    if add_epoch_label and group_col == "epoch":
        summary[group_col] = pd.to_numeric(
            summary[group_col],
            errors="coerce",
        ).astype(int)
        summary["label"] = summary[group_col].apply(lambda x: f"epoch_{int(x)}")

    return summary.sort_values(group_col).reset_index(drop=True)


def collect_metric_columns_from_groups(metric_groups: dict[str, dict[str, Any]]) -> list[str]:
    """
    Collect unique metric column names from grouped metric plot specs.

    Expected format:
        {
            "generation_quality": {
                "columns": ["validity", "uniqueness_among_valid"],
                ...
            },
            ...
        }
    """
    metric_cols: list[str] = []

    for group in metric_groups.values():
        for col in group["columns"]:
            if col not in metric_cols:
                metric_cols.append(col)

    return metric_cols


# ---------------------------------------------------------------------
# Hugging Face Trainer log metrics
# ---------------------------------------------------------------------


def checkpoint_step(path: str | Path) -> int:
    """
    Extract the numeric step from a path containing checkpoint-<step>.

    Example:
        checkpoint-3052 -> 3052

    Returns -1 if no checkpoint step is found.
    """
    path = Path(path)
    match = re.search(r"checkpoint-(\d+)", str(path))

    if match is None:
        return -1

    return int(match.group(1))


def find_trainer_state_json(run_dir: str | Path) -> Path | None:
    """
    Find trainer_state.json for a Hugging Face Trainer run.

    Search order:
        1. <run_dir>/trainer_state.json
        2. latest <run_dir>/checkpoint-*/trainer_state.json

    Returns None if no trainer_state.json is found.
    """
    run_dir = Path(run_dir)

    direct_path = run_dir / "trainer_state.json"
    if direct_path.exists():
        return direct_path

    candidates = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=checkpoint_step,
    )

    if not candidates:
        return None

    return candidates[-1]


def load_trainer_log_history(trainer_state_json: str | Path | None) -> pd.DataFrame:
    """
    Load Hugging Face Trainer log_history from trainer_state.json.

    Returns an empty DataFrame if trainer_state_json is None, missing,
    or contains no log_history.
    """
    if trainer_state_json is None:
        return pd.DataFrame()

    trainer_state_json = Path(trainer_state_json)

    if not trainer_state_json.exists():
        return pd.DataFrame()

    state = read_json(trainer_state_json)

    if not isinstance(state, dict):
        return pd.DataFrame()

    log_history = state.get("log_history", [])

    if not log_history:
        return pd.DataFrame()

    df = pd.DataFrame(log_history)
    df = _coerce_numeric_columns(df)

    return df


def get_eval_loss_by_epoch(trainer_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Extract eval_loss per epoch from Hugging Face Trainer log history.

    Returns columns:
        epoch
        eval_loss
    """
    if trainer_logs.empty:
        return pd.DataFrame(columns=["epoch", "eval_loss"])

    if "eval_loss" not in trainer_logs.columns or "epoch" not in trainer_logs.columns:
        return pd.DataFrame(columns=["epoch", "eval_loss"])

    df = trainer_logs.dropna(subset=["eval_loss", "epoch"]).copy()

    if df.empty:
        return pd.DataFrame(columns=["epoch", "eval_loss"])

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["eval_loss"] = pd.to_numeric(df["eval_loss"], errors="coerce")

    df = df.dropna(subset=["epoch", "eval_loss"])

    if df.empty:
        return pd.DataFrame(columns=["epoch", "eval_loss"])

    df["epoch"] = df["epoch"].round().astype(int)
    df = df.sort_values("epoch").drop_duplicates("epoch", keep="last")

    return df[["epoch", "eval_loss"]].reset_index(drop=True)


def get_train_loss(trainer_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Extract training loss points from Hugging Face Trainer log history.

    Returns columns:
        epoch
        loss
    """
    if trainer_logs.empty:
        return pd.DataFrame(columns=["epoch", "loss"])

    if "loss" not in trainer_logs.columns or "epoch" not in trainer_logs.columns:
        return pd.DataFrame(columns=["epoch", "loss"])

    df = trainer_logs.dropna(subset=["loss", "epoch"]).copy()

    if df.empty:
        return pd.DataFrame(columns=["epoch", "loss"])

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")

    df = df.dropna(subset=["epoch", "loss"])
    df = df.sort_values("epoch")

    return df[["epoch", "loss"]].reset_index(drop=True)


# ---------------------------------------------------------------------
# Generation-monitor metrics
# ---------------------------------------------------------------------


def load_generation_monitor_metrics(path: str | Path) -> pd.DataFrame:
    """
    Load local generation-monitor metrics.

    Expected file:
        <run_dir>/generation_monitor/metrics.jsonl

    Each row should contain:
        epoch
        scalar generation metrics
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Generation monitor metrics file not found: {path}\n"
            "Expected file: <run_dir>/generation_monitor/metrics.jsonl"
        )

    rows: list[dict[str, Any]] = read_jsonl(path)

    if not rows:
        raise ValueError(f"Generation monitor metrics file is empty: {path}")

    df = pd.DataFrame(rows)

    if "epoch" not in df.columns:
        raise KeyError("Generation monitor metrics must contain an 'epoch' column.")

    df = _coerce_numeric_columns(df)

    epoch = pd.to_numeric(df["epoch"], errors="coerce")

    if epoch.isna().any():
        raise ValueError(f"Generation monitor metrics contain invalid epoch values: {path}")

    df["epoch"] = epoch.astype(int)

    df = df.drop_duplicates(subset="epoch", keep="last")
    df = df.sort_values("epoch").reset_index(drop=True)

    return df


def build_training_summary(
    generation_metrics: pd.DataFrame,
    eval_loss_by_epoch: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a training summary table.

    Starts from generation-monitor metrics and optionally merges eval_loss.
    """
    if generation_metrics.empty:
        raise ValueError("generation_metrics is empty; cannot build summary.")

    summary = generation_metrics.copy()

    if eval_loss_by_epoch is not None and not eval_loss_by_epoch.empty:
        summary = summary.merge(eval_loss_by_epoch, on="epoch", how="left")

    return summary.sort_values("epoch").reset_index(drop=True)


def infer_seed_from_run_dir(run_dir: str | Path) -> int | None:
    """
    Infer seed from a directory name such as:
        seed_11
        seed-11

    Returns None if no seed can be inferred.
    """
    run_dir = Path(run_dir)
    match = re.search(r"seed[_-](\d+)", run_dir.name)

    if match is None:
        return None

    return int(match.group(1))


def load_multiseed_generation_monitor_metrics(
    run_root: str | Path,
    generation_metrics_jsonl: str = "generation_monitor/metrics.jsonl",
    seed_glob: str = "seed_*",
) -> pd.DataFrame:
    """
    Load generation-monitor metrics from several seed folders.

    Expected structure:
        <run_root>/seed_11/generation_monitor/metrics.jsonl
        <run_root>/seed_22/generation_monitor/metrics.jsonl
        <run_root>/seed_33/generation_monitor/metrics.jsonl

    Adds:
        seed
        run_dir
        metrics_path
    """
    run_root = Path(run_root)

    if not run_root.exists():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    seed_dirs = sorted(
        path
        for path in run_root.glob(seed_glob)
        if path.is_dir()
    )

    if not seed_dirs:
        raise FileNotFoundError(
            f"No seed directories found under {run_root} using pattern {seed_glob!r}"
        )

    all_dfs: list[pd.DataFrame] = []

    for seed_dir in seed_dirs:
        metrics_path = seed_dir / generation_metrics_jsonl

        if not metrics_path.exists():
            print(f"[WARN] Missing generation monitor metrics: {metrics_path}")
            continue

        seed = infer_seed_from_run_dir(seed_dir)

        if seed is None:
            raise ValueError(f"Could not infer seed from directory name: {seed_dir.name}")

        df = load_generation_monitor_metrics(metrics_path)

        df["seed"] = seed
        df["run_dir"] = str(seed_dir)
        df["metrics_path"] = str(metrics_path)

        all_dfs.append(df)

        print(f"[INFO] Loaded seed={seed} metrics from {metrics_path}")

    if not all_dfs:
        raise ValueError(f"No generation monitor metrics could be loaded from {run_root}")

    out = pd.concat(all_dfs, ignore_index=True)

    return out.sort_values(["epoch", "seed"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Selected-checkpoint comparison metrics
# ---------------------------------------------------------------------


def load_checkpoint_comparison(path: str | Path) -> pd.DataFrame:
    """
    Load checkpoint_comparison.csv produced by selected-checkpoint evaluation.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint comparison CSV not found: {path}")

    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise KeyError("checkpoint_comparison.csv must contain a 'label' column.")

    df = _coerce_numeric_columns(df)

    return df


def add_selection_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple checkpoint selection score when possible.

    Score:
        validity × uniqueness_among_valid × novelty_among_unique_valid

    This is only a quick ranking helper. It should not replace scientific judgment.
    """
    df = df.copy()

    needed = [
        "validity",
        "uniqueness_among_valid",
        "novelty_among_unique_valid",
    ]

    if all(col in df.columns for col in needed):
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["selection_score"] = (
            df["validity"]
            * df["uniqueness_among_valid"]
            * df["novelty_among_unique_valid"]
        )

    return df


def infer_epoch_from_label(label: str) -> int | None:
    """
    Infer epoch number from labels such as:
        epoch_10
        epoch_10_seed_11
        seed_11_epoch_10
    """
    match = re.search(r"epoch[_-](\d+)", str(label))

    if match is None:
        return None

    return int(match.group(1))


def infer_seed_from_label(label: str) -> int | None:
    """
    Infer seed number from labels such as:
        seed_11
        epoch_10_seed_11
        seed_11_epoch_10
    """
    match = re.search(r"seed[_-](\d+)", str(label))

    if match is None:
        return None

    return int(match.group(1))


def add_epoch_seed_from_checkpoint_label(
    df: pd.DataFrame,
    label_col: str = "label",
    require_seed: bool = False,
) -> pd.DataFrame:
    """
    Add epoch, seed, and epoch_label columns from checkpoint labels.

    Expected multi-seed labels:
        epoch_10_seed_11
        epoch_10_seed_22
        epoch_14_seed_11

    If require_seed=True, labels without a seed raise an error.
    """
    if label_col not in df.columns:
        raise KeyError(f"DataFrame must contain {label_col!r} column.")

    df = df.copy()

    df["epoch"] = df[label_col].apply(infer_epoch_from_label)
    df["seed"] = df[label_col].apply(infer_seed_from_label)

    if df["epoch"].isna().any():
        bad = df[df["epoch"].isna()][label_col].tolist()
        raise ValueError(
            "Could not infer epoch from some checkpoint labels: "
            + ", ".join(map(str, bad))
        )

    if require_seed and df["seed"].isna().any():
        bad = df[df["seed"].isna()][label_col].tolist()
        raise ValueError(
            "Could not infer seed from some checkpoint labels: "
            + ", ".join(map(str, bad))
        )

    df["epoch"] = df["epoch"].astype(int)
    df["epoch_label"] = df["epoch"].apply(lambda x: f"epoch_{int(x)}")

    return df