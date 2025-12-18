"""Analysis utilities for LoF benchmark results."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai.log import EvalLog, EvalLogInfo, list_eval_logs, read_eval_log


# =============================================================================
# Log Loading Utilities
# =============================================================================


def load_curated_logs(
    log_ids: list[str],
    log_dir: str = "./logs",
    header_only: bool = False,
) -> dict[str, EvalLog]:
    """Load specific eval logs by their task_id.

    Args:
        log_ids: List of task IDs (the ID portion of the filename, e.g., 'MSnrYp76447Lbe8fD64VZs')
        log_dir: Directory containing eval logs
        header_only: If True, only load metadata (faster for large logs).
                    Note: Cancelled runs will be reloaded with samples to compute metrics.

    Returns:
        Dict mapping task_id to EvalLog
    """
    log_paths = list(list_eval_logs(log_dir))
    id_to_path: dict[str, EvalLogInfo] = {}

    for path in log_paths:
        # Extract task_id from path - it's the last part before .eval
        task_id = getattr(path, 'task_id', None)
        if task_id:
            id_to_path[task_id] = path

    result = {}
    for log_id in log_ids:
        if log_id in id_to_path:
            try:
                log = read_eval_log(id_to_path[log_id], header_only=header_only)
                # For cancelled runs without results, reload with samples to compute metrics
                if header_only and log.results is None:
                    log = read_eval_log(id_to_path[log_id], header_only=False)
                result[log_id] = log
            except Exception as e:
                print(f"Warning: Could not read log {log_id}: {e}")
        else:
            print(f"Warning: Log ID {log_id} not found")

    return result


def get_log_metadata(log: EvalLog) -> dict[str, Any]:
    """Extract key metadata from an eval log.

    Returns dict with: model, renderer, mismatched, dialect, thinking_tokens,
                       reasoning_effort, n_samples, epochs
    """
    task_args = log.eval.task_args or {}
    renderer = task_args.get('renderer', 'parens')
    config = task_args.get('renderer_config') or {}
    mismatched = config.get('mismatched', False) if config else False

    gen_config = log.eval.model_generate_config
    thinking = gen_config.reasoning_tokens if gen_config else 0
    if thinking is None:
        thinking = 0  # None means reasoning disabled

    # Extract reasoning_effort (used by Gemini 3 Flash)
    # For gemini-3-flash: minimal/low -> 'low', medium -> 'high'
    raw_effort = getattr(gen_config, 'reasoning_effort', None) if gen_config else None
    reasoning_effort = None
    if 'gemini-3-flash' in log.eval.model:
        if raw_effort in ('minimal', 'low'):
            reasoning_effort = 'low'
        elif raw_effort == 'medium':
            reasoning_effort = 'high'
        else:
            reasoning_effort = raw_effort

    # Determine dialect name
    if renderer == 'parens':
        dialect = 'canonical'
    elif renderer == 'noisy_parens' and not mismatched:
        dialect = 'noisy-balanced'
    elif renderer == 'noisy_parens' and mismatched:
        dialect = 'noisy-mismatch'
    else:
        dialect = renderer

    return {
        'model': log.eval.model,
        'renderer': renderer,
        'mismatched': mismatched,
        'dialect': dialect,
        'thinking_tokens': thinking,
        'reasoning_effort': reasoning_effort,
        'n_samples': log.eval.dataset.samples if log.eval.dataset else None,
        'epochs': log.eval.config.epochs if log.eval.config else 1,
    }


def get_log_results(log: EvalLog) -> dict[str, float | None]:
    """Extract headline metrics from an eval log.

    Falls back to computing from samples if log.results is None (cancelled runs).

    Returns dict with: per_item_accuracy, all_correct, structure_accuracy
    """
    metrics_out = {'per_item_accuracy': None, 'all_correct': None, 'structure_accuracy': None}

    # Try to get from results first
    results = log.results
    if results and results.scores:
        for score in results.scores:
            val = score.metrics.get('mean')
            if val:
                if score.name == 'per_item_accuracy':
                    metrics_out['per_item_accuracy'] = val.value
                elif score.name == 'all_correct':
                    metrics_out['all_correct'] = val.value
                elif score.name == 'structure_accuracy':
                    metrics_out['structure_accuracy'] = val.value
        return metrics_out

    # Fall back to computing from samples (for cancelled runs)
    if log.samples:
        per_item_vals = []
        all_correct_vals = []
        structure_vals = []

        for sample in log.samples:
            if sample.scores:
                score = list(sample.scores.values())[0]
                if isinstance(score.value, dict):
                    per_item_vals.append(score.value.get('per_item_accuracy', 0))
                    all_correct_vals.append(score.value.get('all_correct', 0))
                    struct = score.value.get('structure_accuracy')
                    if struct is not None:
                        structure_vals.append(struct)

        if per_item_vals:
            metrics_out['per_item_accuracy'] = sum(per_item_vals) / len(per_item_vals)
        if all_correct_vals:
            metrics_out['all_correct'] = sum(all_correct_vals) / len(all_correct_vals)
        if structure_vals:
            metrics_out['structure_accuracy'] = sum(structure_vals) / len(structure_vals)

    return metrics_out


# =============================================================================
# DataFrame Utilities
# =============================================================================


def parse_score_column(df: pd.DataFrame, score_col: str = 'score_lof_composite_scorer') -> pd.DataFrame:
    """Extract dict score values into separate columns.

    Args:
        df: DataFrame with a dict-valued or JSON string score column
        score_col: Name of the score column containing dicts or JSON strings

    Returns:
        DataFrame with new columns: per_item_accuracy, all_correct, structure_accuracy
    """
    import json

    df = df.copy()

    if score_col not in df.columns:
        return df

    def _parse_score(x):
        """Parse score value, handling both dict and JSON string."""
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    df['per_item_accuracy'] = df[score_col].apply(
        lambda x: _parse_score(x).get('per_item_accuracy', 0)
    )
    df['all_correct'] = df[score_col].apply(
        lambda x: _parse_score(x).get('all_correct', 0)
    )
    df['structure_accuracy'] = df[score_col].apply(
        lambda x: _parse_score(x).get('structure_accuracy')
    )

    return df


def normalize_epochs(df: pd.DataFrame, k: int = 2, seed: int = 42) -> pd.DataFrame:
    """Subsample k epochs per sample for fair cross-run comparison.

    Args:
        df: DataFrame with 'id' and 'epoch' columns
        k: Number of epochs to sample per sample
        seed: Random seed for reproducibility

    Returns:
        DataFrame with at most k rows per sample ID
    """
    rng = random.Random(seed)

    rows_to_keep = []
    for sample_id, group in df.groupby('id'):
        epochs = group['epoch'].unique().tolist()
        if len(epochs) <= k:
            rows_to_keep.extend(group.index.tolist())
        else:
            selected_epochs = rng.sample(epochs, k)
            mask = group['epoch'].isin(selected_epochs)
            rows_to_keep.extend(group[mask].index.tolist())

    return df.loc[rows_to_keep].copy()


# =============================================================================
# Epoch-Aware Metrics
# =============================================================================


def compute_k1_accuracy(df: pd.DataFrame, metric_col: str = 'per_item_accuracy') -> float:
    """Compute mean accuracy across all epochs (K=1 single-shot equivalent).

    This gives the expected accuracy if you only run each sample once.
    """
    return df[metric_col].mean()


def compute_pass_at_k(df: pd.DataFrame, k: int, metric_col: str = 'all_correct') -> float:
    """Compute pass@k: probability that at least 1 of k epochs is correct.

    Args:
        df: DataFrame with 'id', 'epoch', and metric columns
        k: Number of attempts
        metric_col: Column containing correctness (0/1)

    Returns:
        Fraction of samples with at least one correct epoch
    """
    # Normalize to k epochs first
    df_k = normalize_epochs(df, k=k)

    # Group by sample and check if any epoch was correct
    passed = df_k.groupby('id')[metric_col].apply(lambda x: x.max() > 0)
    return passed.mean()


def compute_all_at_k(df: pd.DataFrame, k: int, metric_col: str = 'all_correct') -> float:
    """Compute all@k: probability that all k epochs are correct.

    Args:
        df: DataFrame with 'id', 'epoch', and metric columns
        k: Number of attempts
        metric_col: Column containing correctness (0/1)

    Returns:
        Fraction of samples where all epochs were correct
    """
    # Normalize to k epochs first
    df_k = normalize_epochs(df, k=k)

    # Group by sample and check if all epochs were correct
    all_passed = df_k.groupby('id')[metric_col].apply(lambda x: x.min() > 0)
    return all_passed.mean()


def compute_by_dimension(
    df: pd.DataFrame,
    dim: str,
    metric_col: str = 'per_item_accuracy'
) -> pd.DataFrame:
    """Compute metrics grouped by a dimension (difficulty, target, etc).

    Args:
        df: DataFrame with the dimension column and metric columns
        dim: Column name to group by (e.g., 'metadata_difficulty', 'target')
        metric_col: Metric to aggregate

    Returns:
        DataFrame with dim, mean, std, count columns
    """
    grouped = df.groupby(dim)[metric_col].agg(['mean', 'std', 'count'])
    grouped = grouped.reset_index()
    grouped.columns = [dim, 'mean', 'std', 'count']
    return grouped


# =============================================================================
# Original Functions (kept for backwards compatibility)
# =============================================================================


def load_logs(log_dir: str = "./logs", pattern: str = "*") -> list[EvalLog]:
    """Load all eval logs from a directory."""
    log_paths = list_eval_logs(log_dir)
    logs = []
    for path in log_paths:
        if pattern == "*" or pattern in str(path):
            try:
                logs.append(read_eval_log(path))
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")
    return logs


def extract_composite_results(log: EvalLog) -> list[dict[str, Any]]:
    """Extract per-sample results from a composite task log."""
    results = []
    for sample in log.samples or []:
        metadata = sample.metadata or {}

        # Get score values (scores is a dict, not a list)
        score_dict = {}
        if sample.scores:
            # Get first scorer's value from dict
            score = list(sample.scores.values())[0]
            if isinstance(score.value, dict):
                score_dict = score.value

        results.append({
            "id": sample.id,
            "model": log.eval.model,
            "difficulty": metadata.get("difficulty", "unknown"),
            "group_size": metadata.get("group_size", 8),
            "targets": metadata.get("targets", []),
            "count": metadata.get("count", -1),
            "per_item_accuracy": score_dict.get("per_item_accuracy", 0),
            "all_correct": score_dict.get("all_correct", 0),
            "count_match": score_dict.get("count_match", 0),
            "predicted_count": _extract_predicted_count(sample),
        })
    return results


def _extract_predicted_count(sample) -> int:
    """Extract predicted count from sample scores."""
    if sample.scores:
        # Get first scorer from dict
        score = list(sample.scores.values())[0]
        if score.answer:
            try:
                # Parse the answer string to get total_marked
                import ast
                answer = ast.literal_eval(score.answer)
                if isinstance(answer, dict):
                    return answer.get("total_marked", -1)
            except (ValueError, SyntaxError):
                pass
    return -1


def analyze_composite(logs: list[EvalLog]) -> str:
    """Generate composite task analysis report."""
    lines = []

    # Collect all results
    all_results = []
    for log in logs:
        if "composite" in (log.eval.task or ""):
            all_results.extend(extract_composite_results(log))

    if not all_results:
        return "No composite task results found."

    # === Target Count Distribution ===
    lines.append("=== Target Count Distribution ===")
    count_dist = Counter(r["count"] for r in all_results)
    max_count = max(count_dist.values()) if count_dist else 1
    for count in sorted(count_dist.keys()):
        n = count_dist[count]
        bar = "█" * int(40 * n / max_count)
        lines.append(f"  {count}: {bar} {n}")
    lines.append("")

    # === Random Baselines ===
    group_size = all_results[0]["group_size"] if all_results else 8
    lines.append("=== Random Baselines ===")
    lines.append(f"  Per-item:       50.0% (coin flip)")
    lines.append(f"  All-correct:    {100 / (2**group_size):.2f}% ({group_size} coin flips)")

    # Calculate count-match baseline (sum of P(k)^2 for binomial)
    from math import comb
    p_match = sum((comb(group_size, k) / (2**group_size))**2 for k in range(group_size + 1))
    lines.append(f"  Count-match:    {100 * p_match:.1f}%")

    # Calculate expected MAE for random guessing
    mae = _calculate_random_mae(group_size)
    lines.append(f"  Count MAE:      {mae:.2f}")
    lines.append("")

    # === Results by Model ===
    lines.append("=== Results by Model ===")
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    for model, results in sorted(by_model.items()):
        n = len(results)
        per_item = 100 * sum(r["per_item_accuracy"] for r in results) / n
        all_correct = sum(r["all_correct"] for r in results)
        all_correct_pct = 100 * all_correct / n
        count_match = 100 * sum(r["count_match"] for r in results) / n

        # Calculate MAE
        mae_sum = sum(abs(r["predicted_count"] - r["count"])
                      for r in results if r["predicted_count"] >= 0)
        mae_n = sum(1 for r in results if r["predicted_count"] >= 0)
        mae = mae_sum / mae_n if mae_n > 0 else float("nan")

        lines.append(f"{model} (n={n}):")
        lines.append(f"  Per-item:      {per_item:.1f}%")
        lines.append(f"  All-correct:   {all_correct_pct:.1f}% ({all_correct}/{n})")
        lines.append(f"  Count match:   {count_match:.1f}%")
        lines.append(f"  Count MAE:     {mae:.2f}")
    lines.append("")

    # === Accuracy by Difficulty ===
    lines.append("=== Accuracy by Difficulty ===")
    for model, results in sorted(by_model.items()):
        lines.append(f"{model}:")
        by_diff = defaultdict(list)
        for r in results:
            by_diff[r["difficulty"]].append(r)

        for diff in sorted(by_diff.keys()):
            diff_results = by_diff[diff]
            n = len(diff_results)
            per_item = 100 * sum(r["per_item_accuracy"] for r in diff_results) / n
            all_correct = sum(r["all_correct"] for r in diff_results)
            all_correct_pct = 100 * all_correct / n
            lines.append(f"  {diff:12} {per_item:5.1f}% per-item   {all_correct_pct:5.1f}% all-correct  (n={n})")
    lines.append("")

    # === Accuracy by Target ===
    lines.append("=== Accuracy by Target ===")
    for model, results in sorted(by_model.items()):
        lines.append(f"{model}:")
        marked_correct = 0
        marked_total = 0
        unmarked_correct = 0
        unmarked_total = 0

        for r in results:
            targets = r.get("targets", [])
            # We'd need per-item predictions to calculate this properly
            # For now, estimate from per_item_accuracy and target distribution
            n_marked = sum(1 for t in targets if t == "marked")
            n_unmarked = len(targets) - n_marked
            marked_total += n_marked
            unmarked_total += n_unmarked
            # Approximate (would need actual per-item predictions for exact)
            marked_correct += int(n_marked * r["per_item_accuracy"])
            unmarked_correct += int(n_unmarked * r["per_item_accuracy"])

        if marked_total > 0:
            lines.append(f"  marked      : {100*marked_correct/marked_total:.1f}% ({marked_correct}/{marked_total})")
        if unmarked_total > 0:
            lines.append(f"  unmarked    : {100*unmarked_correct/unmarked_total:.1f}% ({unmarked_correct}/{unmarked_total})")

    return "\n".join(lines)


def _calculate_random_mae(n: int) -> float:
    """Calculate expected MAE for random guessing on binomial distribution."""
    from math import comb

    total = 0.0
    for true_k in range(n + 1):
        p_true = comb(n, true_k) / (2**n)
        for guess_k in range(n + 1):
            p_guess = comb(n, guess_k) / (2**n)
            total += p_true * p_guess * abs(true_k - guess_k)
    return total


def analyze_single(logs: list[EvalLog]) -> str:
    """Generate single task analysis report."""
    lines = []

    all_results = []
    for log in logs:
        if "single" in (log.eval.task or "") and "composite" not in (log.eval.task or ""):
            for sample in log.samples or []:
                metadata = sample.metadata or {}
                # Get score from dict (not list)
                correct = False
                if sample.scores:
                    score = list(sample.scores.values())[0]
                    correct = score.value == "C"
                all_results.append({
                    "model": log.eval.model,
                    "difficulty": metadata.get("difficulty", "unknown"),
                    "target": sample.target,
                    "correct": correct,
                    "steps": metadata.get("steps", 0),
                })

    if not all_results:
        return "No single task results found."

    # === Average Steps per Difficulty ===
    lines.append("=== Average Steps per Difficulty ===")
    by_diff = defaultdict(list)
    for r in all_results:
        by_diff[r["difficulty"]].append(r["steps"])
    for diff in sorted(by_diff.keys()):
        avg = sum(by_diff[diff]) / len(by_diff[diff])
        lines.append(f"  {diff:12}: ~{avg:.1f} steps")
    lines.append("")

    # === Results by Model ===
    lines.append("=== Results by Model ===")
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    for model, results in sorted(by_model.items()):
        n = len(results)
        accuracy = 100 * sum(r["correct"] for r in results) / n
        lines.append(f"{model} (n={n}): {accuracy:.1f}%")
    lines.append("")

    # === Accuracy by Difficulty ===
    lines.append("=== Accuracy by Difficulty ===")
    for model, results in sorted(by_model.items()):
        lines.append(f"{model}:")
        by_diff = defaultdict(list)
        for r in results:
            by_diff[r["difficulty"]].append(r["correct"])
        for diff in sorted(by_diff.keys()):
            n = len(by_diff[diff])
            acc = 100 * sum(by_diff[diff]) / n
            lines.append(f"  {diff:12}: {acc:5.1f}%  (n={n})")
    lines.append("")

    # === Accuracy by Target ===
    lines.append("=== Accuracy by Target ===")
    for model, results in sorted(by_model.items()):
        lines.append(f"{model}:")
        by_target = defaultdict(list)
        for r in results:
            by_target[r["target"]].append(r["correct"])
        for target in sorted(by_target.keys()):
            n = len(by_target[target])
            correct = sum(by_target[target])
            lines.append(f"  {target:10}: {100*correct/n:.1f}% ({correct}/{n})")

    return "\n".join(lines)


def print_report(log_dir: str = "./logs") -> None:
    """Print full analysis report."""
    logs = load_logs(log_dir)
    print(f"Loaded {len(logs)} logs from {log_dir}\n")

    single_report = analyze_single(logs)
    if "No single" not in single_report:
        print("=" * 60)
        print("SINGLE TASK ANALYSIS")
        print("=" * 60)
        print(single_report)
        print()

    composite_report = analyze_composite(logs)
    if "No composite" not in composite_report:
        print("=" * 60)
        print("COMPOSITE TASK ANALYSIS")
        print("=" * 60)
        print(composite_report)


if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "./logs"
    print_report(log_dir)
