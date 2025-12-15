"""Analysis utilities for LoF benchmark results."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log


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

        # Get score values
        score_dict = {}
        if sample.scores:
            score = sample.scores[0]
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
    if sample.scores and sample.scores[0].answer:
        try:
            # Parse the answer string to get total_marked
            import ast
            answer = ast.literal_eval(sample.scores[0].answer)
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
                correct = sample.scores[0].value == "C" if sample.scores else False
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
