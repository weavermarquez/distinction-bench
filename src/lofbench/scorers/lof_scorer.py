"""Custom scorers for Laws of Form benchmark using inspect_ai."""

from __future__ import annotations

import json
import re
from typing import Literal

from inspect_ai.scorer import Score, Scorer, Target, accuracy, mean, scorer, stderr


def extract_single_answer(response: str) -> Literal["marked", "unmarked", "unknown"]:
    """
    Extract the final answer from LLM response for single form evaluation.

    Args:
        response: The model's response text

    Returns:
        "marked", "unmarked", or "unknown"
    """
    if not response:
        return "unknown"

    # Try XML tag pattern first
    pattern = r"<answer>\s*(marked|unmarked|\(\)|void|nothing)\s*</answer>"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        ans = match.group(1).lower()
        return "marked" if ans in ("marked", "()") else "unmarked"

    # Fallback: find last occurrence of "marked"/"()" vs "unmarked"/"void"
    response_lower = response.lower()
    mark_pos = response_lower.rfind("marked")
    unmark_pos = response_lower.rfind("unmarked")

    if mark_pos > unmark_pos:
        return "marked"
    elif unmark_pos > mark_pos:
        return "unmarked"
    return "unknown"


def extract_composite_answer(response: str, n: int) -> dict[str, list[str] | int]:
    """
    Extract composite answer from LLM response.

    Args:
        response: The model's response text
        n: Number of items expected

    Returns:
        Dictionary with:
            - "items": list of "marked"/"unmarked"/"unknown" for each item
            - "total_marked": int or -1 if unparseable
    """
    empty: dict[str, list[str] | int] = {"items": ["unknown"] * n, "total_marked": -1}
    if not response:
        return empty

    # Try to find JSON in the response
    # Look for ```json ... ``` or just { ... }
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{[^{}]*"E1"[^{}]*\})', response, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group(1))
            items: list[str] = []
            for i in range(1, n + 1):
                key = f"E{i}"
                val = data.get(key, "unknown")
                if isinstance(val, str):
                    val = val.lower()
                    if val in ("marked", "()"):
                        items.append("marked")
                    elif val in ("unmarked", "void", "nothing"):
                        items.append("unmarked")
                    else:
                        items.append("unknown")
                else:
                    items.append("unknown")
            total = data.get("total_marked", -1)
            if not isinstance(total, int) or total < 0 or total > n:
                total = -1
            return {"items": items, "total_marked": total}
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: try to extract total_marked from text
    match = re.search(r'total_marked["\s:]+(\d+)', response, re.IGNORECASE)
    if match:
        total = int(match.group(1))
        if 0 <= total <= n:
            return {"items": ["unknown"] * n, "total_marked": total}

    return empty


@scorer(metrics=[accuracy(), stderr()])
def lof_single_scorer() -> Scorer:
    """
    Scorer for single form evaluation tasks.

    Extracts the model's answer and compares it to the target.
    Returns "C" for correct, "I" for incorrect.
    """

    async def score(state, target: Target) -> Score:
        # Extract the model's answer
        response = state.output.completion
        answer = extract_single_answer(response)

        # Get expected answer from target
        expected = target.text.lower()

        # Determine correctness
        is_correct = answer == expected
        value = "C" if is_correct else "I"

        # Create explanation
        explanation = f"Expected: {expected}, Got: {answer}"

        return Score(value=value, answer=answer, explanation=explanation)

    return score


@scorer(metrics={"per_item_accuracy": [mean()], "all_correct": [mean()], "count_match": [mean()]})
def lof_composite_scorer() -> Scorer:
    """
    Scorer for composite form evaluation tasks.

    Computes three metrics:
    - per_item_accuracy: fraction of items correctly evaluated
    - all_correct: 1.0 if all items correct, 0.0 otherwise
    - count_match: 1.0 if total_marked count matches, 0.0 otherwise
    """

    async def score(state, target: Target) -> Score:
        # Extract the model's answer
        response = state.output.completion

        # Get targets from metadata
        targets = state.metadata.get("targets", [])
        expected_count = state.metadata.get("count", -1)
        n = len(targets)

        # Extract composite answer
        extracted = extract_composite_answer(response, n)
        items = extracted["items"]
        total_marked = extracted["total_marked"]

        # Compute per-item accuracy
        correct_items = sum(1 for i, item in enumerate(items) if item == targets[i])
        per_item_accuracy = correct_items / n if n > 0 else 0.0

        # Compute all_correct
        all_correct = 1.0 if correct_items == n else 0.0

        # Compute count_match
        count_match = 1.0 if total_marked == expected_count else 0.0

        # Create explanation
        explanation = (
            f"Items correct: {correct_items}/{n}, "
            f"All correct: {all_correct == 1.0}, "
            f"Count match: {count_match == 1.0} "
            f"(expected: {expected_count}, got: {total_marked})"
        )

        return Score(
            value={
                "per_item_accuracy": per_item_accuracy,
                "all_correct": all_correct,
                "count_match": count_match,
            },
            answer=str(extracted),
            explanation=explanation,
        )

    return score
