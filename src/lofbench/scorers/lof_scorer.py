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

    # Fallback: find last occurrence of "marked" vs "unmarked"
    # Use regex to find whole words only (avoid "marked" matching inside "unmarked")
    response_lower = response.lower()

    mark_matches = list(re.finditer(r"\bmarked\b", response_lower))
    unmark_matches = list(re.finditer(r"\bunmarked\b", response_lower))

    mark_pos = mark_matches[-1].start() if mark_matches else -1
    unmark_pos = unmark_matches[-1].start() if unmark_matches else -1

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
    """
    empty: dict[str, list[str] | int] = {"items": ["unknown"] * n}
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
            return {"items": items}
        except (json.JSONDecodeError, TypeError):
            pass

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


@scorer(metrics={"all_correct": [mean()], "per_item_accuracy": [mean()]})
def lof_composite_scorer() -> Scorer:
    """
    Scorer for composite form evaluation tasks.

    Computes two metrics:
    - all_correct: 1.0 if all items correct, 0.0 otherwise (primary)
    - per_item_accuracy: fraction of items correctly evaluated (diagnostic)

    Designed for use with epochs=Epochs(3, "at_least_2") at task level.
    Human baseline expectation: 98-100% all_correct.
    """

    async def score(state, target: Target) -> Score:
        # Extract the model's answer
        response = state.output.completion

        # Get targets from metadata
        targets = state.metadata.get("targets", [])
        n = len(targets)

        # Extract composite answer
        extracted = extract_composite_answer(response, n)
        items = extracted["items"]

        # Compute per-item accuracy
        correct_items = sum(1 for i, item in enumerate(items) if item == targets[i])
        per_item_accuracy = correct_items / n if n > 0 else 0.0

        # Compute all_correct
        all_correct = 1.0 if correct_items == n else 0.0

        # Create explanation
        explanation = f"Items correct: {correct_items}/{n}, All correct: {all_correct == 1.0}"

        return Score(
            value={
                "all_correct": all_correct,
                "per_item_accuracy": per_item_accuracy,
            },
            answer=str(extracted),
            explanation=explanation,
        )

    return score
