"""Dataset factory functions for inspect-ai integration."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from inspect_ai.dataset import MemoryDataset, Sample

from lofbench.core import generate_composite_test_cases, generate_test_cases

if TYPE_CHECKING:
    from inspect_ai.dataset import Dataset

    from lofbench.renderers import FormRenderer


def create_single_dataset(
    n: int = 100,
    seed: int = 2025,
    renderer: FormRenderer | None = None,
    render_seed: int | None = None,
) -> Dataset:
    """Create a dataset for single expression evaluation.

    Args:
        n: Number of test cases
        seed: Seed for test case generation
        renderer: Optional renderer for form transformation
        render_seed: Seed for reproducible rendering

    Returns:
        inspect-ai Dataset with Sample objects
    """
    # Import here to avoid circular imports
    from lofbench.renderers import CanonicalRenderer

    renderer = renderer or CanonicalRenderer()
    cases = generate_test_cases(n=n, seed=seed)

    samples = []
    rng = random.Random(render_seed) if render_seed is not None else None

    for case in cases:
        rendered = renderer.render(case["input"], rng)

        samples.append(
            Sample(
                id=case["id"],
                input="Evaluate the expression",  # Placeholder, template provides real prompt
                target=case["target"],  # "marked" or "unmarked"
                metadata={
                    "expression": rendered.rendered,  # For template substitution
                    "difficulty": case["difficulty"],
                    "depth": case["depth"],
                    "steps": case["steps"],
                    "original_form": case["input"],
                    "renderer": rendered.renderer_name,
                    "render_metadata": rendered.metadata,
                },
            )
        )

    return MemoryDataset(samples=samples, name=f"lof_single_{renderer.name}")


def create_composite_dataset(
    n_groups: int = 100,
    group_size: int = 8,
    seed: int = 2025,
    renderer: FormRenderer | None = None,
    render_seed: int | None = None,
) -> Dataset:
    """Create a dataset for composite (multi-expression) evaluation.

    For composite tasks, the input contains multiple expressions and
    the target is the count of marked expressions.

    Args:
        n_groups: Number of test groups
        group_size: Expressions per group
        seed: Seed for test case generation
        renderer: Optional renderer for form transformation
        render_seed: Seed for reproducible rendering

    Returns:
        inspect-ai Dataset with Sample objects
    """
    from lofbench.renderers import CanonicalRenderer

    renderer = renderer or CanonicalRenderer()
    cases = generate_composite_test_cases(
        n_groups=n_groups,
        group_size=group_size,
        seed=seed,
    )

    samples = []
    rng = random.Random(render_seed) if render_seed is not None else None

    for case in cases:
        # Render each expression
        rendered_exprs = [renderer.render(expr, rng) for expr in case["expressions"]]

        # Format as numbered list for the prompt
        formatted_input = "\n".join(f"E{i + 1}. {r.rendered}" for i, r in enumerate(rendered_exprs))

        samples.append(
            Sample(
                id=case["id"],
                input="Evaluate the expressions",  # Placeholder, template provides real prompt
                target=str(case["count"]),  # Count of marked expressions
                metadata={
                    "expressions": formatted_input,  # For template substitution
                    "difficulty": case["difficulty"],
                    "group_size": case["group_size"],
                    "original_expressions": case["expressions"],
                    "targets": case["targets"],  # Per-expression targets
                    "count": case["count"],
                    "renderer": renderer.name,
                    "rendered_expressions": [r.rendered for r in rendered_exprs],
                },
            )
        )

    return MemoryDataset(samples=samples, name=f"lof_composite_{renderer.name}")
