"""Composite (multi-expression) LoF evaluation task."""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_ai.solver import generate, prompt_template, system_message

from lofbench.datasets import create_composite_dataset
from lofbench.renderers import get_renderer
from lofbench.scorers import lof_composite_scorer

from lofbench.tasks.prompts import COMPOSITE_SYSTEM_PROMPT, COMPOSITE_USER_TEMPLATE


@task
def composite_lof_task(
    n_groups: int = 100,
    group_size: int = 8,
    seed: int = 2025,
    renderer: str = "canonical",
    render_seed: int | None = None,
    **renderer_kwargs: Any,
) -> Task:
    """Composite LoF expression evaluation task.

    Evaluates model ability to reduce multiple Laws of Form expressions
    and count how many are marked. This tests systematic reasoning
    across a batch of problems.

    Random baseline for group_size=8: ~11% (1/9)

    Args:
        n_groups: Number of test groups
        group_size: Expressions per group
        seed: Seed for test case generation
        renderer: Renderer name (canonical, noisy_parens, etc.)
        render_seed: Seed for reproducible rendering
        **renderer_kwargs: Additional renderer configuration

    Returns:
        inspect-ai Task instance
    """
    renderer_instance = get_renderer(renderer, **renderer_kwargs)

    dataset = create_composite_dataset(
        n_groups=n_groups,
        group_size=group_size,
        seed=seed,
        renderer=renderer_instance,
        render_seed=render_seed,
    )

    return Task(
        dataset=dataset,
        solver=[
            system_message(COMPOSITE_SYSTEM_PROMPT),
            prompt_template(COMPOSITE_USER_TEMPLATE),
            generate(),
        ],
        scorer=lof_composite_scorer(),
    )
