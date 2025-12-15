"""Single expression LoF evaluation task."""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_ai.solver import generate, prompt_template, system_message

from lofbench.datasets import create_single_dataset
from lofbench.renderers import get_renderer
from lofbench.scorers import lof_single_scorer

from .prompts import SINGLE_SYSTEM_PROMPT, SINGLE_USER_TEMPLATE


@task
def single_lof_task(
    n: int = 100,
    seed: int = 2025,
    renderer: str = "canonical",
    render_seed: int | None = None,
    **renderer_kwargs: Any,
) -> Task:
    """Single LoF expression evaluation task.

    Evaluates model ability to reduce individual Laws of Form expressions
    to their canonical form (marked or unmarked).

    Args:
        n: Number of test cases
        seed: Seed for test case generation
        renderer: Renderer name (canonical, noisy_parens, etc.)
        render_seed: Seed for reproducible rendering
        **renderer_kwargs: Additional renderer configuration

    Returns:
        inspect-ai Task instance
    """
    renderer_instance = get_renderer(renderer, **renderer_kwargs)

    dataset = create_single_dataset(
        n=n,
        seed=seed,
        renderer=renderer_instance,
        render_seed=render_seed,
    )

    return Task(
        dataset=dataset,
        solver=[
            system_message(SINGLE_SYSTEM_PROMPT),
            prompt_template(SINGLE_USER_TEMPLATE),
            generate(),
        ],
        scorer=lof_single_scorer(),
    )
