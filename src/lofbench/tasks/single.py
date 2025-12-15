"""Single expression LoF evaluation task."""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import chain_of_thought, generate, prompt_template, system_message

from lofbench.core import DIFFICULTY_CONFIGS
from lofbench.datasets import create_single_dataset
from lofbench.renderers import get_renderer
from lofbench.scorers import lof_single_scorer

from lofbench.tasks.prompts import SINGLE_SYSTEM_PROMPT, SINGLE_USER_TEMPLATE


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

    # Calculate difficulty distribution for metadata
    base_per_diff = n // len(DIFFICULTY_CONFIGS)
    remainder = n % len(DIFFICULTY_CONFIGS)
    difficulty_counts = {
        config[0]: base_per_diff + (1 if i < remainder else 0)
        for i, config in enumerate(DIFFICULTY_CONFIGS)
    }

    # Model-specific reasoning flags should be passed via CLI:
    #   Claude Opus 4.5: --reasoning-tokens 10000
    #   GPT-5.2: --reasoning-effort high
    #   Gemini 3.0: --reasoning-tokens 10000
    return Task(
        dataset=dataset,
        solver=[
            system_message(SINGLE_SYSTEM_PROMPT),
            chain_of_thought(),
            prompt_template(SINGLE_USER_TEMPLATE),
            generate(),
        ],
        scorer=lof_single_scorer(),
        config=GenerateConfig(
            temperature=1,
            max_tokens=64000,  # Opus max; other models will cap to their limits
        ),
        metadata={
            "n": n,
            "seed": seed,
            "renderer": renderer,
            "render_seed": render_seed,
            "difficulty_distribution": difficulty_counts,
            "difficulty_configs": {
                config[0]: {
                    "min_depth": config[1],
                    "max_depth": config[2],
                    "max_width": config[3],
                    "max_marks": config[4],
                }
                for config in DIFFICULTY_CONFIGS
            },
        },
    )
