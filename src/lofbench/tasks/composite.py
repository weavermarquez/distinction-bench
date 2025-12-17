"""Composite (multi-expression) LoF evaluation task."""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import chain_of_thought, generate, prompt_template, system_message

from lofbench.core import DIFFICULTY_CONFIGS
from lofbench.datasets import create_composite_dataset
from lofbench.renderers import get_renderer
from lofbench.scorers import lof_composite_scorer
from lofbench.tasks.prompts import COMPOSITE_SYSTEM_PROMPT, COMPOSITE_USER_TEMPLATE


@task
def composite_lof_task(
    n_groups: int = 100,
    group_size: int = 4,
    seed: int = 2025,
    renderer: str = "canonical",
    render_seed: int | None = None,
    renderer_config: dict | None = None,
) -> Task:
    """Composite LoF expression evaluation task.

    Evaluates model ability to reduce multiple Laws of Form expressions
    and count how many are marked. Tests systematic reasoning and consistency.

    Uses 3 epochs with at_least_2 reducer: must get all_correct on 2+ attempts.
    Random baseline for group_size=4 with at_least_2(3): ~1.1%
    Human baseline expectation: 98-100%

    Args:
        n_groups: Number of test groups
        group_size: Expressions per group
        seed: Seed for test case generation
        renderer: Renderer name (canonical, noisy_parens, etc.)
        render_seed: Seed for reproducible rendering
        renderer_config: Dict of renderer options (e.g., {"mismatched": true})

    Returns:
        inspect-ai Task instance
    """
    renderer_instance = get_renderer(renderer, **(renderer_config or {}))

    dataset = create_composite_dataset(
        n_groups=n_groups,
        group_size=group_size,
        seed=seed,
        renderer=renderer_instance,
        render_seed=render_seed,
    )

    # Calculate difficulty distribution for metadata
    base_per_diff = n_groups // len(DIFFICULTY_CONFIGS)
    remainder = n_groups % len(DIFFICULTY_CONFIGS)
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
            system_message(COMPOSITE_SYSTEM_PROMPT),
            chain_of_thought(),
            prompt_template(COMPOSITE_USER_TEMPLATE),
            generate(),
        ],
        scorer=lof_composite_scorer(),
        epochs=Epochs(3, "at_least_2"),
        config=GenerateConfig(
            # temperature=1,
            max_tokens=64000,  # Even playing field; models cap to their own limits
        ),
        metadata={
            "n_groups": n_groups,
            "group_size": group_size,
            "seed": seed,
            "renderer": renderer,
            "render_seed": render_seed,
            "renderer_config": renderer_config,
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
