"""lofbench - Laws of Form benchmark toolkit."""

from lofbench.core import (
    canonical_string,
    evaluate,
    form_depth,
    form_to_string,
    generate_composite_test_cases,
    generate_form_string,
    generate_test_cases,
    simplify_string,
    string_depth,
    string_to_form,
)
from lofbench.renderers import (
    FormRenderer,
    RenderedForm,
    get_renderer,
    list_renderers,
)
from lofbench.tasks import composite_lof_task, single_lof_task

__all__ = [
    # Core
    "form_to_string",
    "string_to_form",
    "form_depth",
    "string_depth",
    "simplify_string",
    "canonical_string",
    "evaluate",
    "generate_form_string",
    "generate_test_cases",
    "generate_composite_test_cases",
    # inspect-ai tasks
    "single_lof_task",
    "composite_lof_task",
    # Renderers
    "FormRenderer",
    "RenderedForm",
    "get_renderer",
    "list_renderers",
]

__version__ = "0.2.0"
