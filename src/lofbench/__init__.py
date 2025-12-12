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
]

__version__ = "0.1.0"
