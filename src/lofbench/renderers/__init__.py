"""Form renderers for transforming canonical forms into alternative representations.

This module provides a flexible system for rendering Laws of Form expressions
in various notations while preserving their structural meaning.
"""

from __future__ import annotations

from typing import Any

from .base import FormRenderer, RenderedForm
from .canonical import CanonicalConfig, CanonicalRenderer
from .noisy_parens import BRACKET_PAIRS, NoisyParensConfig, NoisyParensRenderer

__all__ = [
    "FormRenderer",
    "RenderedForm",
    "CanonicalRenderer",
    "CanonicalConfig",
    "NoisyParensRenderer",
    "NoisyParensConfig",
    "BRACKET_PAIRS",
    "get_renderer",
    "register_renderer",
    "list_renderers",
]


# Registry mapping renderer names to their classes
_RENDERER_REGISTRY: dict[str, type[FormRenderer]] = {
    "canonical": CanonicalRenderer,
    "noisy_parens": NoisyParensRenderer,
}


def get_renderer(name: str, **kwargs: Any) -> FormRenderer:
    """Get a renderer instance by name.

    Args:
        name: The name of the renderer to instantiate
        **kwargs: Additional keyword arguments to pass to the renderer constructor

    Returns:
        An instance of the requested renderer

    Raises:
        ValueError: If the renderer name is not registered

    Examples:
        >>> renderer = get_renderer("canonical")
        >>> renderer = get_renderer("noisy_parens", config=NoisyParensConfig(mismatched=True))
    """
    if name not in _RENDERER_REGISTRY:
        available = ", ".join(sorted(_RENDERER_REGISTRY.keys()))
        raise ValueError(f"Unknown renderer: {name!r}. Available renderers: {available}")

    renderer_cls = _RENDERER_REGISTRY[name]
    return renderer_cls(**kwargs)


def register_renderer(name: str, renderer_cls: type[FormRenderer]) -> None:
    """Register a new renderer class.

    This allows users to add custom renderers to the system.

    Args:
        name: The unique name to register the renderer under
        renderer_cls: The FormRenderer subclass to register

    Raises:
        TypeError: If renderer_cls is not a subclass of FormRenderer
        ValueError: If the name is already registered

    Examples:
        >>> class MyRenderer(FormRenderer):
        ...     @property
        ...     def name(self):
        ...         return "my_renderer"
        ...     def render(self, form_string, rng=None):
        ...         return RenderedForm(form_string, form_string, self.name, {})
        >>> register_renderer("my_renderer", MyRenderer)
    """
    if not isinstance(renderer_cls, type) or not issubclass(renderer_cls, FormRenderer):
        raise TypeError(
            f"renderer_cls must be a subclass of FormRenderer, got {type(renderer_cls)}"
        )

    if name in _RENDERER_REGISTRY:
        raise ValueError(f"Renderer {name!r} is already registered")

    _RENDERER_REGISTRY[name] = renderer_cls


def list_renderers() -> list[str]:
    """List all registered renderer names.

    Returns:
        Sorted list of registered renderer names

    Examples:
        >>> list_renderers()
        ['canonical', 'noisy_parens']
    """
    return sorted(_RENDERER_REGISTRY.keys())
