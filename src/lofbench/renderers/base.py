"""Base classes for form renderers."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RenderedForm:
    """A form that has been rendered by a renderer."""

    original: str
    rendered: str
    renderer_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FormRenderer(ABC):
    """Abstract base class for form renderers.

    Renderers transform form strings into alternative representations
    while preserving the underlying structure. Examples include:
    - Identity/canonical rendering (no transformation)
    - Noisy parentheses (substituting different bracket types)
    - Whitespace variations
    - Alternative notations
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this renderer."""
        pass

    @abstractmethod
    def render(self, form_string: str, rng: random.Random | None = None) -> RenderedForm:
        """Render a single form string.

        Args:
            form_string: The input form string to render
            rng: Optional Random instance for reproducibility

        Returns:
            RenderedForm containing the original, rendered version, and metadata
        """
        pass

    def render_batch(self, forms: list[str], seed: int | None = None) -> list[RenderedForm]:
        """Render a batch of forms with optional seed for reproducibility.

        Args:
            forms: List of form strings to render
            seed: Optional seed for reproducible rendering

        Returns:
            List of RenderedForm objects
        """
        rng = random.Random(seed) if seed is not None else None
        return [self.render(form, rng) for form in forms]
