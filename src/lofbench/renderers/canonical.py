"""Canonical (identity) renderer."""

from __future__ import annotations

import random

from .base import FormRenderer, RenderedForm


class CanonicalRenderer(FormRenderer):
    """Identity renderer that returns forms unchanged.

    This renderer provides a baseline for comparing other rendering
    strategies. It simply returns the input form without any transformation.
    """

    @property
    def name(self) -> str:
        return "canonical"

    def render(self, form_string: str, rng: random.Random | None = None) -> RenderedForm:
        """Return the form unchanged.

        Args:
            form_string: The input form string
            rng: Unused (included for interface compatibility)

        Returns:
            RenderedForm with original and rendered being identical
        """
        return RenderedForm(
            original=form_string, rendered=form_string, renderer_name=self.name, metadata={}
        )
