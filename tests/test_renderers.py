"""Tests for lofbench.renderers module."""

import random

from lofbench.renderers import (
    BRACKET_PAIRS,
    CanonicalRenderer,
    NoisyParensConfig,
    NoisyParensRenderer,
    get_renderer,
    list_renderers,
)


class TestRendererRegistry:
    """Test renderer registration and lookup."""

    def test_list_renderers(self):
        renderers = list_renderers()
        assert "canonical" in renderers
        assert "noisy_parens" in renderers

    def test_get_canonical_renderer(self):
        renderer = get_renderer("canonical")
        assert isinstance(renderer, CanonicalRenderer)
        assert renderer.name == "canonical"

    def test_get_noisy_parens_renderer(self):
        renderer = get_renderer("noisy_parens")
        assert isinstance(renderer, NoisyParensRenderer)
        assert renderer.name == "noisy_parens"

    def test_get_unknown_renderer(self):
        try:
            get_renderer("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)


class TestCanonicalRenderer:
    """Test canonical (identity) renderer."""

    def test_render_simple(self):
        renderer = CanonicalRenderer()
        result = renderer.render("()")
        assert result.original == "()"
        assert result.rendered == "()"
        assert result.renderer_name == "canonical"
        assert result.metadata == {"spacing": False}

    def test_render_nested(self):
        renderer = CanonicalRenderer()
        result = renderer.render("(())()")
        assert result.original == "(())()"
        assert result.rendered == "(())()"

    def test_render_empty(self):
        renderer = CanonicalRenderer()
        result = renderer.render("")
        assert result.original == ""
        assert result.rendered == ""

    def test_render_batch(self):
        renderer = CanonicalRenderer()
        forms = ["()", "(())", "()()"]
        results = renderer.render_batch(forms)
        assert len(results) == 3
        assert all(r.original == r.rendered for r in results)

    def test_render_with_spacing(self):
        renderer = CanonicalRenderer(spacing=True)
        rng = random.Random(42)
        result = renderer.render("(())()", rng)
        assert result.original == "(())()"
        assert result.metadata == {"spacing": True}
        # Spacing may add spaces between tokens
        assert result.rendered.replace(" ", "") == "(())()"

    def test_spacing_kwargs_override(self):
        """Test that spacing kwarg works for CLI convenience."""
        renderer = CanonicalRenderer(spacing=True)
        assert renderer.config.spacing is True


class TestNoisyParensRenderer:
    """Test noisy parentheses renderer."""

    def test_render_transforms(self):
        renderer = NoisyParensRenderer()
        rng = random.Random(42)
        result = renderer.render("(())", rng)
        assert result.original == "(())"
        # Should be transformed (may still be () by chance, but very unlikely)
        assert result.renderer_name == "noisy_parens"
        assert "substitution_map" in result.metadata

    def test_render_matched_mode(self):
        config = NoisyParensConfig(mismatched=False)
        renderer = NoisyParensRenderer(config)
        rng = random.Random(123)
        result = renderer.render("(())", rng)
        assert result.metadata["mismatched"] is False

    def test_render_mismatched_mode(self):
        config = NoisyParensConfig(mismatched=True)
        renderer = NoisyParensRenderer(config)
        rng = random.Random(456)
        result = renderer.render("(())", rng)
        assert result.metadata["mismatched"] is True

    def test_render_empty(self):
        renderer = NoisyParensRenderer()
        result = renderer.render("")
        assert result.original == ""
        assert result.rendered == ""

    def test_render_reproducibility(self):
        renderer = NoisyParensRenderer()
        rng1 = random.Random(999)
        rng2 = random.Random(999)
        result1 = renderer.render("(())()", rng1)
        result2 = renderer.render("(())()", rng2)
        assert result1.rendered == result2.rendered

    def test_custom_bracket_pairs(self):
        config = NoisyParensConfig(
            bracket_pairs=[("[", "]"), ("{", "}")],
            mismatched=False,
        )
        renderer = NoisyParensRenderer(config)
        rng = random.Random(42)
        result = renderer.render("()", rng)
        # Should only use [ ] or { }
        assert result.rendered in ["[]", "{}"]

    def test_kwargs_override(self):
        """Test that kwargs override config values (for CLI convenience)."""
        # Direct kwargs without config
        renderer = NoisyParensRenderer(mismatched=True)
        assert renderer.config.mismatched is True

        # kwargs override config values
        config = NoisyParensConfig(mismatched=False)
        renderer = NoisyParensRenderer(config=config, mismatched=True)
        assert renderer.config.mismatched is True


class TestBracketPairs:
    """Test bracket pairs constant."""

    def test_bracket_pairs_count(self):
        assert len(BRACKET_PAIRS) == 10

    def test_ascii_pairs_included(self):
        assert ("(", ")") in BRACKET_PAIRS
        assert ("[", "]") in BRACKET_PAIRS
        assert ("{", "}") in BRACKET_PAIRS
        assert ("<", ">") in BRACKET_PAIRS

    def test_unicode_pairs_included(self):
        # Mathematical angle brackets
        assert ("\u27e8", "\u27e9") in BRACKET_PAIRS
