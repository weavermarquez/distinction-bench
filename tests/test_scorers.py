"""Tests for lofbench.scorers module."""

from lofbench.scorers import extract_composite_answer, extract_single_answer


class TestExtractSingleAnswer:
    """Test answer extraction from LLM responses."""

    def test_extract_tagged_marked(self):
        assert extract_single_answer("<answer>marked</answer>") == "marked"
        assert extract_single_answer("<answer>MARKED</answer>") == "marked"
        assert extract_single_answer("<answer>()</answer>") == "marked"

    def test_extract_tagged_unmarked(self):
        assert extract_single_answer("<answer>unmarked</answer>") == "unmarked"
        assert extract_single_answer("<answer>void</answer>") == "unmarked"
        assert extract_single_answer("<answer>VOID</answer>") == "unmarked"

    def test_extract_fallback_marked(self):
        assert extract_single_answer("The final answer is marked.") == "marked"
        assert extract_single_answer("This reduces to ()") == "marked"

    def test_extract_fallback_unmarked(self):
        assert extract_single_answer("The expression is void.") == "unmarked"
        assert extract_single_answer("The result is void") == "unmarked"

    def test_extract_empty(self):
        assert extract_single_answer("") == "unknown"

    def test_extract_ambiguous(self):
        assert extract_single_answer("no clear answer here") == "unknown"


class TestExtractCompositeAnswer:
    """Test composite answer extraction (JSON format)."""

    def test_extract_json_fenced(self):
        response = """Here are my answers:
```json
{"E1": "marked", "E2": "unmarked", "E3": "marked", "total_marked": 2}
```
"""
        result = extract_composite_answer(response, 3)
        assert result["items"] == ["marked", "unmarked", "marked"]
        assert result["total_marked"] == 2

    def test_extract_json_bare(self):
        response = '{"E1": "marked", "E2": "unmarked", "total_marked": 1}'
        result = extract_composite_answer(response, 2)
        assert result["items"] == ["marked", "unmarked"]
        assert result["total_marked"] == 1

    def test_extract_fallback_total(self):
        response = "I think total_marked: 5"
        result = extract_composite_answer(response, 8)
        assert result["items"] == ["unknown"] * 8
        assert result["total_marked"] == 5

    def test_extract_empty(self):
        result = extract_composite_answer("", 8)
        assert result["items"] == ["unknown"] * 8
        assert result["total_marked"] == -1

    def test_extract_invalid_total(self):
        response = '{"E1": "marked", "total_marked": 100}'
        result = extract_composite_answer(response, 8)
        assert result["total_marked"] == -1

    def test_extract_unparseable(self):
        result = extract_composite_answer("no json here", 8)
        assert result["items"] == ["unknown"] * 8
        assert result["total_marked"] == -1
