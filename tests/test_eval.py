"""Tests for lofbench.eval module."""

from lofbench.eval import extract_answer, extract_composite_answer


class TestExtractAnswer:
    """Test answer extraction from LLM responses."""

    def test_extract_tagged_marked(self):
        assert extract_answer("<answer>marked</answer>") == "()"
        assert extract_answer("<answer>MARKED</answer>") == "()"
        assert extract_answer("<answer>()</answer>") == "()"

    def test_extract_tagged_unmarked(self):
        assert extract_answer("<answer>unmarked</answer>") == "void"
        assert extract_answer("<answer>void</answer>") == "void"
        assert extract_answer("<answer>VOID</answer>") == "void"

    def test_extract_fallback_marked(self):
        assert extract_answer("The final answer is marked.") == "()"
        assert extract_answer("This reduces to ()") == "()"

    def test_extract_fallback_unmarked(self):
        assert extract_answer("The expression is void.") == "void"
        assert extract_answer("The result is void") == "void"

    def test_extract_none(self):
        assert extract_answer(None) == "unknown"

    def test_extract_ambiguous(self):
        assert extract_answer("no clear answer here") == "unknown"


class TestExtractCompositeAnswer:
    """Test composite answer extraction (JSON format)."""

    def test_extract_json_fenced(self):
        response = '''Here are my answers:
```json
{"E1": "marked", "E2": "unmarked", "E3": "marked", "total_marked": 2}
```
'''
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

    def test_extract_none(self):
        result = extract_composite_answer(None, 8)
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
