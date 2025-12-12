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
    """Test composite answer extraction."""

    def test_extract_tagged(self):
        response = """
        <answer>
        E1: marked
        E2: unmarked
        E3: marked
        total_marked: 2
        </answer>
        """
        assert extract_composite_answer(response, 8) == 2

    def test_extract_fallback(self):
        assert extract_composite_answer("total: 5", 8) == 5
        assert extract_composite_answer("count: 4", 8) == 4

    def test_extract_none(self):
        assert extract_composite_answer(None, 8) == -1

    def test_extract_out_of_range(self):
        assert extract_composite_answer("total_marked: 100", 8) == -1
        assert extract_composite_answer("total_marked: -1", 8) == -1

    def test_extract_unparseable(self):
        assert extract_composite_answer("no number here", 8) == -1
