"""
Unit tests for MathGrader class and its grading functions.

Tests cover:
- Regex pattern matching (format matching)
- Format scoring (exact and approximate)
- Answer checking logic
- Number extraction and comparison
- Edge cases in grading
"""

import pytest
import re


class TestMathGraderInitialization:
    """Tests for MathGrader initialization."""

    def test_grader_initialization(self):
        """Test that MathGrader initializes with correct tokens."""
        from rl_finetuning import MathGrader

        grader = MathGrader()

        assert grader.reasoning_start == "<start_working_out>"
        assert grader.reasoning_end == "<end_working_out>"
        assert grader.solution_start == "<SOLUTION>"
        assert grader.solution_end == "</SOLUTION>"

    def test_match_format_regex_compiled(self):
        """Test that match_format regex is properly compiled."""
        from rl_finetuning import MathGrader

        grader = MathGrader()

        assert isinstance(grader.match_format, re.Pattern)
        assert isinstance(grader.match_numbers, re.Pattern)


class TestMatchFormatExactly:
    """Tests for match_format_exactly scoring."""

    def test_exact_format_match_scores_high(self, sample_math_completions_correct):
        """Test that exactly matching format scores 3.0."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        scores = grader.match_format_exactly(sample_math_completions_correct)

        assert len(scores) == 1
        assert scores[0] == 3.0

    def test_wrong_format_scores_zero(self, sample_math_completions_wrong_format):
        """Test that wrong format scores 0."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        scores = grader.match_format_exactly(sample_math_completions_wrong_format)

        assert len(scores) == 1
        assert scores[0] == 0

    def test_partial_format_scores_zero(self):
        """Test that partial format (missing tags) scores 0."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        # Missing end_working_out tag
        completions = [
            [{"content": "<start_working_out>Work<SOLUTION>4</SOLUTION>"}]
        ]
        scores = grader.match_format_exactly(completions)

        assert scores[0] == 0

    def test_multiple_completions(self):
        """Test scoring multiple completions at once."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        completions = [
            [{"content": "<start_working_out>work<end_working_out><SOLUTION>4</SOLUTION>"}],
            [{"content": "Just the answer: 4"}],
            [{"content": "<start_working_out>more work<end_working_out><SOLUTION>5</SOLUTION>"}],
        ]
        scores = grader.match_format_exactly(completions)

        assert len(scores) == 3
        assert scores[0] == 3.0  # Correct format
        assert scores[1] == 0     # Wrong format
        assert scores[2] == 3.0  # Correct format


class TestMatchFormatApproximately:
    """Tests for match_format_approximately scoring."""

    def test_perfect_format_scores_positive(self):
        """Test that perfect format with single occurrence of each tag scores +2.0."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        completions = [
            [{"content": "<start_working_out>work<end_working_out><SOLUTION>4</SOLUTION>"}]
        ]
        scores = grader.match_format_approximately(completions)

        # 4 tags, each appearing once = 4 * 0.5 = 2.0
        assert scores[0] == 2.0

    def test_duplicate_tags_penalized(self):
        """Test that duplicate tags are penalized."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        # Duplicate start_working_out tag
        completions = [
            [{"content": "<start_working_out>work<start_working_out><end_working_out><SOLUTION>4</SOLUTION>"}]
        ]
        scores = grader.match_format_approximately(completions)

        # start_working_out appears twice: -0.5
        # end_working_out appears once: +0.5
        # SOLUTION appears once: +0.5
        # /SOLUTION appears once: +0.5
        assert scores[0] == 1.0

    def test_missing_tags_penalized(self):
        """Test that missing tags are penalized."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        # Missing all tags
        completions = [
            [{"content": "The answer is 4."}]
        ]
        scores = grader.match_format_approximately(completions)

        # All 4 tags missing: 4 * -0.5 = -2.0
        assert scores[0] == -2.0

    def test_partial_tags_mixed_score(self):
        """Test partial tag presence gives mixed score."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        # Only solution tags present
        completions = [
            [{"content": "<SOLUTION>4</SOLUTION>"}]
        ]
        scores = grader.match_format_approximately(completions)

        # start_working_out missing: -0.5
        # end_working_out missing: -0.5
        # SOLUTION once: +0.5
        # /SOLUTION once: +0.5
        assert scores[0] == 0.0


class TestCheckAnswer:
    """Tests for check_answer scoring."""

    def test_exact_answer_match_scores_high(self):
        """Test that exact answer match scores 3.0."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "What is 2 + 2?"}]]
        completions = [
            [{"content": "<start_working_out>2+2=4<end_working_out><SOLUTION>4</SOLUTION>"}]
        ]
        answer = ["4"]

        scores = grader.check_answer(prompts, completions, answer)

        assert len(scores) == 1
        assert scores[0] == 3.0

    def test_answer_with_whitespace_scores_partial(self):
        """Test that answer with extra whitespace scores 1.5."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "What is 2 + 2?"}]]
        completions = [
            [{"content": "<start_working_out>work<end_working_out><SOLUTION> 4 </SOLUTION>"}]
        ]
        answer = ["4"]

        scores = grader.check_answer(prompts, completions, answer)

        # Whitespace trimmed match = 1.5
        assert scores[0] == 1.5

    def test_close_numerical_answer_rewarded(self):
        """Test that numerically close answers get partial credit."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "Calculate something"}]]
        completions = [
            [{"content": "<start_working_out>work<end_working_out><SOLUTION>95</SOLUTION>"}]
        ]
        answer = ["100"]

        scores = grader.check_answer(prompts, completions, answer)

        # 95/100 = 0.95, within 0.9-1.1 range = 0.5
        assert scores[0] == 0.5

    def test_very_wrong_answer_penalized(self):
        """Test that very wrong numerical answers are penalized."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "Calculate something"}]]
        completions = [
            [{"content": "<start_working_out>work<end_working_out><SOLUTION>50</SOLUTION>"}]
        ]
        answer = ["100"]

        scores = grader.check_answer(prompts, completions, answer)

        # 50/100 = 0.5, outside 0.8-1.2 range = -1.0
        assert scores[0] == -1.0

    def test_no_format_match_scores_zero(self):
        """Test that response without format gets 0 for answer check."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "What is 2 + 2?"}]]
        completions = [
            [{"content": "The answer is 4"}]  # No SOLUTION tags
        ]
        answer = ["4"]

        scores = grader.check_answer(prompts, completions, answer)

        assert scores[0] == 0

    def test_non_numeric_answer_handling(self):
        """Test handling of non-numeric answers."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "What color is the sky?"}]]
        completions = [
            [{"content": "<start_working_out>thinking<end_working_out><SOLUTION>blue</SOLUTION>"}]
        ]
        answer = ["red"]

        scores = grader.check_answer(prompts, completions, answer)

        # Non-matching non-numeric answer gets penalized
        assert scores[0] == -0.5


class TestCheckNumbers:
    """Tests for check_numbers function."""

    def test_exact_number_match(self):
        """Test exact number match scores 1.5."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "What is 10?"}]]
        completions = [
            [{"content": "<SOLUTION>10</SOLUTION>"}]
        ]
        answer = ["10"]

        scores = grader.check_numbers(prompts, completions, answer)

        assert scores[0] == 1.5

    def test_number_extraction_from_solution(self):
        """Test that numbers are extracted from SOLUTION tags."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "Calculate"}]]
        completions = [
            [{"content": "Some text <SOLUTION>42.5</SOLUTION>"}]
        ]
        answer = ["42.5"]

        scores = grader.check_numbers(prompts, completions, answer)

        assert scores[0] == 1.5

    def test_no_solution_tag_scores_zero(self):
        """Test that missing SOLUTION tag scores 0."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "Calculate"}]]
        completions = [
            [{"content": "The answer is 42"}]
        ]
        answer = ["42"]

        scores = grader.check_numbers(prompts, completions, answer)

        assert scores[0] == 0

    def test_wrong_number_scores_zero(self):
        """Test that wrong number scores 0."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        prompts = [[{"content": "Calculate"}]]
        completions = [
            [{"content": "<SOLUTION>5</SOLUTION>"}]
        ]
        answer = ["10"]

        scores = grader.check_numbers(prompts, completions, answer)

        assert scores[0] == 0.0


class TestRegexPatterns:
    """Tests for regex pattern matching."""

    def test_match_format_multiline(self):
        """Test that match_format handles multiline content."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        # The regex uses ^ and $ anchors, so content must match from start to end
        response = """<start_working_out>
        Let me think step by step:
        1. First step
        2. Second step
        <end_working_out>
        <SOLUTION>42</SOLUTION>"""

        match = grader.match_format.search(response)
        assert match is not None
        assert match.group(1) == "42"

    def test_match_numbers_extracts_first_number(self):
        """Test that match_numbers extracts the first number after SOLUTION."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        response = "<SOLUTION>The answer is 123.45</SOLUTION>"

        match = grader.match_numbers.search(response)
        assert match is not None
        assert match.group(1) == "123.45"

    def test_match_format_with_extra_whitespace(self):
        """Test format matching with extra whitespace."""
        from rl_finetuning import MathGrader

        grader = MathGrader()
        response = "   <start_working_out>work<end_working_out>  <SOLUTION>4</SOLUTION>   "

        match = grader.match_format.search(response)
        assert match is not None


class TestBaseGraderFunction:
    """Tests for BaseGraderFunction and eval_fun decorator."""

    def test_base_grader_raises_not_implemented(self):
        """Test that BaseGraderFunction.grade raises NotImplementedError."""
        from rl_finetuning import BaseGraderFunction

        grader = BaseGraderFunction()

        with pytest.raises(NotImplementedError):
            grader.grade("test")

    def test_eval_fun_decorator(self):
        """Test that eval_fun decorator wraps functions correctly."""
        from rl_finetuning import eval_fun, BaseGraderFunction

        @eval_fun
        def simple_grader(response):
            return 1.0 if "correct" in response else 0.0

        assert isinstance(simple_grader, BaseGraderFunction)
        assert simple_grader.grade("correct answer") == 1.0
        assert simple_grader.grade("wrong answer") == 0.0
