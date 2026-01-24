"""
Unit tests for Pydantic models used in RL fine-tuning.

Tests cover:
- TrajectoryScore model
- TrajectoryGradingOutput model
- Field validation
- Score constraints (0.0 to 1.0)
"""

import json
from unittest.mock import MagicMock

import pytest


class TestTrajectoryScore:
    """Tests for TrajectoryScore Pydantic model."""

    def test_create_valid_trajectory_score(self):
        """Test creating a valid TrajectoryScore."""
        from rl_finetuning import TrajectoryScore

        score = TrajectoryScore(trajectory="This is a test response", score=0.85)

        assert score.trajectory == "This is a test response"
        assert score.score == 0.85

    def test_score_at_minimum_boundary(self):
        """Test score at minimum boundary (0.0)."""
        from rl_finetuning import TrajectoryScore

        score = TrajectoryScore(trajectory="Response", score=0.0)

        assert score.score == 0.0

    def test_score_at_maximum_boundary(self):
        """Test score at maximum boundary (1.0)."""
        from rl_finetuning import TrajectoryScore

        score = TrajectoryScore(trajectory="Response", score=1.0)

        assert score.score == 1.0

    def test_score_below_minimum_raises_error(self):
        """Test that score below 0.0 raises validation error."""
        from rl_finetuning import TrajectoryScore
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TrajectoryScore(trajectory="Response", score=-0.1)

    def test_score_above_maximum_raises_error(self):
        """Test that score above 1.0 raises validation error."""
        from rl_finetuning import TrajectoryScore
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TrajectoryScore(trajectory="Response", score=1.1)

    def test_trajectory_can_be_empty_string(self):
        """Test that trajectory can be an empty string."""
        from rl_finetuning import TrajectoryScore

        score = TrajectoryScore(trajectory="", score=0.5)

        assert score.trajectory == ""

    def test_trajectory_with_special_characters(self):
        """Test trajectory with special characters."""
        from rl_finetuning import TrajectoryScore

        special_text = "Response with <tags> and 'quotes' and \"double quotes\""
        score = TrajectoryScore(trajectory=special_text, score=0.7)

        assert score.trajectory == special_text

    def test_trajectory_with_unicode(self):
        """Test trajectory with unicode characters."""
        from rl_finetuning import TrajectoryScore

        unicode_text = "Response with unicode: cafe"
        score = TrajectoryScore(trajectory=unicode_text, score=0.6)

        assert score.trajectory == unicode_text

    def test_score_with_decimal_precision(self):
        """Test score with high decimal precision."""
        from rl_finetuning import TrajectoryScore

        score = TrajectoryScore(trajectory="Response", score=0.123456789)

        assert score.score == pytest.approx(0.123456789)


class TestTrajectoryGradingOutput:
    """Tests for TrajectoryGradingOutput Pydantic model."""

    def test_create_with_single_result(self, sample_trajectory_scores):
        """Test creating output with a single result."""
        from rl_finetuning import TrajectoryScore, TrajectoryGradingOutput

        single_score = TrajectoryScore(**sample_trajectory_scores[0])
        output = TrajectoryGradingOutput(results=[single_score])

        assert len(output.results) == 1
        assert output.results[0].trajectory == "Response 1"
        assert output.results[0].score == 0.85

    def test_create_with_multiple_results(self, sample_trajectory_scores):
        """Test creating output with multiple results."""
        from rl_finetuning import TrajectoryScore, TrajectoryGradingOutput

        scores = [TrajectoryScore(**s) for s in sample_trajectory_scores]
        output = TrajectoryGradingOutput(results=scores)

        assert len(output.results) == 3
        assert output.results[0].score == 0.85
        assert output.results[1].score == 0.72
        assert output.results[2].score == 0.91

    def test_create_with_empty_results(self):
        """Test creating output with empty results list."""
        from rl_finetuning import TrajectoryGradingOutput

        output = TrajectoryGradingOutput(results=[])

        assert len(output.results) == 0

    def test_create_from_dict(self, sample_trajectory_scores):
        """Test creating output from dict (simulating JSON parsing)."""
        from rl_finetuning import TrajectoryGradingOutput

        data = {"results": sample_trajectory_scores}
        output = TrajectoryGradingOutput(**data)

        assert len(output.results) == 3

    def test_results_type_validation(self):
        """Test that results must be a list of TrajectoryScore."""
        from rl_finetuning import TrajectoryGradingOutput
        from pydantic import ValidationError

        # Invalid: results is not a list
        with pytest.raises(ValidationError):
            TrajectoryGradingOutput(results="not a list")

    def test_results_item_type_validation(self):
        """Test that results items must be valid TrajectoryScore."""
        from rl_finetuning import TrajectoryGradingOutput
        from pydantic import ValidationError

        # Invalid: score out of range
        with pytest.raises(ValidationError):
            TrajectoryGradingOutput(
                results=[{"trajectory": "Test", "score": 2.0}]  # score > 1.0
            )


class TestParseLiteLLMJsonResponse:
    """Tests for parse_litellm_json_response static method."""

    def test_parse_valid_json_response(self):
        """Test parsing a valid JSON response from LiteLLM."""
        from rl_finetuning import RLFinetuningEngine, TrajectoryGradingOutput
        from unittest.mock import MagicMock

        # Create mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''```json
{
    "results": [
        {"trajectory": "Response 1", "score": 0.8},
        {"trajectory": "Response 2", "score": 0.6}
    ]
}
```'''

        result = RLFinetuningEngine.parse_litellm_json_response(mock_response)

        assert isinstance(result, TrajectoryGradingOutput)
        assert len(result.results) == 2
        assert result.results[0].score == 0.8

    def test_parse_json_without_markdown_wrapper(self):
        """Test parsing JSON without markdown code block."""
        from rl_finetuning import RLFinetuningEngine

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
    "results": [
        {"trajectory": "Test", "score": 0.5}
    ]
}'''

        result = RLFinetuningEngine.parse_litellm_json_response(mock_response)

        assert len(result.results) == 1
        assert result.results[0].score == 0.5

    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises error."""
        from rl_finetuning import RLFinetuningEngine
        import json

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"

        with pytest.raises(json.JSONDecodeError):
            RLFinetuningEngine.parse_litellm_json_response(mock_response)


class TestModelSerialization:
    """Tests for model serialization and deserialization."""

    def test_trajectory_score_to_dict(self):
        """Test TrajectoryScore serialization to dict."""
        from rl_finetuning import TrajectoryScore

        score = TrajectoryScore(trajectory="Test", score=0.75)
        data = score.model_dump()

        assert isinstance(data, dict)
        assert data["trajectory"] == "Test"
        assert data["score"] == 0.75

    def test_trajectory_grading_output_to_dict(self, sample_trajectory_scores):
        """Test TrajectoryGradingOutput serialization to dict."""
        from rl_finetuning import TrajectoryScore, TrajectoryGradingOutput

        scores = [TrajectoryScore(**s) for s in sample_trajectory_scores]
        output = TrajectoryGradingOutput(results=scores)
        data = output.model_dump()

        assert isinstance(data, dict)
        assert "results" in data
        assert len(data["results"]) == 3

    def test_round_trip_serialization(self, sample_trajectory_scores):
        """Test round-trip serialization (object -> dict -> object)."""
        from rl_finetuning import TrajectoryScore, TrajectoryGradingOutput

        # Create original object
        scores = [TrajectoryScore(**s) for s in sample_trajectory_scores]
        original = TrajectoryGradingOutput(results=scores)

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back to object
        restored = TrajectoryGradingOutput(**data)

        # Verify equality
        assert len(restored.results) == len(original.results)
        for orig, rest in zip(original.results, restored.results):
            assert orig.trajectory == rest.trajectory
            assert orig.score == rest.score
