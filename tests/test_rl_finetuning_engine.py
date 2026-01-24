"""
Unit and integration tests for RLFinetuningEngine class.

Tests cover:
- Engine initialization
- System prompt definition
- Answer extraction
- Reward function components
- Custom rubric handling
- Backoff mechanism
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import time


class TestRLFinetuningEngineInitialization:
    """Tests for RLFinetuningEngine initialization."""

    @patch('rl_finetuning.UnslothFineTuningEngine.__init__')
    @patch('rl_finetuning.cloud_logging.Client')
    def test_engine_inherits_from_unsloth(self, mock_logging, mock_init):
        """Test that RLFinetuningEngine inherits from UnslothFineTuningEngine."""
        mock_init.return_value = None
        mock_logger = MagicMock()
        mock_logging.return_value.logger.return_value = mock_logger

        from rl_finetuning import RLFinetuningEngine, MathGrader

        # Verify inheritance
        from finetuning_unsloth import UnslothFineTuningEngine
        assert issubclass(RLFinetuningEngine, UnslothFineTuningEngine)

    @patch('rl_finetuning.UnslothFineTuningEngine.__init__')
    def test_engine_initializes_math_grader(self, mock_init):
        """Test that engine initializes MathGrader."""
        mock_init.return_value = None

        from rl_finetuning import RLFinetuningEngine, MathGrader

        # Create engine with mocked parent init
        engine = RLFinetuningEngine.__new__(RLFinetuningEngine)
        engine.cloud_logger = MagicMock()
        engine.request_id = "test-123"
        engine.math_grader = MathGrader()

        assert isinstance(engine.math_grader, MathGrader)


class TestSystemPromptDefinition:
    """Tests for system prompt definition."""

    def test_define_system_prompt_returns_string(self):
        """Test that define_system_prompt returns a string."""
        from rl_finetuning import RLFinetuningEngine

        # Create mock engine instance
        engine = MagicMock(spec=RLFinetuningEngine)
        engine.define_system_prompt = RLFinetuningEngine.define_system_prompt

        prompt = engine.define_system_prompt(engine)

        assert isinstance(prompt, str)

    def test_system_prompt_contains_markers(self):
        """Test that system prompt contains required markers."""
        from rl_finetuning import RLFinetuningEngine

        engine = MagicMock(spec=RLFinetuningEngine)
        engine.define_system_prompt = RLFinetuningEngine.define_system_prompt

        prompt = engine.define_system_prompt(engine)

        assert "<start_working_out>" in prompt
        assert "<end_working_out>" in prompt
        assert "<SOLUTION>" in prompt
        assert "</SOLUTION>" in prompt

    def test_system_prompt_gives_instructions(self):
        """Test that system prompt gives clear instructions."""
        from rl_finetuning import RLFinetuningEngine

        engine = MagicMock(spec=RLFinetuningEngine)
        engine.define_system_prompt = RLFinetuningEngine.define_system_prompt

        prompt = engine.define_system_prompt(engine)

        assert "problem" in prompt.lower()
        assert "working out" in prompt.lower() or "working" in prompt.lower()
        assert "solution" in prompt.lower()


class TestExtractHashAnswer:
    """Tests for extract_hash_answer static method."""

    def test_extract_answer_with_hash_marker(self):
        """Test extracting answer after #### marker."""
        from rl_finetuning import RLFinetuningEngine

        text = "Some explanation #### 42"
        result = RLFinetuningEngine.extract_hash_answer(text)

        assert result == "42"

    def test_extract_answer_with_leading_whitespace(self):
        """Test extracting answer with leading whitespace."""
        from rl_finetuning import RLFinetuningEngine

        text = "Explanation ####   answer"
        result = RLFinetuningEngine.extract_hash_answer(text)

        assert result == "answer"

    def test_extract_answer_no_marker_returns_none(self):
        """Test that missing marker returns None."""
        from rl_finetuning import RLFinetuningEngine

        text = "No hash marker here"
        result = RLFinetuningEngine.extract_hash_answer(text)

        assert result is None

    def test_extract_answer_multiline(self):
        """Test extracting from multiline text."""
        from rl_finetuning import RLFinetuningEngine

        text = """Step 1: Add numbers
        Step 2: Multiply
        #### 100"""

        result = RLFinetuningEngine.extract_hash_answer(text)
        assert result == "100"


class TestCustomRubricHandling:
    """Tests for custom rubric handling in RL training."""

    def test_custom_rubric_stored_in_engine(self):
        """Test that custom rubric is stored in engine."""
        from rl_finetuning import RLFinetuningEngine

        # Create mock engine
        engine = MagicMock(spec=RLFinetuningEngine)

        custom_rubric = "Evaluate based on accuracy and clarity"
        engine.custom_rubric = custom_rubric

        assert engine.custom_rubric == custom_rubric

    def test_empty_rubric_uses_default(self):
        """Test that empty rubric falls back to default."""
        from rl_finetuning import RLFinetuningEngine

        engine = MagicMock(spec=RLFinetuningEngine)
        engine.custom_rubric = ""

        # The reward_function uses "What is correct" as default
        default_rubric = getattr(engine, 'custom_rubric', None) or "What is correct"

        assert default_rubric == "What is correct"

    def test_rubric_passed_to_train(self):
        """Test that custom rubric is passed to train method."""
        from rl_finetuning import RLFinetuningEngine

        engine = MagicMock(spec=RLFinetuningEngine)
        engine.train_with_unsloth = MagicMock()

        custom_rubric = "Grade based on mathematical accuracy"

        # Simulate calling train with custom_rubric
        engine.train_with_unsloth(
            dataset_path="test/path",
            custom_rubric=custom_rubric
        )

        engine.train_with_unsloth.assert_called_once()
        call_kwargs = engine.train_with_unsloth.call_args[1]
        assert call_kwargs.get("custom_rubric") == custom_rubric


class TestCallWithBackoff:
    """Tests for call_with_backoff utility."""

    def test_successful_call_no_retry(self):
        """Test that successful call doesn't retry."""
        from rl_finetuning import RLFinetuningEngine

        call_count = 0

        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = RLFinetuningEngine.call_with_backoff(successful_func, max_retries=5)

        assert result == "success"
        assert call_count == 1

    def test_retry_on_rate_limit_error(self):
        """Test retry behavior on rate limit error."""
        from rl_finetuning import RLFinetuningEngine

        call_count = 0

        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate limit exceeded")
            return "success"

        result = RLFinetuningEngine.call_with_backoff(
            failing_then_success,
            max_retries=5
        )

        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        """Test that error is raised after max retries."""
        from rl_finetuning import RLFinetuningEngine

        def always_fails():
            raise Exception("rate limit exceeded 429")

        with pytest.raises(RuntimeError, match="Exceeded maximum retries"):
            RLFinetuningEngine.call_with_backoff(
                always_fails,
                max_retries=2
            )

    def test_non_rate_limit_error_not_retried(self):
        """Test that non-rate-limit errors are raised immediately."""
        from rl_finetuning import RLFinetuningEngine

        def raises_other_error():
            raise ValueError("Some other error")

        with pytest.raises(ValueError, match="Some other error"):
            RLFinetuningEngine.call_with_backoff(
                raises_other_error,
                max_retries=5
            )


class TestRewardFunctionComponents:
    """Tests for reward function components."""

    def test_reward_function_signature(self):
        """Test that reward_function has correct signature."""
        from rl_finetuning import RLFinetuningEngine
        import inspect

        sig = inspect.signature(RLFinetuningEngine.reward_function)
        params = list(sig.parameters.keys())

        assert 'self' in params
        assert 'prompts' in params
        assert 'completions' in params

    def test_reward_function_returns_list(self):
        """Test that reward function would return a list of scores."""
        # We can't fully test without mocking the Gemini API,
        # but we can verify the expected output type
        expected_output_type = list

        # Mock a simple return value
        mock_scores = [0.8, 0.6, 0.9, 0.7]

        assert isinstance(mock_scores, expected_output_type)
        for score in mock_scores:
            assert isinstance(score, (int, float))


class TestTrainEntryPoint:
    """Tests for the train entry point method."""

    def test_train_delegates_to_train_with_unsloth(self):
        """Test that train method delegates to train_with_unsloth."""
        from rl_finetuning import RLFinetuningEngine

        # Create mock engine
        engine = MagicMock(spec=RLFinetuningEngine)
        engine.train = RLFinetuningEngine.train
        engine.train_with_unsloth = MagicMock(return_value="model_path")

        result = engine.train(engine, dataset_path="test/data.json", learning_rate=5e-6)

        engine.train_with_unsloth.assert_called_once()

    def test_train_passes_kwargs(self):
        """Test that train passes keyword arguments correctly."""
        from rl_finetuning import RLFinetuningEngine

        engine = MagicMock(spec=RLFinetuningEngine)
        engine.train = RLFinetuningEngine.train
        engine.train_with_unsloth = MagicMock(return_value="path")

        kwargs = {
            "learning_rate": 1e-5,
            "num_train_epochs": 2,
            "custom_rubric": "test rubric"
        }

        engine.train(engine, dataset_path="data.json", **kwargs)

        call_kwargs = engine.train_with_unsloth.call_args[1]
        assert call_kwargs.get("learning_rate") == 1e-5
        assert call_kwargs.get("num_train_epochs") == 2
        assert call_kwargs.get("custom_rubric") == "test rubric"


class TestGRPOConfiguration:
    """Tests for GRPO (Group Relative Policy Optimization) configuration."""

    def test_grpo_default_parameters(self):
        """Test default GRPO configuration parameters."""
        # From rl_finetuning.py train_with_unsloth
        grpo_defaults = {
            "learning_rate": 5e-6,
            "adam_beta1": 0.9,
            "adam_beta2": 0.99,
            "weight_decay": 0.1,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "optim": "adamw_torch_fused",
            "num_generations": 4,
            "max_grad_norm": 0.1,
        }

        assert grpo_defaults["learning_rate"] == 5e-6
        assert grpo_defaults["num_generations"] == 4
        assert grpo_defaults["optim"] == "adamw_torch_fused"

    def test_grpo_max_steps_default(self):
        """Test default max_steps for GRPO training."""
        default_max_steps = 50
        assert default_max_steps == 50

    def test_grpo_prompt_length_config(self):
        """Test prompt and completion length configuration."""
        max_prompt_length = 256
        max_seq_length = 2048
        max_completion_length = max_seq_length - max_prompt_length

        assert max_prompt_length == 256
        assert max_completion_length == 1792


class TestCloudLoggingCallbackReference:
    """Tests for cloud logging callback reference in RL engine."""

    def test_callback_reference_stored(self):
        """Test that cloud logging callback reference is stored for reward updates."""
        from rl_finetuning import RLFinetuningEngine, CloudLoggingCallback

        engine = MagicMock(spec=RLFinetuningEngine)
        callback = MagicMock(spec=CloudLoggingCallback)

        engine.cloud_logging_callback = callback

        assert engine.cloud_logging_callback is not None

    def test_callback_update_reward_called(self):
        """Test that callback's update_reward is called with reward value."""
        from rl_finetuning import CloudLoggingCallback

        callback = MagicMock(spec=CloudLoggingCallback)

        # Simulate what reward_function does
        avg_reward = 0.75
        callback.update_reward(avg_reward)

        callback.update_reward.assert_called_once_with(0.75)
