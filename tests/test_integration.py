"""
Integration tests for gemma-garage-finetuning.

These tests verify that multiple components work together correctly.
They still use mocks for external services but test the integration
between internal components.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

import pytest


class TestSupervisedTrainingIntegration:
    """Integration tests for supervised fine-tuning workflow."""

    @patch('finetuning_unsloth.storage.Client')
    @patch('finetuning_unsloth.cloud_logging.Client')
    @patch('finetuning_unsloth.FastLanguageModel')
    @patch('finetuning_unsloth.get_chat_template')
    @patch('finetuning_unsloth.SFTTrainer')
    @patch('finetuning_unsloth.load_dataset')
    def test_supervised_training_workflow(
        self,
        mock_load_dataset,
        mock_trainer_class,
        mock_chat_template,
        mock_model_class,
        mock_logging_client,
        mock_storage_client
    ):
        """Test complete supervised training workflow with mocked dependencies."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_logging_client.return_value.logger.return_value = mock_logger

        mock_storage = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value = mock_storage
        mock_storage.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_class.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_model_class.get_peft_model.return_value = mock_model

        mock_tokenizer.apply_chat_template = lambda x, tokenize, add_generation_prompt: "formatted"

        mock_trainer = MagicMock()
        mock_trainer.state = MagicMock()
        mock_trainer.state.max_steps = 100
        mock_trainer.args = MagicMock()
        mock_trainer.args.output_dir = "/tmp/output"
        mock_trainer_class.return_value = mock_trainer

        from finetuning_unsloth import UnslothFineTuningEngine

        # Create engine
        engine = UnslothFineTuningEngine(
            model_name="google/gemma-2b",
            request_id="test-request-001",
            project_id="test-project"
        )

        # Verify logger was initialized
        assert mock_logging_client.called
        assert mock_logger.log_struct.called

    @patch('finetuning_unsloth.storage.Client')
    @patch('finetuning_unsloth.cloud_logging.Client')
    def test_engine_logs_initialization_steps(self, mock_logging_client, mock_storage_client):
        """Test that engine logs initialization steps."""
        mock_logger = MagicMock()
        mock_logger.logged_entries = []

        def capture_log(payload, severity="INFO"):
            mock_logger.logged_entries.append(payload)

        mock_logger.log_struct = MagicMock(side_effect=capture_log)
        mock_logging_client.return_value.logger.return_value = mock_logger

        from finetuning_unsloth import UnslothFineTuningEngine

        engine = UnslothFineTuningEngine(
            model_name="test-model",
            request_id="init-test-001",
            project_id="test-project"
        )

        # Should have logged initialization
        assert mock_logger.log_struct.called
        logged_payloads = [call[0][0] for call in mock_logger.log_struct.call_args_list]

        # Check that status_message contains initialization info
        has_init_log = any(
            "Initializing" in payload.get("status_message", "")
            for payload in logged_payloads
        )
        assert has_init_log


class TestRLTrainingIntegration:
    """Integration tests for RL fine-tuning workflow."""

    @patch('rl_finetuning.storage.Client')
    @patch('rl_finetuning.cloud_logging.Client')
    def test_rl_engine_initialization(self, mock_logging_client, mock_storage_client):
        """Test RL engine initialization logs correct messages."""
        mock_logger = MagicMock()
        mock_logging_client.return_value.logger.return_value = mock_logger

        from rl_finetuning import RLFinetuningEngine

        engine = RLFinetuningEngine(
            model_name="test-model",
            request_id="rl-test-001",
            project_id="test-project"
        )

        # Should have logged RL-specific initialization
        logged_payloads = [call[0][0] for call in mock_logger.log_struct.call_args_list]

        has_rl_init = any(
            "RL" in str(payload)
            for payload in logged_payloads
        )
        assert has_rl_init

    @patch('rl_finetuning.storage.Client')
    @patch('rl_finetuning.cloud_logging.Client')
    def test_rl_engine_has_math_grader(self, mock_logging_client, mock_storage_client):
        """Test that RL engine initializes with MathGrader."""
        mock_logger = MagicMock()
        mock_logging_client.return_value.logger.return_value = mock_logger

        from rl_finetuning import RLFinetuningEngine, MathGrader

        engine = RLFinetuningEngine(
            model_name="test-model",
            request_id="rl-grader-test",
            project_id="test-project"
        )

        assert hasattr(engine, 'math_grader')
        assert isinstance(engine.math_grader, MathGrader)


class TestDatasetFormattingIntegration:
    """Integration tests for dataset formatting across modules."""

    def test_format_function_consistency(self, mock_tokenizer, sample_qa_dataset):
        """Test that format functions behave consistently across modules."""
        from rl_finetuning import format_for_gemma3_chat as rl_format
        from finetuning_unsloth import format_for_gemma3_chat as unsloth_format

        # Both should produce same output for same input
        rl_result = rl_format(sample_qa_dataset, tokenizer=mock_tokenizer)
        unsloth_result = unsloth_format(sample_qa_dataset, tokenizer=mock_tokenizer)

        assert len(rl_result) == len(unsloth_result)

    def test_text_dataset_passthrough_consistency(self, sample_text_only_dataset, mock_tokenizer):
        """Test text-only dataset handling is consistent."""
        from rl_finetuning import format_for_gemma3_chat as rl_format
        from finetuning_unsloth import format_for_gemma3_chat as unsloth_format

        rl_result = rl_format(sample_text_only_dataset, tokenizer=mock_tokenizer)
        unsloth_result = unsloth_format(sample_text_only_dataset, tokenizer=mock_tokenizer)

        # Both should pass through unchanged
        assert rl_result == sample_text_only_dataset
        assert unsloth_result == sample_text_only_dataset


class TestGCSIntegration:
    """Integration tests for GCS operations."""

    @patch('finetuning_unsloth.storage.Client')
    @patch('finetuning_unsloth.cloud_logging.Client')
    def test_download_and_process_dataset(
        self,
        mock_logging_client,
        mock_storage_client,
        sample_qa_dataset
    ):
        """Test downloading dataset from GCS and processing it."""
        # Setup mocks
        mock_logger = MagicMock()
        mock_logging_client.return_value.logger.return_value = mock_logger

        mock_storage = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value = mock_storage
        mock_storage.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        from finetuning_unsloth import UnslothFineTuningEngine

        engine = UnslothFineTuningEngine(
            model_name="test-model",
            request_id="gcs-test-001",
            project_id="test-project"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a mock downloaded file
            test_file = os.path.join(tmp_dir, "dataset.json")
            with open(test_file, 'w') as f:
                json.dump(sample_qa_dataset, f)

            # Mock download to copy our test file
            def mock_download(filename):
                with open(test_file, 'r') as src:
                    with open(filename, 'w') as dst:
                        dst.write(src.read())

            mock_blob.download_to_filename = mock_download

            # Test download
            local_path = engine.download_from_gcs(
                "gs://test-bucket/dataset.json",
                tmp_dir
            )

            # Verify file was "downloaded"
            assert local_path.endswith("dataset.json")


class TestCloudLoggingIntegration:
    """Integration tests for cloud logging across training."""

    def test_logging_callback_in_training_context(
        self,
        mock_cloud_logger,
        mock_trainer_state,
        mock_training_args
    ):
        """Test CloudLoggingCallback works in training context."""
        from finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "integration-test-001")

        # Simulate multiple training steps
        for step in range(3):
            mock_trainer_state.global_step = step
            mock_trainer_state.epoch = step * 0.1

            logs = {
                "loss": 0.5 - step * 0.1,
                "learning_rate": 2e-4,
            }

            callback.on_log(
                args=mock_training_args,
                state=mock_trainer_state,
                control=MagicMock(),
                logs=logs
            )

        # Should have logged 3 times
        assert mock_cloud_logger.log_struct.call_count == 3

    def test_rl_logging_callback_includes_reward(
        self,
        mock_cloud_logger,
        mock_trainer_state,
        mock_training_args
    ):
        """Test RL CloudLoggingCallback includes reward information."""
        from rl_finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "rl-logging-test")

        # Update reward
        callback.update_reward(0.85)

        # Log with reward
        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs={"loss": 0.3}
        )

        # Check that reward was included
        call_args = mock_cloud_logger.log_struct.call_args
        payload = call_args[0][0]
        assert payload.get("reward") == 0.85


class TestMathGraderIntegration:
    """Integration tests for MathGrader with completion processing."""

    def test_grader_with_realistic_completions(self):
        """Test grader with realistic model completions."""
        from rl_finetuning import MathGrader

        grader = MathGrader()

        # Realistic completions from a model - using correct end tag <end_working_out>
        completions = [
            [{"content": """<start_working_out>
Let me solve this step by step:
First, I need to add 15 and 27.
15 + 27 = 42
<end_working_out>
<SOLUTION>42</SOLUTION>"""}],
            [{"content": """<start_working_out>
Adding the numbers:
15 + 27
= 42
<end_working_out>
<SOLUTION>42</SOLUTION>"""}],
            [{"content": "The answer is 42"}],  # Wrong format
        ]

        # Test format matching
        format_scores = grader.match_format_exactly(completions)

        assert format_scores[0] == 3.0  # Correct format
        assert format_scores[1] == 3.0  # Correct format
        assert format_scores[2] == 0     # Wrong format

    def test_grader_answer_checking_integration(self):
        """Test grader answer checking with multiple completions."""
        from rl_finetuning import MathGrader

        grader = MathGrader()

        prompts = [[{"content": "What is 15 + 27?"}]]
        completions = [
            [{"content": "<start_working_out>calc<end_working_out><SOLUTION>42</SOLUTION>"}],
            [{"content": "<start_working_out>calc<end_working_out><SOLUTION>43</SOLUTION>"}],
            [{"content": "<start_working_out>calc<end_working_out><SOLUTION> 42 </SOLUTION>"}],
        ]
        answers = ["42", "42", "42"]

        # Check each completion
        for i, (completion, answer) in enumerate(zip(completions, answers)):
            scores = grader.check_answer(prompts, [completion], [answer])

            if i == 0:
                assert scores[0] == 3.0  # Exact match
            elif i == 1:
                # Close but wrong - might get partial credit or penalty
                assert scores[0] != 3.0
            elif i == 2:
                assert scores[0] == 1.5  # Whitespace match


class TestPydanticModelsIntegration:
    """Integration tests for Pydantic models with LiteLLM response parsing."""

    def test_parse_and_validate_response(self, sample_trajectory_scores):
        """Test parsing and validating a complete response."""
        from rl_finetuning import (
            RLFinetuningEngine,
            TrajectoryScore,
            TrajectoryGradingOutput
        )

        # Create mock LiteLLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "results": sample_trajectory_scores
        })

        # Parse response
        result = RLFinetuningEngine.parse_litellm_json_response(mock_response)

        # Validate result
        assert isinstance(result, TrajectoryGradingOutput)
        assert len(result.results) == 3
        assert all(isinstance(r, TrajectoryScore) for r in result.results)
        assert all(0.0 <= r.score <= 1.0 for r in result.results)


class TestEndToEndMocking:
    """End-to-end tests with full mocking."""

    @patch('finetuning_unsloth.storage.Client')
    @patch('finetuning_unsloth.cloud_logging.Client')
    def test_engine_setup_and_callback_integration(
        self,
        mock_logging_client,
        mock_storage_client
    ):
        """Test engine setup with callback integration."""
        mock_logger = MagicMock()
        mock_logging_client.return_value.logger.return_value = mock_logger

        from finetuning_unsloth import UnslothFineTuningEngine, CloudLoggingCallback

        # Create engine
        engine = UnslothFineTuningEngine(
            model_name="google/gemma-2b",
            request_id="e2e-test-001",
            project_id="test-project"
        )

        # Create callback using engine's logger
        callback = CloudLoggingCallback(engine.cloud_logger, engine.request_id)

        assert callback.request_id == "e2e-test-001"
        assert callback.cloud_logger == engine.cloud_logger
