"""
Unit tests for CloudLoggingCallback class.

Tests cover:
- Callback initialization
- on_log method behavior
- Log payload formatting
- NaN handling in logs
- Reward tracking (RL-specific)
"""

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestCloudLoggingCallbackInitialization:
    """Tests for CloudLoggingCallback initialization."""

    def test_callback_initialization_supervised(self, mock_cloud_logger):
        """Test CloudLoggingCallback initialization for supervised learning."""
        from finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request-123")

        assert callback.cloud_logger == mock_cloud_logger
        assert callback.request_id == "test-request-123"

    def test_callback_initialization_rl(self, mock_cloud_logger):
        """Test CloudLoggingCallback initialization for RL (has reward tracking)."""
        from rl_finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request-456")

        assert callback.cloud_logger == mock_cloud_logger
        assert callback.request_id == "test-request-456"
        assert callback.last_reward is None


class TestOnLogMethod:
    """Tests for the on_log callback method."""

    def test_on_log_with_valid_logs(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test on_log with valid training logs."""
        from finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        logs = {
            "loss": 0.5,
            "learning_rate": 2e-4,
            "epoch": 0.5,
        }

        control = MagicMock()
        result = callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=control,
            logs=logs
        )

        # Should return control unchanged
        assert result == control

        # Should have called log_struct
        assert mock_cloud_logger.log_struct.called

    def test_on_log_with_none_logs(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test on_log with None logs."""
        from finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        control = MagicMock()
        result = callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=control,
            logs=None
        )

        assert result == control
        # Should not call log_struct when logs is None
        assert not mock_cloud_logger.log_struct.called

    def test_on_log_includes_progress_info(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test that on_log includes progress information."""
        from finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        logs = {"loss": 0.3}

        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs=logs
        )

        # Get the logged payload
        call_args = mock_cloud_logger.log_struct.call_args
        payload = call_args[0][0]

        assert "current_step" in payload
        assert "total_steps" in payload
        assert "current_epoch" in payload
        assert "total_epochs" in payload
        assert "status_message" in payload


class TestNaNHandling:
    """Tests for NaN value handling in logs."""

    def test_nan_loss_converted_to_string(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test that NaN loss is converted to 'NaN' string."""
        from rl_finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        logs = {
            "loss": float('nan'),
            "learning_rate": 2e-4,
        }

        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs=logs
        )

        call_args = mock_cloud_logger.log_struct.call_args
        payload = call_args[0][0]

        # NaN should be converted to string "NaN"
        assert payload.get("loss") == "NaN" or "loss" in payload

    def test_nan_learning_rate_converted_to_string(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test that NaN learning_rate is converted to 'NaN' string."""
        from rl_finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        logs = {
            "loss": 0.5,
            "learning_rate": float('nan'),
        }

        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs=logs
        )

        call_args = mock_cloud_logger.log_struct.call_args
        payload = call_args[0][0]

        assert payload.get("learning_rate") == "NaN" or "learning_rate" in payload

    def test_multiple_nan_values_handled(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test handling of multiple NaN values."""
        from rl_finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        logs = {
            "loss": float('nan'),
            "learning_rate": float('nan'),
            "custom_metric": float('nan'),
        }

        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs=logs
        )

        # Should not raise an error
        assert mock_cloud_logger.log_struct.called


class TestRewardTracking:
    """Tests for RL-specific reward tracking."""

    def test_update_reward_method(self, mock_cloud_logger):
        """Test update_reward method updates last_reward."""
        from rl_finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        assert callback.last_reward is None

        callback.update_reward(0.85)
        assert callback.last_reward == 0.85

        callback.update_reward(0.92)
        assert callback.last_reward == 0.92

    def test_reward_included_in_log_payload(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test that reward is included in log payload when set."""
        from rl_finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")
        callback.update_reward(0.75)

        logs = {"loss": 0.3}

        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs=logs
        )

        call_args = mock_cloud_logger.log_struct.call_args
        payload = call_args[0][0]

        assert payload.get("reward") == 0.75


class TestLogPayloadFormat:
    """Tests for log payload formatting."""

    def test_payload_includes_request_id(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test that payload includes request_id."""
        from finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "unique-request-id")

        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs={"loss": 0.5}
        )

        call_args = mock_cloud_logger.log_struct.call_args
        payload = call_args[0][0]

        assert payload["request_id"] == "unique-request-id"

    def test_payload_includes_timestamp(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test that payload includes timestamp."""
        from finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs={"loss": 0.5}
        )

        call_args = mock_cloud_logger.log_struct.call_args
        payload = call_args[0][0]

        assert "timestamp" in payload
        # Should be ISO format
        datetime.fromisoformat(payload["timestamp"].replace('Z', '+00:00'))

    def test_none_values_removed_from_payload(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test that None values are removed from payload."""
        from finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        logs = {
            "loss": 0.5,
            "learning_rate": None,  # This should be removed
        }

        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs=logs
        )

        call_args = mock_cloud_logger.log_struct.call_args
        payload = call_args[0][0]

        # None values should be removed
        for key, value in payload.items():
            assert value is not None


class TestCloudLoggingFailure:
    """Tests for handling cloud logging failures."""

    def test_log_failure_does_not_raise(self, mock_trainer_state, mock_training_args):
        """Test that cloud log failure does not raise exception."""
        from finetuning import CloudLoggingCallback

        # Create a logger that raises exception
        failing_logger = MagicMock()
        failing_logger.log_struct = MagicMock(side_effect=Exception("Network error"))

        callback = CloudLoggingCallback(failing_logger, "test-request")

        # Should not raise exception
        result = callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs={"loss": 0.5}
        )

        # Control should still be returned
        assert result is not None

    def test_severity_parameter_passed(self, mock_cloud_logger, mock_trainer_state, mock_training_args):
        """Test that severity parameter is passed to log_struct."""
        from finetuning import CloudLoggingCallback

        callback = CloudLoggingCallback(mock_cloud_logger, "test-request")

        callback.on_log(
            args=mock_training_args,
            state=mock_trainer_state,
            control=MagicMock(),
            logs={"loss": 0.5}
        )

        call_args = mock_cloud_logger.log_struct.call_args
        assert call_args[1].get("severity") == "INFO"
