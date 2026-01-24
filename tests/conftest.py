"""
Pytest configuration and fixtures for gemma-garage-finetuning tests.

This module provides:
- Mock fixtures for external dependencies (GCS, HuggingFace, torch, etc.)
- Sample test data fixtures
- Common test utilities
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ============================================================================
# Module-level mocks for ML libraries
# These MUST be set up before importing any project modules
# ============================================================================

# Import our mock setup module - this installs all mocks into sys.modules
# Need to add tests directory to path first
sys.path.insert(0, os.path.dirname(__file__))
from mock_imports import setup_mocks


# ============================================================================
# Mock fixtures for external services
# ============================================================================

@pytest.fixture
def mock_cloud_logger():
    """Create a mock cloud logger that captures log entries."""
    logger = MagicMock()
    logger.logged_entries = []

    def capture_log(payload, severity="INFO"):
        logger.logged_entries.append({
            "payload": payload,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    logger.log_struct = MagicMock(side_effect=capture_log)
    return logger


@pytest.fixture
def mock_cloud_logging_client(mock_cloud_logger):
    """Create a mock Google Cloud Logging client."""
    client = MagicMock()
    client.logger.return_value = mock_cloud_logger
    return client


@pytest.fixture
def mock_storage_client():
    """Create a mock GCS storage client."""
    client = MagicMock()
    bucket = MagicMock()
    blob = MagicMock()

    client.bucket.return_value = bucket
    bucket.blob.return_value = blob
    blob.download_to_filename = MagicMock()
    blob.upload_from_filename = MagicMock()

    return client


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with apply_chat_template method."""
    tokenizer = MagicMock()

    def mock_apply_chat_template(conversation, tokenize=False, add_generation_prompt=False):
        # Simulate Gemma 3 chat template output
        parts = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<bos><start_of_turn>system\n{content}<end_of_turn>")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        return "".join(parts)

    tokenizer.apply_chat_template = mock_apply_chat_template
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.save_pretrained = MagicMock()

    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model object."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.use_cache = True
    model.save_pretrained = MagicMock()
    return model


@pytest.fixture
def mock_trainer_state():
    """Create a mock trainer state."""
    state = MagicMock()
    state.global_step = 10
    state.max_steps = 100
    state.epoch = 0.5
    state.num_train_epochs = 3
    return state


@pytest.fixture
def mock_training_args():
    """Create mock training arguments."""
    args = MagicMock()
    args.output_dir = "/tmp/test_output"
    args.num_train_epochs = 3
    return args


# ============================================================================
# Test data fixtures
# ============================================================================

@pytest.fixture
def sample_qa_dataset():
    """Provide sample QA pairs dataset."""
    return {
        "summary": "Test math dataset for fine-tuning",
        "qa_pairs": [
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What is 3 * 4?", "answer": "12"},
            {"question": "What is 100 / 5?", "answer": "20"},
        ]
    }


@pytest.fixture
def sample_text_only_dataset():
    """Provide sample text-only dataset."""
    return [
        {"text": "This is the first example text."},
        {"text": "This is the second example text."},
        {"text": "This is the third example text."},
    ]


@pytest.fixture
def sample_messages_dataset():
    """Provide sample messages-format dataset."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Goodbye"},
                {"role": "assistant", "content": "See you later!"}
            ]
        }
    ]


@pytest.fixture
def temp_dataset_file(sample_qa_dataset):
    """Create a temporary dataset file and return its path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_qa_dataset, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_text_dataset_file(sample_text_only_dataset):
    """Create a temporary text-only dataset file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_text_only_dataset, f)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# Math grading test fixtures
# ============================================================================

@pytest.fixture
def sample_math_completions_correct():
    """Provide sample completions with correct math format and answer."""
    return [
        [{"content": "<start_working_out>Let me solve: 2 + 2 = 4<end_working_out><SOLUTION>4</SOLUTION>"}]
    ]


@pytest.fixture
def sample_math_completions_wrong_format():
    """Provide sample completions with incorrect format."""
    return [
        [{"content": "The answer is 4"}]
    ]


@pytest.fixture
def sample_math_completions_wrong_answer():
    """Provide sample completions with correct format but wrong answer."""
    return [
        [{"content": "<start_working_out>2 + 2 = 5<end_working_out><SOLUTION>5</SOLUTION>"}]
    ]


@pytest.fixture
def sample_prompts():
    """Provide sample prompts for math grading."""
    return [[{"content": "What is 2 + 2?"}]]


# ============================================================================
# Pydantic model fixtures
# ============================================================================

@pytest.fixture
def sample_trajectory_scores():
    """Provide sample trajectory score data."""
    return [
        {"trajectory": "Response 1", "score": 0.85},
        {"trajectory": "Response 2", "score": 0.72},
        {"trajectory": "Response 3", "score": 0.91},
    ]


# ============================================================================
# Command line argument fixtures
# ============================================================================

@pytest.fixture
def default_training_args():
    """Provide default training task arguments."""
    return [
        '--dataset', 'gs://test-bucket/dataset.json',
        '--output_dir', 'gs://test-bucket/output',
        '--model_name', 'google/gemma-2b',
        '--epochs', '3',
        '--learning_rate', '2e-4',
        '--lora_rank', '8',
        '--request_id', 'test-request-123',
        '--project_id', 'test-project',
    ]


@pytest.fixture
def rl_training_args(default_training_args):
    """Provide RL fine-tuning task arguments."""
    return default_training_args + [
        '--job_type', 'rl_finetuning',
        '--custom_rubric', 'Grade the response based on accuracy and clarity.',
    ]


# ============================================================================
# Utility functions
# ============================================================================

def create_mock_training_logs(loss=0.5, learning_rate=2e-4, **extra):
    """Create mock training logs dictionary."""
    logs = {
        "loss": loss,
        "learning_rate": learning_rate,
        "epoch": 0.5,
    }
    logs.update(extra)
    return logs
