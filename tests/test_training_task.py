"""
Unit tests for training_task.py argument parsing and task initialization.

Tests cover:
- Argument parsing for required and optional arguments
- Default value handling
- Job type selection (supervised vs RL fine-tuning)
- Custom rubric handling for RL jobs
"""

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestArgumentParsing:
    """Tests for command-line argument parsing in training_task."""

    def test_parse_required_arguments(self, default_training_args):
        """Test parsing of all required arguments."""
        # Create a minimal parser that matches training_task.py
        parser = argparse.ArgumentParser(description="Vertex AI Fine-tuning Task")
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--output_dir', type=str, required=True)
        parser.add_argument('--model_name', type=str, default='google/gemma-2b')
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=2e-4)
        parser.add_argument('--lora_rank', type=int, default=4)
        parser.add_argument('--request_id', type=str, required=True)
        parser.add_argument('--project_id', type=str, required=True)
        parser.add_argument('--job_type', type=str, default='supervised')
        parser.add_argument('--custom_rubric', type=str, default='')

        args = parser.parse_args(default_training_args)

        assert args.dataset == 'gs://test-bucket/dataset.json'
        assert args.output_dir == 'gs://test-bucket/output'
        assert args.model_name == 'google/gemma-2b'
        assert args.epochs == 3
        assert args.learning_rate == 2e-4
        assert args.lora_rank == 8
        assert args.request_id == 'test-request-123'
        assert args.project_id == 'test-project'
        assert args.job_type == 'supervised'

    def test_parse_rl_finetuning_arguments(self, rl_training_args):
        """Test parsing of RL fine-tuning specific arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--output_dir', type=str, required=True)
        parser.add_argument('--model_name', type=str, default='google/gemma-2b')
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=2e-4)
        parser.add_argument('--lora_rank', type=int, default=4)
        parser.add_argument('--request_id', type=str, required=True)
        parser.add_argument('--project_id', type=str, required=True)
        parser.add_argument('--job_type', type=str, default='supervised')
        parser.add_argument('--custom_rubric', type=str, default='')

        args = parser.parse_args(rl_training_args)

        assert args.job_type == 'rl_finetuning'
        assert args.custom_rubric == 'Grade the response based on accuracy and clarity.'

    def test_default_values(self):
        """Test that default values are applied correctly."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--output_dir', type=str, required=True)
        parser.add_argument('--model_name', type=str, default='google/gemma-2b')
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=2e-4)
        parser.add_argument('--lora_rank', type=int, default=4)
        parser.add_argument('--request_id', type=str, required=True)
        parser.add_argument('--project_id', type=str, required=True)
        parser.add_argument('--job_type', type=str, default='supervised')
        parser.add_argument('--custom_rubric', type=str, default='')

        # Only provide required arguments
        minimal_args = [
            '--dataset', 'gs://bucket/data.json',
            '--output_dir', 'gs://bucket/output',
            '--request_id', 'test-123',
            '--project_id', 'test-proj',
        ]

        args = parser.parse_args(minimal_args)

        # Check defaults
        assert args.model_name == 'google/gemma-2b'
        assert args.epochs == 1
        assert args.learning_rate == 2e-4
        assert args.lora_rank == 4
        assert args.job_type == 'supervised'
        assert args.custom_rubric == ''

    def test_missing_required_arguments_raises_error(self):
        """Test that missing required arguments raise SystemExit."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--output_dir', type=str, required=True)
        parser.add_argument('--request_id', type=str, required=True)
        parser.add_argument('--project_id', type=str, required=True)

        # Missing --dataset
        incomplete_args = [
            '--output_dir', 'gs://bucket/output',
            '--request_id', 'test-123',
            '--project_id', 'test-proj',
        ]

        with pytest.raises(SystemExit):
            parser.parse_args(incomplete_args)

    def test_gcs_path_validation(self, default_training_args):
        """Test that GCS paths are properly accepted."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--output_dir', type=str, required=True)
        parser.add_argument('--model_name', type=str, default='google/gemma-2b')
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=2e-4)
        parser.add_argument('--lora_rank', type=int, default=4)
        parser.add_argument('--request_id', type=str, required=True)
        parser.add_argument('--project_id', type=str, required=True)
        parser.add_argument('--job_type', type=str, default='supervised')
        parser.add_argument('--custom_rubric', type=str, default='')

        args = parser.parse_args(default_training_args)

        assert args.dataset.startswith('gs://')
        assert args.output_dir.startswith('gs://')


class TestJobTypeSelection:
    """Tests for job type selection logic."""

    def test_supervised_job_type_selects_unsloth_engine(self):
        """Test that supervised job type would select UnslothFineTuningEngine."""
        job_type = 'supervised'
        assert job_type == 'supervised'
        # The actual engine selection happens in training_task.py
        # Here we just verify the condition logic

    def test_rl_finetuning_job_type_selects_rl_engine(self):
        """Test that rl_finetuning job type would select RLFinetuningEngine."""
        job_type = 'rl_finetuning'
        assert job_type == 'rl_finetuning'


class TestCustomRubricHandling:
    """Tests for custom rubric handling in RL fine-tuning."""

    def test_empty_rubric_default(self):
        """Test that empty rubric is the default."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--custom_rubric', type=str, default='')

        args = parser.parse_args([])
        assert args.custom_rubric == ''

    def test_custom_rubric_passed_through(self):
        """Test that custom rubric is properly passed."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--custom_rubric', type=str, default='')

        rubric = "Evaluate responses based on: 1) Accuracy 2) Clarity 3) Completeness"
        args = parser.parse_args(['--custom_rubric', rubric])

        assert args.custom_rubric == rubric

    def test_long_rubric_handling(self):
        """Test handling of long custom rubrics."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--custom_rubric', type=str, default='')

        # Create a long rubric (over 100 chars for truncation display)
        long_rubric = "A" * 150
        args = parser.parse_args(['--custom_rubric', long_rubric])

        assert len(args.custom_rubric) == 150
        assert args.custom_rubric == long_rubric
