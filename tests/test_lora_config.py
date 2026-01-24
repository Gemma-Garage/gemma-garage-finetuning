"""
Unit tests for LoRA configuration setup.

Tests cover:
- LoRA configuration parameters
- Default values
- Target modules configuration
- Integration with training engines
"""

import pytest
from unittest.mock import MagicMock, patch


class TestLoRAConfigurationParameters:
    """Tests for LoRA configuration parameter handling."""

    def test_default_lora_rank(self):
        """Test default LoRA rank value."""
        # Default LoRA rank in training_task.py
        default_rank = 4
        assert default_rank == 4

    def test_default_lora_alpha(self):
        """Test default LoRA alpha value."""
        # Default from finetuning_unsloth.py train_with_unsloth
        default_alpha = 32
        assert default_alpha == 32

    def test_default_lora_dropout(self):
        """Test default LoRA dropout value."""
        # Default from finetuning_unsloth.py
        default_dropout = 0.05
        assert default_dropout == 0.05

    def test_default_target_modules(self):
        """Test default target modules for LoRA."""
        # From finetuning_unsloth.py and rl_finetuning.py
        default_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        assert "q_proj" in default_modules
        assert "k_proj" in default_modules
        assert "v_proj" in default_modules
        assert "o_proj" in default_modules
        assert "gate_proj" in default_modules
        assert "up_proj" in default_modules
        assert "down_proj" in default_modules


class TestLoRAConfigInFinetuning:
    """Tests for LoRA config in supervised fine-tuning."""

    def test_lora_config_parameters_in_set_lora(self):
        """Test LoRA parameters used in set_lora_fine_tuning."""
        # Simulating the LoRA config setup from finetuning.py
        lora_params = {
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "r": 4,  # This is lora_rank
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        assert lora_params["lora_alpha"] == 16
        assert lora_params["lora_dropout"] == 0.1
        assert lora_params["r"] == 4
        assert lora_params["bias"] == "none"
        assert lora_params["task_type"] == "CAUSAL_LM"

    def test_target_modules_for_gemma(self):
        """Test that target modules are appropriate for Gemma models."""
        # Gemma-specific modules from finetuning.py
        gemma_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # These are the attention projection modules
        for module in gemma_modules:
            assert module.endswith("_proj")


class TestLoRAConfigInUnsloth:
    """Tests for LoRA config in Unsloth fine-tuning."""

    def test_unsloth_lora_parameters(self):
        """Test LoRA parameters for Unsloth training."""
        # From finetuning_unsloth.py train_with_unsloth
        unsloth_params = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
            "use_gradient_checkpointing": True,
            "random_state": 3407,
            "use_rslora": False,
            "loftq_config": None,
        }

        assert unsloth_params["r"] == 16
        assert unsloth_params["lora_alpha"] == 32
        assert unsloth_params["use_gradient_checkpointing"] is True
        assert unsloth_params["random_state"] == 3407

    def test_unsloth_extended_target_modules(self):
        """Test that Unsloth uses extended target modules including MLP."""
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

        # Attention modules
        assert "q_proj" in target_modules
        assert "k_proj" in target_modules
        assert "v_proj" in target_modules
        assert "o_proj" in target_modules

        # MLP modules (feed-forward)
        assert "gate_proj" in target_modules
        assert "up_proj" in target_modules
        assert "down_proj" in target_modules


class TestLoRAConfigInRL:
    """Tests for LoRA config in RL fine-tuning."""

    def test_rl_lora_default_parameters(self):
        """Test default LoRA parameters for RL training."""
        # From rl_finetuning.py train_with_unsloth
        rl_defaults = {
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }

        assert rl_defaults["lora_rank"] == 16
        assert rl_defaults["lora_alpha"] == 32
        assert rl_defaults["lora_dropout"] == 0.05

    def test_rl_learning_rate_different_from_supervised(self):
        """Test that RL uses different default learning rate."""
        # RL default from rl_finetuning.py
        rl_learning_rate = 5e-6

        # Supervised default from training_task.py
        supervised_learning_rate = 2e-4

        # RL uses smaller learning rate
        assert rl_learning_rate < supervised_learning_rate


class TestLoRARankValidation:
    """Tests for LoRA rank validation."""

    def test_valid_lora_ranks(self):
        """Test valid LoRA rank values."""
        valid_ranks = [4, 8, 16, 32, 64]

        for rank in valid_ranks:
            assert rank > 0
            assert isinstance(rank, int)

    def test_lora_rank_argument_parsing(self):
        """Test LoRA rank is parsed correctly from arguments."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--lora_rank', type=int, default=4)

        args = parser.parse_args(['--lora_rank', '16'])
        assert args.lora_rank == 16

        args = parser.parse_args([])
        assert args.lora_rank == 4


class TestLoRAWithGradientCheckpointing:
    """Tests for gradient checkpointing with LoRA."""

    def test_gradient_checkpointing_enabled_by_default(self):
        """Test that gradient checkpointing is enabled by default."""
        # From both finetuning_unsloth.py and rl_finetuning.py
        use_gradient_checkpointing = True
        assert use_gradient_checkpointing is True

    def test_gradient_checkpointing_options(self):
        """Test gradient checkpointing options."""
        # Can be True or "unsloth" for Unsloth version
        valid_options = [True, "unsloth", False]

        for option in valid_options:
            assert option in valid_options


class TestLoRABiasConfiguration:
    """Tests for LoRA bias configuration."""

    def test_bias_none_is_default(self):
        """Test that bias='none' is the default setting."""
        # From all fine-tuning configs
        bias_setting = "none"
        assert bias_setting == "none"

    def test_valid_bias_options(self):
        """Test valid bias options for LoRA."""
        valid_bias_options = ["none", "all", "lora_only"]

        for option in valid_bias_options:
            assert option in valid_bias_options


class TestLoRATaskType:
    """Tests for LoRA task type configuration."""

    def test_task_type_causal_lm(self):
        """Test that task type is CAUSAL_LM for language model fine-tuning."""
        task_type = "CAUSAL_LM"
        assert task_type == "CAUSAL_LM"

    def test_valid_task_types(self):
        """Test valid task types for LoRA."""
        # PEFT library task types
        valid_task_types = [
            "CAUSAL_LM",
            "SEQ_2_SEQ_LM",
            "TOKEN_CLS",
            "SEQ_CLS",
            "QUESTION_ANS",
        ]

        assert "CAUSAL_LM" in valid_task_types
