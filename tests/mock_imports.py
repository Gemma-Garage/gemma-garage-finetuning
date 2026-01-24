"""
Mock imports for testing without ML dependencies.

This module must be imported BEFORE any project modules to properly
mock external dependencies like torch, unsloth, transformers, etc.
"""

import sys
from unittest.mock import MagicMock, PropertyMock


def setup_mocks():
    """
    Set up mock modules for ML libraries.
    Must be called before importing any project modules.
    """
    # Create sophisticated mock for torch
    torch_mock = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    torch_mock.cuda.is_bf16_supported = MagicMock(return_value=True)
    torch_mock.float16 = "float16"
    torch_mock.bfloat16 = "bfloat16"

    # Mock datasets
    datasets_mock = MagicMock()
    datasets_mock.load_dataset = MagicMock(return_value=MagicMock())
    datasets_mock.Dataset = MagicMock()
    datasets_mock.Dataset.from_list = MagicMock(return_value=MagicMock())

    # Mock transformers
    transformers_mock = MagicMock()
    transformers_mock.TrainerCallback = type('TrainerCallback', (), {
        'on_log': lambda self, *args, **kwargs: kwargs.get('control')
    })
    transformers_mock.TrainingArguments = MagicMock()
    transformers_mock.AutoModelForCausalLM = MagicMock()
    transformers_mock.AutoTokenizer = MagicMock()
    transformers_mock.BitsAndBytesConfig = MagicMock()
    transformers_mock.logging = MagicMock()
    transformers_mock.pipeline = MagicMock()
    transformers_mock.Gemma3ForCausalLM = MagicMock()

    # Mock trl
    trl_mock = MagicMock()
    trl_mock.SFTTrainer = MagicMock()
    trl_mock.SFTConfig = MagicMock()
    trl_mock.GRPOTrainer = MagicMock()
    trl_mock.GRPOConfig = MagicMock()

    # Mock peft
    peft_mock = MagicMock()
    peft_mock.LoraConfig = MagicMock()

    # Mock unsloth
    unsloth_mock = MagicMock()
    unsloth_mock.FastLanguageModel = MagicMock()
    unsloth_mock.FastLanguageModel.from_pretrained = MagicMock(
        return_value=(MagicMock(), MagicMock())
    )
    unsloth_mock.FastLanguageModel.get_peft_model = MagicMock(return_value=MagicMock())

    unsloth_chat_templates_mock = MagicMock()
    unsloth_chat_templates_mock.get_chat_template = MagicMock(return_value=MagicMock())

    # Mock google cloud
    google_mock = MagicMock()
    google_cloud_mock = MagicMock()

    # Storage mock
    storage_mock = MagicMock()
    storage_client_mock = MagicMock()
    storage_mock.Client = MagicMock(return_value=storage_client_mock)

    # Logging mock
    cloud_logging_mock = MagicMock()
    cloud_logging_client_mock = MagicMock()
    cloud_logging_mock.Client = MagicMock(return_value=cloud_logging_client_mock)
    mock_logger = MagicMock()
    cloud_logging_client_mock.logger = MagicMock(return_value=mock_logger)

    # litellm mock
    litellm_mock = MagicMock()
    litellm_mock.completion = MagicMock()

    # genai mock
    genai_mock = MagicMock()

    # Install all mocks
    mock_modules = {
        'torch': torch_mock,
        'torch.cuda': torch_mock.cuda,
        'torch.utils': MagicMock(),
        'torch.utils.data': MagicMock(),
        'datasets': datasets_mock,
        'transformers': transformers_mock,
        'trl': trl_mock,
        'peft': peft_mock,
        'unsloth': unsloth_mock,
        'unsloth.chat_templates': unsloth_chat_templates_mock,
        'google': google_mock,
        'google.cloud': google_cloud_mock,
        'google.cloud.storage': storage_mock,
        'google.cloud.logging': cloud_logging_mock,
        'google.generativeai': genai_mock,
        'litellm': litellm_mock,
    }

    for mod_name, mock_mod in mock_modules.items():
        sys.modules[mod_name] = mock_mod

    return mock_modules


# Auto-setup when this module is imported
_mocks = setup_mocks()
