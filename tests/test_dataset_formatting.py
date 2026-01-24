"""
Unit tests for dataset loading and formatting functions.

Tests cover:
- format_for_gemma3_chat function behavior
- QA pairs dataset formatting
- Text-only dataset passthrough
- List of QA dicts formatting
- Error handling for unsupported formats
"""

import pytest
from unittest.mock import MagicMock


class TestFormatForGemma3Chat:
    """Tests for the format_for_gemma3_chat function."""

    def test_text_only_dataset_passthrough(self, sample_text_only_dataset, mock_tokenizer):
        """Test that text-only datasets pass through unchanged."""
        # Import the function - we need to handle the import carefully
        # since it may fail if ML libraries aren't available
        from rl_finetuning import format_for_gemma3_chat

        result = format_for_gemma3_chat(sample_text_only_dataset, tokenizer=mock_tokenizer)

        assert result == sample_text_only_dataset
        assert len(result) == 3
        for item in result:
            assert 'text' in item
            assert len(item) == 1

    def test_qa_pairs_dataset_formatting(self, sample_qa_dataset, mock_tokenizer):
        """Test formatting of QA pairs dataset with tokenizer."""
        from rl_finetuning import format_for_gemma3_chat

        result = format_for_gemma3_chat(sample_qa_dataset, tokenizer=mock_tokenizer)

        # Should have same number of items as qa_pairs
        assert len(result) == len(sample_qa_dataset['qa_pairs'])

        # Each result should have 'text' key
        for item in result:
            assert 'text' in item
            # Text should not start with <bos> (it gets stripped)
            assert not item['text'].startswith('<bos>')

    def test_qa_pairs_without_tokenizer_raises_error(self, sample_qa_dataset):
        """Test that QA pairs without tokenizer raises ValueError."""
        from rl_finetuning import format_for_gemma3_chat

        with pytest.raises(ValueError, match="Tokenizer with apply_chat_template required"):
            format_for_gemma3_chat(sample_qa_dataset, tokenizer=None)

    def test_list_of_qa_dicts_formatting(self, mock_tokenizer):
        """Test formatting of a list of QA dicts (not wrapped in 'qa_pairs')."""
        from rl_finetuning import format_for_gemma3_chat

        qa_list = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]

        result = format_for_gemma3_chat(qa_list, tokenizer=mock_tokenizer)

        assert len(result) == 2
        for item in result:
            assert 'text' in item

    def test_list_of_qa_dicts_without_tokenizer_raises_error(self):
        """Test that list of QA dicts without tokenizer raises ValueError."""
        from rl_finetuning import format_for_gemma3_chat

        qa_list = [
            {"question": "Q1?", "answer": "A1"},
        ]

        with pytest.raises(ValueError, match="Tokenizer with apply_chat_template required"):
            format_for_gemma3_chat(qa_list, tokenizer=None)

    def test_unsupported_format_raises_error(self, mock_tokenizer):
        """Test that unsupported data formats raise ValueError."""
        from rl_finetuning import format_for_gemma3_chat

        # Unsupported format: dict without qa_pairs
        bad_data = {"some_key": "some_value"}

        with pytest.raises(ValueError, match="Input data must be"):
            format_for_gemma3_chat(bad_data, tokenizer=mock_tokenizer)

    def test_unsupported_list_format_raises_error(self, mock_tokenizer):
        """Test that unsupported list formats raise ValueError."""
        from rl_finetuning import format_for_gemma3_chat

        # List with mixed keys
        bad_list = [
            {"foo": "bar"},
            {"baz": "qux"},
        ]

        with pytest.raises(ValueError, match="Input data must be"):
            format_for_gemma3_chat(bad_list, tokenizer=mock_tokenizer)

    def test_custom_system_prompt(self, mock_tokenizer):
        """Test that custom system prompt is applied."""
        from rl_finetuning import format_for_gemma3_chat

        qa_list = [{"question": "Q?", "answer": "A"}]
        custom_prompt = "You are a math tutor."

        result = format_for_gemma3_chat(qa_list, tokenizer=mock_tokenizer, system_prompt=custom_prompt)

        assert len(result) == 1
        # The custom prompt should be in the formatted text
        assert custom_prompt in result[0]['text']

    def test_bos_token_stripping(self, mock_tokenizer):
        """Test that <bos> token is stripped from formatted output."""
        from rl_finetuning import format_for_gemma3_chat

        # Modify mock to return text with <bos>
        def mock_template_with_bos(conversation, tokenize=False, add_generation_prompt=False):
            return "<bos>formatted text here"

        mock_tokenizer.apply_chat_template = mock_template_with_bos

        qa_list = [{"question": "Q?", "answer": "A"}]
        result = format_for_gemma3_chat(qa_list, tokenizer=mock_tokenizer)

        assert result[0]['text'] == "formatted text here"
        assert not result[0]['text'].startswith('<bos>')


class TestFormatForGemma3ChatInFinetuningUnsloth:
    """Tests for format_for_gemma3_chat in finetuning_unsloth.py"""

    def test_function_exists_in_both_modules(self):
        """Test that format_for_gemma3_chat exists in both modules."""
        from rl_finetuning import format_for_gemma3_chat as rl_format
        from finetuning_unsloth import format_for_gemma3_chat as unsloth_format

        # Both should be callable
        assert callable(rl_format)
        assert callable(unsloth_format)


class TestDatasetEdgeCases:
    """Tests for edge cases in dataset formatting."""

    def test_empty_qa_pairs_list(self, mock_tokenizer):
        """Test handling of empty QA pairs list."""
        from rl_finetuning import format_for_gemma3_chat

        data = {"qa_pairs": []}
        result = format_for_gemma3_chat(data, tokenizer=mock_tokenizer)

        assert result == []

    def test_empty_text_list(self, mock_tokenizer):
        """Test handling of empty text list."""
        from rl_finetuning import format_for_gemma3_chat

        # Empty list doesn't match any validation patterns in the function
        # Since it's an empty list and all() returns True for empty iterables,
        # the text-only check (all items have 'text' key) passes vacuously
        # and the function returns the empty list unchanged
        data = []
        result = format_for_gemma3_chat(data, tokenizer=mock_tokenizer)
        # The function treats empty list as valid text-only dataset (vacuously true)
        assert result == []

    def test_qa_with_extra_fields_ignored(self, mock_tokenizer):
        """Test that extra fields in QA pairs are ignored."""
        from rl_finetuning import format_for_gemma3_chat

        data = {
            "qa_pairs": [
                {
                    "question": "Q?",
                    "answer": "A",
                    "extra_field": "should be ignored",
                    "metadata": {"key": "value"}
                }
            ]
        }

        # Should succeed without error
        result = format_for_gemma3_chat(data, tokenizer=mock_tokenizer)
        assert len(result) == 1

    def test_unicode_content_handling(self, mock_tokenizer):
        """Test handling of unicode content in questions and answers."""
        from rl_finetuning import format_for_gemma3_chat

        data = {
            "qa_pairs": [
                {"question": "What is 2 + 2?", "answer": "4"},
                {"question": "Qu'est-ce que c'est?", "answer": "C'est un test."},
            ]
        }

        result = format_for_gemma3_chat(data, tokenizer=mock_tokenizer)
        assert len(result) == 2

    def test_multiline_content(self, mock_tokenizer):
        """Test handling of multiline questions and answers."""
        from rl_finetuning import format_for_gemma3_chat

        data = {
            "qa_pairs": [
                {
                    "question": "Solve this:\nLine 1\nLine 2",
                    "answer": "Answer:\nPart 1\nPart 2"
                }
            ]
        }

        result = format_for_gemma3_chat(data, tokenizer=mock_tokenizer)
        assert len(result) == 1
