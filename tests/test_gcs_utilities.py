"""
Unit tests for GCS (Google Cloud Storage) utility functions.

Tests cover:
- upload_to_gcs function
- download_from_gcs method
- GCS path parsing
- Error handling
"""

import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest


class TestGCSPathParsing:
    """Tests for GCS path parsing logic."""

    def test_parse_gcs_path_with_bucket_and_path(self):
        """Test parsing GCS path with bucket and blob path."""
        gcs_path = "gs://my-bucket/path/to/file.json"

        # Simulate the parsing logic used in upload_to_gcs
        bucket_name, *blob_path = gcs_path.replace("gs://", "").split("/", 1)
        blob_path_prefix = blob_path[0] if blob_path else ""

        assert bucket_name == "my-bucket"
        assert blob_path_prefix == "path/to/file.json"

    def test_parse_gcs_path_bucket_only(self):
        """Test parsing GCS path with bucket only."""
        gcs_path = "gs://my-bucket"

        bucket_name, *blob_path = gcs_path.replace("gs://", "").split("/", 1)
        blob_path_prefix = blob_path[0] if blob_path else ""

        assert bucket_name == "my-bucket"
        assert blob_path_prefix == ""

    def test_parse_gcs_path_nested_path(self):
        """Test parsing GCS path with deeply nested path."""
        gcs_path = "gs://bucket/a/b/c/d/file.json"

        bucket_name, *blob_path = gcs_path.replace("gs://", "").split("/", 1)
        blob_path_prefix = blob_path[0] if blob_path else ""

        assert bucket_name == "bucket"
        assert blob_path_prefix == "a/b/c/d/file.json"


class TestUploadToGCS:
    """Tests for upload_to_gcs function."""

    @patch('finetuning_unsloth.storage.Client')
    def test_upload_single_file(self, mock_storage_client):
        """Test uploading a single file to GCS."""
        from finetuning_unsloth import upload_to_gcs

        # Create mock objects
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Create a temp directory with a single file
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test content")

            upload_to_gcs(tmp_dir, "gs://test-bucket/output")

            # Verify bucket was accessed
            mock_client.bucket.assert_called_with("test-bucket")

            # Verify blob was created and uploaded
            assert mock_bucket.blob.called
            assert mock_blob.upload_from_filename.called

    @patch('finetuning_unsloth.storage.Client')
    def test_upload_multiple_files(self, mock_storage_client):
        """Test uploading multiple files to GCS."""
        from finetuning_unsloth import upload_to_gcs

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Create a temp directory with multiple files
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i in range(3):
                test_file = os.path.join(tmp_dir, f"test_{i}.txt")
                with open(test_file, 'w') as f:
                    f.write(f"content {i}")

            upload_to_gcs(tmp_dir, "gs://test-bucket/output")

            # Should have uploaded 3 files
            assert mock_blob.upload_from_filename.call_count == 3

    @patch('finetuning_unsloth.storage.Client')
    def test_upload_nested_directory(self, mock_storage_client):
        """Test uploading nested directory structure to GCS."""
        from finetuning_unsloth import upload_to_gcs

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Create a temp directory with nested structure
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = os.path.join(tmp_dir, "subdir")
            os.makedirs(nested_dir)

            # File in root
            with open(os.path.join(tmp_dir, "root.txt"), 'w') as f:
                f.write("root content")

            # File in nested dir
            with open(os.path.join(nested_dir, "nested.txt"), 'w') as f:
                f.write("nested content")

            upload_to_gcs(tmp_dir, "gs://test-bucket/output")

            # Should have uploaded 2 files
            assert mock_blob.upload_from_filename.call_count == 2


class TestDownloadFromGCS:
    """Tests for download_from_gcs method."""

    @patch('finetuning_unsloth.storage.Client')
    @patch('finetuning_unsloth.cloud_logging.Client')
    def test_download_file_from_gcs(self, mock_logging_client, mock_storage_client):
        """Test downloading a file from GCS."""
        # Setup mocks
        mock_storage = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_logger = MagicMock()

        mock_storage_client.return_value = mock_storage
        mock_storage.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_logging_client.return_value.logger.return_value = mock_logger

        from finetuning_unsloth import UnslothFineTuningEngine

        # Create engine (this will use mocked clients)
        engine = UnslothFineTuningEngine(
            model_name="test-model",
            request_id="test-123",
            project_id="test-project"
        )

        # Test download
        with tempfile.TemporaryDirectory() as tmp_dir:
            gcs_path = "gs://test-bucket/path/to/file.json"

            # Mock the download
            def mock_download(filename):
                with open(filename, 'w') as f:
                    f.write('{"test": "data"}')

            mock_blob.download_to_filename = mock_download

            local_path = engine.download_from_gcs(gcs_path, tmp_dir)

            # Verify path parsing
            mock_storage.bucket.assert_called_with("test-bucket")
            mock_bucket.blob.assert_called_with("path/to/file.json")

            # Verify local path
            assert local_path.endswith("file.json")
            assert os.path.dirname(local_path) == tmp_dir

    @patch('finetuning_unsloth.storage.Client')
    @patch('finetuning_unsloth.cloud_logging.Client')
    def test_download_creates_temp_dir_if_none(self, mock_logging_client, mock_storage_client):
        """Test that download creates temp dir if local_dir is None."""
        mock_storage = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_logger = MagicMock()

        mock_storage_client.return_value = mock_storage
        mock_storage.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_logging_client.return_value.logger.return_value = mock_logger

        from finetuning_unsloth import UnslothFineTuningEngine

        engine = UnslothFineTuningEngine(
            model_name="test-model",
            request_id="test-123",
            project_id="test-project"
        )

        def mock_download(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                f.write('test')

        mock_blob.download_to_filename = mock_download

        local_path = engine.download_from_gcs("gs://bucket/file.json", local_dir=None)

        # Should have created a temp directory
        assert local_path is not None
        assert "file.json" in local_path


class TestGCSPathValidation:
    """Tests for GCS path validation."""

    def test_gcs_path_starts_with_gs(self):
        """Test that valid GCS paths start with gs://."""
        valid_paths = [
            "gs://bucket/file.json",
            "gs://my-bucket/path/to/data.json",
            "gs://bucket-name-123/output/",
        ]

        for path in valid_paths:
            assert path.startswith("gs://")

    def test_detect_non_gcs_path(self):
        """Test detection of non-GCS paths."""
        non_gcs_paths = [
            "/local/path/to/file.json",
            "./relative/path.json",
            "http://example.com/file.json",
            "s3://bucket/file.json",
        ]

        for path in non_gcs_paths:
            assert not path.startswith("gs://")


class TestUploadToGCSInRL:
    """Tests for upload_to_gcs in rl_finetuning module."""

    @patch('rl_finetuning.storage.Client')
    def test_rl_upload_function_exists(self, mock_storage_client):
        """Test that upload_to_gcs exists in rl_finetuning module."""
        from rl_finetuning import upload_to_gcs

        assert callable(upload_to_gcs)

    @patch('rl_finetuning.storage.Client')
    def test_rl_upload_same_behavior(self, mock_storage_client):
        """Test that RL upload has same behavior as supervised."""
        from rl_finetuning import upload_to_gcs

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "model.bin")
            with open(test_file, 'w') as f:
                f.write("model data")

            upload_to_gcs(tmp_dir, "gs://bucket/rl-output")

            mock_client.bucket.assert_called_with("bucket")
            assert mock_blob.upload_from_filename.called
