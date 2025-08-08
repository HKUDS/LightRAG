"""Tests for API configuration module"""

import os
from unittest.mock import patch

import pytest

from lightrag.api.config import (
    get_ollama_host,
    get_default_host,
    parse_args,
    update_uvicorn_mode_config,
    global_args,
    ollama_server_infos,
    get_env_value,
)


class TestConfigFunctions:
    """Test configuration utility functions"""

    def test_get_ollama_host(self):
        """Test get_ollama_host function"""
        # Test default behavior
        result = get_ollama_host()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_default_host(self):
        """Test get_default_host function"""
        # Test with different binding types
        host = get_default_host("ollama")
        assert isinstance(host, str)

        host = get_default_host("openai")
        assert isinstance(host, str)

    def test_get_env_value(self):
        """Test get_env_value function"""
        # Test with existing environment variable
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = get_env_value("TEST_VAR", "default")
            assert result == "test_value"

        # Test with default value
        result = get_env_value("NONEXISTENT_VAR", "default_value")
        assert result == "default_value"

    def test_parse_args(self):
        """Test parse_args function"""
        # Mock sys.argv to prevent actual argument parsing during tests
        with patch("sys.argv", ["lightrag-server"]):
            args = parse_args()
            assert args is not None
            # Check that args has expected attributes
            assert hasattr(args, "host")
            assert hasattr(args, "port")

    def test_update_uvicorn_mode_config(self):
        """Test update_uvicorn_mode_config function"""
        # This function updates global config, test it doesn't crash
        try:
            update_uvicorn_mode_config()
            # If no exception, test passes
            assert True
        except Exception as e:
            pytest.fail(f"update_uvicorn_mode_config raised an exception: {e}")


class TestGlobalConfiguration:
    """Test global configuration objects"""

    def test_global_args_exists(self):
        """Test that global_args is available"""
        assert global_args is not None
        # Check for common attributes
        assert hasattr(global_args, "host") or hasattr(global_args, "port")

    def test_ollama_server_infos_exists(self):
        """Test that ollama_server_infos is available"""
        assert ollama_server_infos is not None
        # Check that it has expected attributes
        assert hasattr(ollama_server_infos, "_lightrag_name") or hasattr(
            ollama_server_infos, "_lightrag_tag"
        )


class TestConfigIntegration:
    """Integration tests for configuration system"""

    def test_config_with_environment_variables(self):
        """Test configuration with environment variables"""
        env_vars = {
            "HOST": "test_host",
            "PORT": "8080",
            "WORKING_DIR": "/test/dir",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Test that environment variables can be read
            host = get_env_value("HOST", "default_host")
            assert host == "test_host"

            port = get_env_value("PORT", "9621")
            assert port == "8080"
