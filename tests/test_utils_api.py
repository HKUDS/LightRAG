"""Tests for API utility functions"""

from unittest.mock import MagicMock, patch

import pytest

from lightrag.api.utils_api import (
    check_env_file,
    get_combined_auth_dependency,
    display_splash_screen,
)


class TestUtilityFunctions:
    """Test utility functions"""

    def test_check_env_file_exists(self):
        """Test check_env_file when .env exists"""
        with patch("os.path.exists", return_value=True):
            result = check_env_file()
            assert result is True

    def test_check_env_file_missing_non_interactive(self):
        """Test check_env_file when .env missing and non-interactive"""
        with patch("os.path.exists", return_value=False):
            with patch("sys.stdin.isatty", return_value=False):
                result = check_env_file()
                assert result is True

    def test_check_env_file_missing_interactive_yes(self):
        """Test check_env_file when .env missing and user says yes"""
        with patch("os.path.exists", return_value=False):
            with patch("sys.stdin.isatty", return_value=True):
                with patch("builtins.input", return_value="yes"):
                    result = check_env_file()
                    assert result is True

    def test_check_env_file_missing_interactive_no(self):
        """Test check_env_file when .env missing and user says no"""
        with patch("os.path.exists", return_value=False):
            with patch("sys.stdin.isatty", return_value=True):
                with patch("builtins.input", return_value="no"):
                    result = check_env_file()
                    assert result is False

    def test_get_combined_auth_dependency(self):
        """Test get_combined_auth_dependency function"""
        # Test without API key
        dependency = get_combined_auth_dependency()
        assert dependency is not None

        # Test with API key
        dependency = get_combined_auth_dependency("test_api_key")
        assert dependency is not None

    def test_display_splash_screen(self):
        """Test display_splash_screen function"""
        # Create a mock args object
        mock_args = MagicMock()
        mock_args.host = "localhost"
        mock_args.port = 8080
        mock_args.workers = 4

        # Test that function doesn't crash
        try:
            display_splash_screen(mock_args)
            assert True
        except Exception as e:
            pytest.fail(f"display_splash_screen raised an exception: {e}")


# Simple integration test
class TestUtilsAPIIntegration:
    """Test integration of utils_api functions"""

    def test_environment_check_workflow(self):
        """Test environment checking workflow"""
        # Test normal flow with .env file present
        with patch("os.path.exists", return_value=True):
            result = check_env_file()
            assert result is True

        # Test auth dependency creation
        auth_dep = get_combined_auth_dependency()
        assert auth_dep is not None
