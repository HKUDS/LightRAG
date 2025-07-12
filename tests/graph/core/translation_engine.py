"""
Enhanced translation engine for multilingual test support
"""

import inspect
import os
from typing import Dict, List, Union, Any, Optional


# Language configuration
def get_language():
    """Get current language setting"""
    return os.getenv("TEST_LANGUAGE", "chinese").lower()


LANGUAGE = get_language()
if LANGUAGE not in ["english", "chinese"]:
    LANGUAGE = "chinese"

language_to_index = {
    "chinese": 0,
    "english": 1,
}


def t_enhanced(key: str, translations_dict: Optional[Dict[str, Any]] = None) -> str:
    """
    Enhanced translation function that automatically detects the calling test function
    by walking up the call stack until it finds a function with 'test' in its name.

    Args:
        key: Translation key to look up
        translations_dict: Optional translations dictionary (will be imported if not provided)

    Returns:
        Translated string for the current language
    """
    if translations_dict is None:
        # Import here to avoid circular imports
        try:
            from ..translations import get_all_translations

            translations_dict = get_all_translations()
        except ImportError:
            # Fallback to empty dict if translations not available yet
            translations_dict = {}

    # Ensure we have a valid dictionary
    if not isinstance(translations_dict, dict):
        translations_dict = {}

    # Get the calling function name using inspect
    frame = inspect.currentframe()
    try:
        current_frame = frame.f_back if frame else None
        test_function_name = None

        # Walk up the call stack to find the first function with 'test' in its name
        while current_frame:
            function_name = current_frame.f_code.co_name
            if "test" in function_name.lower():
                test_function_name = function_name
                break
            current_frame = current_frame.f_back

        # If we found a test function, look for translations in that section first
        if test_function_name and test_function_name in translations_dict:
            test_translations = translations_dict[test_function_name]
            if isinstance(test_translations, dict) and key in test_translations:
                value = test_translations[key]
                if isinstance(value, list):
                    idx = language_to_index.get(get_language(), 0)
                    return value[idx]
                return value

        # Fall back to common section
        common_translations = translations_dict.get("common", {})
        if isinstance(common_translations, dict) and key in common_translations:
            value = common_translations[key]
            if isinstance(value, list):
                idx = language_to_index.get(get_language(), 0)
                return value[idx]
            return value

        # If not found anywhere, return the key as string
        return str(key)

    finally:
        del frame


# Legacy function for backward compatibility
def t(key: str) -> str:
    """Legacy translation function - redirects to t_enhanced"""
    return t_enhanced(key)
