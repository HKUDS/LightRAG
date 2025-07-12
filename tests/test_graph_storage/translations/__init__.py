"""
Translation utilities for multilingual support
"""

from .common import COMMON_TRANSLATIONS
from .basic_test import BASIC_TEST_TRANSLATIONS
from .advanced_test import ADVANCED_TEST_TRANSLATIONS
from .batch_test import BATCH_TEST_TRANSLATIONS
from .special_char_test import SPECIAL_CHAR_TEST_TRANSLATIONS
from .undirected_test import UNDIRECTED_TEST_TRANSLATIONS
from .utility import UTILITY_TRANSLATIONS


def get_all_translations():
    """
    Combine all translations into a single dictionary organized by test function
    """
    return {
        "common": COMMON_TRANSLATIONS,
        "test_graph_basic": BASIC_TEST_TRANSLATIONS,
        "test_graph_advanced": ADVANCED_TEST_TRANSLATIONS,
        "test_graph_batch_operations": BATCH_TEST_TRANSLATIONS,
        "test_graph_special_characters": SPECIAL_CHAR_TEST_TRANSLATIONS,
        "test_graph_undirected_property": UNDIRECTED_TEST_TRANSLATIONS,
        "test_check_env_file": UTILITY_TRANSLATIONS,
        "setup_kuzu_test_environment": UTILITY_TRANSLATIONS,
        "cleanup_kuzu_test_environment": UTILITY_TRANSLATIONS,
        "initialize_graph_test_storage": UTILITY_TRANSLATIONS,
        # Additional test translations will be added here as we create them
    }
