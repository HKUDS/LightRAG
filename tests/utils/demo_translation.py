#!/usr/bin/env python
"""
Demo script to show English translation feature for LightRAG test
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set language to English
os.environ["TEST_LANGUAGE"] = "english"

# Import and run just the translation demo
from tests.test_graph_storage import t


def demo_translations():
    """Demonstrate the translation feature"""

    print("=== LightRAG Graph Storage Test - Translation Demo ===\n")

    print("English translations:")
    print(f"- Artificial Intelligence: {t('artificial_intelligence')}")
    print(f"- Machine Learning: {t('machine_learning')}")
    print(f"- Deep Learning: {t('deep_learning')}")
    print(f"- Description: {t('ai_desc')}")
    print(f"- Insert node: {t('insert_node')}")
    print(f"- Test completed: {t('basic_test_complete')}")

    print("\nChinese translations (for comparison):")
    # Temporarily switch to Chinese
    import tests.test_graph_storage

    tests.test_graph_storage.LANGUAGE = "chinese"

    print(f"- Artificial Intelligence: {t('artificial_intelligence')}")
    print(f"- Machine Learning: {t('machine_learning')}")
    print(f"- Deep Learning: {t('deep_learning')}")
    print(f"- Description: {t('ai_desc')[:50]}...")
    print(f"- Insert node: {t('insert_node')}")
    print(f"- Test completed: {t('basic_test_complete')}")

    print("\n=== Translation Feature Working! ===")
    print("\nTo use in the actual test:")
    print("1. Set TEST_LANGUAGE=english in .env file")
    print("2. Or use: python test_graph_storage.py --language english")
    print("3. Or use: TEST_LANGUAGE=english python test_graph_storage.py")


if __name__ == "__main__":
    demo_translations()
