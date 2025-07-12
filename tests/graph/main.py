#!/usr/bin/env python
"""
Main test runner for organized graph storage tests
"""

import asyncio
import argparse
from ascii_colors import ASCIIColors

from .core.storage_setup import initialize_graph_test_storage, test_check_env_file
from .core.translation_engine import t
from .tests.basic import test_graph_basic
from .tests.advanced import test_graph_advanced
from .tests.batch import test_graph_batch_operations
from .tests.special_chars import test_graph_special_characters
from .tests.undirected import test_graph_undirected_property


async def run_basic_test():
    """Run basic graph storage test"""
    storage = await initialize_graph_test_storage()
    if storage is None:
        ASCIIColors.red(t("init_storage_failed"))
        return False

    try:
        result = await test_graph_basic(storage)
        return result
    finally:
        # Cleanup
        if hasattr(storage, "close"):
            await storage.close()
        if hasattr(storage, "_temp_dir"):
            from .core.storage_setup import cleanup_kuzu_test_environment

            cleanup_kuzu_test_environment(storage._temp_dir)


async def run_advanced_test():
    """Run advanced graph storage test"""
    storage = await initialize_graph_test_storage()
    if storage is None:
        ASCIIColors.red(t("init_storage_failed"))
        return False

    try:
        ASCIIColors.blue(t("starting_advanced_test"))
        result = await test_graph_advanced(storage)
        return result
    finally:
        # Cleanup
        if hasattr(storage, "close"):
            await storage.close()
        if hasattr(storage, "_temp_dir"):
            from .core.storage_setup import cleanup_kuzu_test_environment

            cleanup_kuzu_test_environment(storage._temp_dir)


async def run_batch_test():
    """Run batch operations graph storage test"""
    storage = await initialize_graph_test_storage()
    if storage is None:
        ASCIIColors.red(t("init_storage_failed"))
        return False

    try:
        ASCIIColors.blue(t("starting_batch_operations_test"))
        result = await test_graph_batch_operations(storage)
        return result
    finally:
        # Cleanup
        if hasattr(storage, "close"):
            await storage.close()
        if hasattr(storage, "_temp_dir"):
            from .core.storage_setup import cleanup_kuzu_test_environment

            cleanup_kuzu_test_environment(storage._temp_dir)


async def run_special_characters_test():
    """Run special characters graph storage test"""
    storage = await initialize_graph_test_storage()
    if storage is None:
        ASCIIColors.red(t("init_storage_failed"))
        return False

    try:
        ASCIIColors.blue(t("starting_special_character_test"))
        result = await test_graph_special_characters(storage)
        return result
    finally:
        # Cleanup
        if hasattr(storage, "close"):
            await storage.close()
        if hasattr(storage, "_temp_dir"):
            from .core.storage_setup import cleanup_kuzu_test_environment

            cleanup_kuzu_test_environment(storage._temp_dir)


async def run_undirected_test():
    """Run undirected graph property test"""
    storage = await initialize_graph_test_storage()
    if storage is None:
        ASCIIColors.red(t("init_storage_failed"))
        return False

    try:
        ASCIIColors.blue(t("starting_undirected_graph_test"))
        result = await test_graph_undirected_property(storage)
        return result
    finally:
        # Cleanup
        if hasattr(storage, "close"):
            await storage.close()
        if hasattr(storage, "_temp_dir"):
            from .core.storage_setup import cleanup_kuzu_test_environment

            cleanup_kuzu_test_environment(storage._temp_dir)


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Graph Storage Test Suite")
    parser.add_argument(
        "--test",
        choices=["basic", "advanced", "batch", "special", "undirected", "all"],
        default="basic",
        help="Test type to run",
    )
    parser.add_argument(
        "--language",
        choices=["chinese", "english"],
        default="chinese",
        help="Language for test output",
    )

    args = parser.parse_args()

    # Set language
    import os

    os.environ["TEST_LANGUAGE"] = args.language

    # Check environment
    if not test_check_env_file():
        return

    # Print header
    print(t("program_title"))

    # Run selected test
    if args.test == "basic":
        success = await run_basic_test()
        if success:
            ASCIIColors.green("✅ Basic test passed!")
        else:
            ASCIIColors.red("❌ Basic test failed!")
    elif args.test == "advanced":
        success = await run_advanced_test()
        if success:
            ASCIIColors.green("✅ Advanced test passed!")
        else:
            ASCIIColors.red("❌ Advanced test failed!")
    elif args.test == "batch":
        success = await run_batch_test()
        if success:
            ASCIIColors.green("✅ Batch operations test passed!")
        else:
            ASCIIColors.red("❌ Batch operations test failed!")
    elif args.test == "special":
        success = await run_special_characters_test()
        if success:
            ASCIIColors.green("✅ Special characters test passed!")
        else:
            ASCIIColors.red("❌ Special characters test failed!")
    elif args.test == "undirected":
        success = await run_undirected_test()
        if success:
            ASCIIColors.green("✅ Undirected graph property test passed!")
        else:
            ASCIIColors.red("❌ Undirected graph property test failed!")
    elif args.test == "all":
        # Run all available tests
        tests = [
            ("basic", run_basic_test),
            ("advanced", run_advanced_test),
            ("batch", run_batch_test),
            ("special", run_special_characters_test),
            ("undirected", run_undirected_test),
        ]

        all_passed = True
        for test_name, test_func in tests:
            ASCIIColors.blue(f"\n=== Running {test_name} test ===")
            try:
                success = await test_func()
                if success:
                    ASCIIColors.green(f"✅ {test_name.capitalize()} test passed!")
                else:
                    ASCIIColors.red(f"❌ {test_name.capitalize()} test failed!")
                    all_passed = False
            except Exception as e:
                ASCIIColors.red(
                    f"❌ {test_name.capitalize()} test failed with error: {e}"
                )
                all_passed = False

        if all_passed:
            ASCIIColors.green("\n🎉 All tests passed!")
        else:
            ASCIIColors.red("\n❌ Some tests failed!")
    else:
        ASCIIColors.yellow(
            f"Test type '{args.test}' not yet implemented in organized structure"
        )
        ASCIIColors.cyan(
            "Currently available: basic, advanced, batch, special, undirected, all"
        )


if __name__ == "__main__":
    asyncio.run(main())
