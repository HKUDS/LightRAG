#!/usr/bin/env python3
"""
Interactive CLI for LightRAG Graph Storage Test Suite
Allows users to select tests to run with bilingual support
"""

import asyncio
import sys
import os
from typing import List, Tuple, Optional
import argparse

# Add parent directory to path so we can import from tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ascii_colors import ASCIIColors
from tests.graph.core.storage_setup import (
    initialize_graph_test_storage,
    cleanup_kuzu_test_environment,
)
from tests.graph.core.translation_engine import t


# Import test functions
from tests.graph.tests.basic import test_graph_basic
from tests.graph.tests.advanced import test_graph_advanced
from tests.graph.tests.batch import test_graph_batch_operations
from tests.graph.tests.special_chars import test_graph_special_characters
from tests.graph.tests.undirected import test_graph_undirected_property


class TestRunner:
    """Interactive test runner with bilingual support"""

    def __init__(self):
        self.language = None  # Will be set by language selection
        self.test_functions = {}  # Will be populated after language selection

    def _initialize_test_functions(self):
        """Initialize test functions after language is selected"""
        self.test_functions = {
            "basic": {
                "func": test_graph_basic,
                "name": self._get_text("basic_test_name"),
                "description": self._get_text("basic_test_desc"),
            },
            "advanced": {
                "func": test_graph_advanced,
                "name": self._get_text("advanced_test_name"),
                "description": self._get_text("advanced_test_desc"),
            },
            "batch": {
                "func": test_graph_batch_operations,
                "name": self._get_text("batch_test_name"),
                "description": self._get_text("batch_test_desc"),
            },
            "special": {
                "func": test_graph_special_characters,
                "name": self._get_text("special_test_name"),
                "description": self._get_text("special_test_desc"),
            },
            "undirected": {
                "func": test_graph_undirected_property,
                "name": self._get_text("undirected_test_name"),
                "description": self._get_text("undirected_test_desc"),
            },
        }

    def _get_text(self, key: str) -> str:
        """Get translated text for a key"""
        translations = {
            "basic_test_name": ["基础测试", "Basic Test"],
            "basic_test_desc": [
                "节点插入、边创建、基本图操作",
                "Node insertion, edge creation, basic graph operations",
            ],
            "advanced_test_name": ["高级测试", "Advanced Test"],
            "advanced_test_desc": [
                "复杂图结构、多跳关系、高级查询",
                "Complex graph structures, multi-hop relationships, advanced queries",
            ],
            "batch_test_name": ["批量测试", "Batch Test"],
            "batch_test_desc": [
                "批量操作、事务处理、性能优化",
                "Bulk operations, transaction handling, performance optimization",
            ],
            "special_test_name": ["特殊字符测试", "Special Characters Test"],
            "special_test_desc": [
                "Unicode支持、特殊字符编码、国际化",
                "Unicode support, special character encoding, internationalization",
            ],
            "undirected_test_name": ["无向图测试", "Undirected Graph Test"],
            "undirected_test_desc": [
                "双向关系、无向图行为、一致性验证",
                "Bidirectional relationships, undirected graph behavior, consistency validation",
            ],
            "welcome": [
                "欢迎使用LightRAG图存储测试套件",
                "Welcome to LightRAG Graph Storage Test Suite",
            ],
            "select_language": [
                "请选择语言:",
                "Please select language:",
            ],
            "select_tests": [
                "请选择要运行的测试 (用逗号分隔多个选项，或输入 'all' 运行所有测试):",
                "Please select tests to run (comma-separated for multiple, or 'all' for all tests):",
            ],
            "available_tests": ["可用测试:", "Available tests:"],
            "invalid_selection": [
                "无效选择，请重试",
                "Invalid selection, please try again",
            ],
            "running_test": ["正在运行测试", "Running test"],
            "test_passed": ["测试通过", "Test passed"],
            "test_failed": ["测试失败", "Test failed"],
            "all_tests_passed": ["所有测试通过!", "All tests passed!"],
            "all_tests": ["所有测试", "All tests"],
            "some_tests_failed": ["部分测试失败", "Some tests failed"],
            "storage_init_failed": ["存储初始化失败", "Storage initialization failed"],
            "test_summary": ["测试摘要", "Test Summary"],
            "enter_choice": ["请输入选择", "Enter your choice"],
            "press_enter": ["按回车键继续...", "Press Enter to continue..."],
            "language_en": ["英语", "English"],
            "language_zh": ["中文", "Chinese"],
            "continue_question": [
                "是否继续运行其他测试? (y/n)",
                "Continue with other tests? (y/n)",
            ],
            "goodbye": ["再见!", "Goodbye!"],
            "error_occurred": ["发生错误", "Error occurred"],
            "select_storage": ["选择存储后端:", "Select storage backend:"],
            "networkx_storage": ["NetworkX存储 (默认)", "NetworkX Storage (default)"],
            "kuzu_storage": ["Kuzu数据库存储", "Kuzu Database Storage"],
            "neo4j_storage": ["Neo4j存储", "Neo4j Storage"],
            "mongodb_storage": ["MongoDB存储", "MongoDB Storage"],
            "storage_selection": ["存储后端选择", "Storage Backend Selection"],
        }

        if key in translations:
            return (
                translations[key][0]
                if self.language == "chinese"
                else translations[key][1]
            )
        return key

    def display_header(self):
        """Display the program header"""
        ASCIIColors.cyan("=" * 60)
        ASCIIColors.yellow(self._get_text("welcome"))
        ASCIIColors.cyan("=" * 60)
        print()

    def display_language_selection(self) -> str:
        """Display language selection menu in both languages"""
        print("=" * 60)
        print("🌐 Language Selection / 语言选择")
        print("=" * 60)
        print("1. English")
        print("2. 中文 (Chinese)")
        print()

        while True:
            try:
                choice = input("Enter your choice / 请输入选择 (1-2): ").strip()
                if choice == "1":
                    return "english"
                elif choice == "2":
                    return "chinese"
                else:
                    print("❌ Invalid selection, please try again / 无效选择，请重试")
            except KeyboardInterrupt:
                print("\nGoodbye! / 再见!")
                sys.exit(0)

    def display_storage_selection(self) -> str:
        """Display storage backend selection"""
        ASCIIColors.blue(self._get_text("storage_selection"))
        print(f"1. {self._get_text('networkx_storage')}")
        print(f"2. {self._get_text('kuzu_storage')}")
        print(f"3. {self._get_text('neo4j_storage')}")
        print(f"4. {self._get_text('mongodb_storage')}")
        print()

        while True:
            try:
                choice = input(f"{self._get_text('enter_choice')} (1-4): ").strip()
                if choice == "1":
                    return "NetworkXStorage"
                elif choice == "2":
                    return "KuzuDBStorage"
                elif choice == "3":
                    return "Neo4JStorage"
                elif choice == "4":
                    return "MongoGraphStorage"
                else:
                    ASCIIColors.red(self._get_text("invalid_selection"))
            except KeyboardInterrupt:
                print(f"\n{self._get_text('goodbye')}")
                sys.exit(0)

    def display_test_menu(self) -> List[str]:
        """Display test selection menu and get user choice"""
        ASCIIColors.blue(self._get_text("available_tests"))
        print()

        for i, (key, test_info) in enumerate(self.test_functions.items(), 1):
            ASCIIColors.green(f"{i}. {test_info['name']}")
            print(f"   {test_info['description']}")
            print()

        ASCIIColors.yellow(
            f"{len(self.test_functions) + 1}. {self._get_text('all_tests')}"
        )
        print()

        while True:
            try:
                ASCIIColors.blue(self._get_text("select_tests"))
                choice = input(f"{self._get_text('enter_choice')}: ").strip()

                if choice.lower() == "all":
                    return list(self.test_functions.keys())

                # Parse comma-separated choices
                selected_tests = []
                for c in choice.split(","):
                    c = c.strip()
                    if c.isdigit():
                        idx = int(c) - 1
                        if 0 <= idx < len(self.test_functions):
                            test_key = list(self.test_functions.keys())[idx]
                            selected_tests.append(test_key)
                        elif idx == len(self.test_functions):  # "All tests" option
                            return list(self.test_functions.keys())
                    else:
                        # Try to match test name directly
                        if c in self.test_functions:
                            selected_tests.append(c)

                if selected_tests:
                    return selected_tests
                else:
                    ASCIIColors.red(self._get_text("invalid_selection"))

            except KeyboardInterrupt:
                print(f"\n{self._get_text('goodbye')}")
                sys.exit(0)

    async def run_test(self, test_key: str, storage) -> bool:
        """Run a specific test"""
        if test_key not in self.test_functions:
            return False

        test_info = self.test_functions[test_key]
        ASCIIColors.blue(f"\n{self._get_text('running_test')}: {test_info['name']}")
        ASCIIColors.cyan("=" * 50)

        try:
            result = await test_info["func"](storage)
            if result:
                ASCIIColors.green(
                    f"✅ {test_info['name']} - {self._get_text('test_passed')}"
                )
            else:
                ASCIIColors.red(
                    f"❌ {test_info['name']} - {self._get_text('test_failed')}"
                )
            return result
        except Exception as e:
            ASCIIColors.red(
                f"❌ {test_info['name']} - {self._get_text('error_occurred')}: {e}"
            )
            return False

    async def run_selected_tests(self, selected_tests: List[str], storage_backend: str):
        """Run the selected tests"""
        # Set storage backend
        os.environ["LIGHTRAG_GRAPH_STORAGE"] = storage_backend

        # Initialize storage
        storage = await initialize_graph_test_storage()
        if storage is None:
            ASCIIColors.red(self._get_text("storage_init_failed"))
            return

        try:
            results = {}
            for test_key in selected_tests:
                success = await self.run_test(test_key, storage)
                results[test_key] = success

            # Display summary
            print("\n" + "=" * 60)
            ASCIIColors.yellow(self._get_text("test_summary"))
            print("=" * 60)

            passed = sum(1 for success in results.values() if success)
            total = len(results)

            for test_key, success in results.items():
                test_name = self.test_functions[test_key]["name"]
                status = (
                    self._get_text("test_passed")
                    if success
                    else self._get_text("test_failed")
                )
                color = ASCIIColors.green if success else ASCIIColors.red
                color(f"{'✅' if success else '❌'} {test_name}: {status}")

            print(f"\n{self._get_text('test_summary')}: {passed}/{total}")

            if passed == total:
                ASCIIColors.green(self._get_text("all_tests_passed"))
            else:
                ASCIIColors.red(self._get_text("some_tests_failed"))

        finally:
            # Cleanup
            if storage and hasattr(storage, "close"):
                await storage.close()
            if storage and hasattr(storage, "_temp_dir"):
                cleanup_kuzu_test_environment(storage._temp_dir)

    async def run_interactive(self):
        """Run the interactive test selection"""
        # Language selection (always shown)
        self.language = self.display_language_selection()
        os.environ["TEST_LANGUAGE"] = self.language

        # Initialize test functions after language is selected
        self._initialize_test_functions()

        # Display header
        self.display_header()

        # Storage backend selection
        storage_backend = self.display_storage_selection()

        while True:
            try:
                # Test selection
                selected_tests = self.display_test_menu()

                # Run tests
                await self.run_selected_tests(selected_tests, storage_backend)

                # Ask if user wants to continue
                print(f"\n{self._get_text('continue_question')}")
                continue_choice = input().strip().lower()

                if continue_choice not in ["y", "yes", "是", "是的"]:
                    break

            except KeyboardInterrupt:
                print(f"\n{self._get_text('goodbye')}")
                break

        print(f"\n{self._get_text('goodbye')}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Interactive LightRAG Graph Storage Test Suite"
    )
    parser.add_argument(
        "--language",
        choices=["english", "chinese"],
        help="Language for the interface (if not specified, user will be prompted)",
    )
    parser.add_argument(
        "--storage",
        choices=[
            "NetworkXStorage",
            "KuzuDBStorage",
            "Neo4JStorage",
            "MongoGraphStorage",
        ],
        help="Storage backend to use (if not specified, user will be prompted)",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["basic", "advanced", "batch", "special", "undirected", "all"],
        help="Tests to run (if not specified, user will be prompted)",
    )

    args = parser.parse_args()

    # Quick mode - if all parameters are provided, run without interaction
    if args.language and args.storage and args.tests:
        runner = TestRunner()
        runner.language = args.language
        os.environ["TEST_LANGUAGE"] = args.language
        runner._initialize_test_functions()

        selected_tests = (
            list(runner.test_functions.keys()) if "all" in args.tests else args.tests
        )
        asyncio.run(runner.run_selected_tests(selected_tests, args.storage))
    else:
        # Interactive mode
        runner = TestRunner()
        asyncio.run(runner.run_interactive())


if __name__ == "__main__":
    main()
