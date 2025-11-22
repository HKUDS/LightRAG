#!/usr/bin/env python3
"""
Integration test script for LightRAG with production setup.

This script tests:
- Document indexing with C++ code repository
- Query operations (naive, local, global, hybrid)
- API endpoints (insert, query, graph retrieval)
- Integration with Redis, Neo4j, and Milvus storage backends
"""

import asyncio
import json
import os
import sys
import logging
from pathlib import Path
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Integration test runner for LightRAG."""

    def __init__(self, base_url: str = "http://localhost:9621"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)
        self.test_results = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
        self.test_results.append(
            {"test": test_name, "passed": passed, "message": message}
        )

    async def wait_for_server(self, max_retries: int = 30, retry_delay: int = 2):
        """Wait for LightRAG server to be ready."""
        logger.info("Waiting for LightRAG server to be ready...")

        for i in range(max_retries):
            try:
                response = await self.client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info("✅ LightRAG server is ready!")
                    return True
            except Exception as e:
                logger.debug(f"Attempt {i+1}/{max_retries}: Server not ready yet - {e}")

            await asyncio.sleep(retry_delay)

        logger.error("❌ Server failed to become ready in time")
        return False

    async def test_health_endpoint(self):
        """Test health check endpoint."""
        test_name = "Health Check"
        try:
            response = await self.client.get(f"{self.base_url}/health")
            passed = response.status_code == 200
            self.log_result(test_name, passed, f"Status: {response.status_code}")
            return passed
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
            return False

    async def test_insert_text(self, text: str, description: str = ""):
        """Test document insertion via API."""
        test_name = f"Insert Document{' - ' + description if description else ''}"
        try:
            response = await self.client.post(
                f"{self.base_url}/documents/text",
                json={"text": text, "description": description},
            )
            passed = response.status_code == 200
            self.log_result(test_name, passed, f"Status: {response.status_code}")
            return passed
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
            return False

    async def test_insert_file(self, file_path: Path, retry_count: int = 2):
        """Test file insertion via API with retry logic and fallback to text endpoint."""
        test_name = f"Insert File - {file_path.name}"

        # Check if this is a header file that should use text endpoint
        use_text_endpoint = file_path.suffix in [".h", ".hpp", ".hh"]

        for attempt in range(retry_count + 1):
            try:
                if use_text_endpoint:
                    # Use text insertion endpoint for header files
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    response = await self.client.post(
                        f"{self.base_url}/documents/text",
                        json={"text": content, "file_source": file_path.name},
                    )
                else:
                    # Use file upload endpoint for other files
                    with open(file_path, "rb") as f:
                        files = {"file": (file_path.name, f, "text/plain")}
                        response = await self.client.post(
                            f"{self.base_url}/documents/upload", files=files
                        )

                if response.status_code == 200:
                    self.log_result(test_name, True, f"Status: {response.status_code}")
                    return True
                elif response.status_code == 400:
                    # Check if it's unsupported file type error
                    try:
                        error_detail = response.json()
                        error_msg = error_detail.get("detail", "")
                        if (
                            "Unsupported file type" in error_msg
                            and not use_text_endpoint
                        ):
                            # Fallback to text endpoint
                            logger.info(
                                f"File type not supported for upload, trying text endpoint for {file_path.name}"
                            )
                            use_text_endpoint = True
                            continue
                    except (json.JSONDecodeError, ValueError, KeyError):
                        pass

                    self.log_result(test_name, False, f"Status: {response.status_code}")
                    return False
                elif response.status_code == 500:
                    # Try to get error details
                    try:
                        error_detail = response.json()
                        error_msg = error_detail.get("detail", "Unknown error")
                    except (json.JSONDecodeError, ValueError, KeyError):
                        error_msg = (
                            response.text[:200] if response.text else "No error details"
                        )

                    if attempt < retry_count:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {file_path.name}: {error_msg}. Retrying..."
                        )
                        await asyncio.sleep(2)  # Wait before retry
                        continue
                    else:
                        self.log_result(
                            test_name,
                            False,
                            f"Status: {response.status_code}, Error: {error_msg}",
                        )
                        return False
                else:
                    self.log_result(test_name, False, f"Status: {response.status_code}")
                    return False

            except Exception as e:
                if attempt < retry_count:
                    logger.warning(
                        f"Attempt {attempt + 1} exception for {file_path.name}: {e}. Retrying..."
                    )
                    await asyncio.sleep(2)
                    continue
                else:
                    self.log_result(test_name, False, f"Error: {e}")
                    return False

        return False

    async def test_query(self, query: str, mode: str = "hybrid"):
        """Test query endpoint."""
        test_name = f"Query ({mode} mode)"
        try:
            response = await self.client.post(
                f"{self.base_url}/query",
                json={"query": query, "mode": mode, "stream": False},
            )
            passed = response.status_code == 200

            if passed:
                result = response.json()
                response_text = result.get("response", "")
                logger.info(f"Query response preview: {response_text[:200]}...")

            self.log_result(test_name, passed, f"Status: {response.status_code}")
            return passed
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
            return False

    async def test_query_with_data(self, query: str, mode: str = "hybrid"):
        """Test query/data endpoint that returns structured data."""
        test_name = f"Query Data ({mode} mode)"
        try:
            response = await self.client.post(
                f"{self.base_url}/query/data",
                json={"query": query, "mode": mode, "top_k": 10},
            )
            passed = response.status_code == 200

            if passed:
                result = response.json()
                # Validate response structure
                has_data = "data" in result
                has_metadata = "metadata" in result
                if not (has_data and has_metadata):
                    passed = False
                    self.log_result(
                        test_name, passed, "Missing required fields in response"
                    )
                else:
                    data = result.get("data", {})
                    entities_count = len(data.get("entities", []))
                    relations_count = len(data.get("relationships", []))
                    chunks_count = len(data.get("chunks", []))
                    logger.info(
                        f"Retrieved: {entities_count} entities, {relations_count} relations, {chunks_count} chunks"
                    )
                    self.log_result(
                        test_name, passed, f"Status: {response.status_code}"
                    )
            else:
                self.log_result(test_name, passed, f"Status: {response.status_code}")

            return passed
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
            return False

    async def test_graph_data(self):
        """Test graph data retrieval endpoint."""
        test_name = "Graph Data Retrieval"
        try:
            response = await self.client.get(f"{self.base_url}/graph/label/list")
            passed = response.status_code == 200

            if passed:
                result = response.json()
                # Result is a list of labels
                if isinstance(result, list):
                    logger.info(f"Graph contains {len(result)} unique labels")
                else:
                    logger.info(f"Graph data: {result}")

            self.log_result(test_name, passed, f"Status: {response.status_code}")
            return passed
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
            return False

    async def run_all_tests(self, cpp_repo_path: Path):
        """Run all integration tests."""
        logger.info("=" * 80)
        logger.info("Starting LightRAG Integration Tests")
        logger.info("=" * 80)

        # Wait for server to be ready
        if not await self.wait_for_server():
            logger.error("Server not ready. Aborting tests.")
            return False

        # Test 1: Health check
        await self.test_health_endpoint()

        # Test 2: Index C++ files
        logger.info("\n--- Testing Document Indexing ---")
        cpp_files = list(cpp_repo_path.glob("**/*.cpp")) + list(
            cpp_repo_path.glob("**/*.h")
        )
        for cpp_file in cpp_files:
            if cpp_file.is_file():
                await self.test_insert_file(cpp_file)
                await asyncio.sleep(
                    0.5
                )  # Small delay between uploads to avoid overwhelming server

        # Also insert the README
        readme_file = cpp_repo_path / "README.md"
        if readme_file.exists():
            await self.test_insert_file(readme_file)

        # Wait a bit for indexing to complete
        logger.info("Waiting for indexing to complete...")
        await asyncio.sleep(5)

        # Test 3: Query operations
        logger.info("\n--- Testing Query Operations ---")
        test_queries = [
            ("What is the Calculator class?", "hybrid"),
            ("Describe the main function", "local"),
            ("What mathematical operations are supported?", "global"),
            ("How does the power function work?", "naive"),
        ]

        for query, mode in test_queries:
            await self.test_query(query, mode)
            await asyncio.sleep(1)  # Brief delay between queries

        # Test 4: Query with structured data
        logger.info("\n--- Testing Query Data Endpoint ---")
        await self.test_query_with_data(
            "What classes are defined in the code?", "hybrid"
        )
        await self.test_query_with_data("List all functions", "local")

        # Test 5: Graph data retrieval
        logger.info("\n--- Testing Graph Retrieval ---")
        await self.test_graph_data()

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Test Summary")
        logger.info("=" * 80)

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed

        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")

        if failed > 0:
            logger.info("\nFailed Tests:")
            for result in self.test_results:
                if not result["passed"]:
                    logger.info(f"  - {result['test']}: {result['message']}")

        return failed == 0


async def main():
    """Main test execution."""
    # Get test repository path
    script_dir = Path(__file__).parent
    cpp_repo_path = script_dir / "sample_cpp_repo"

    if not cpp_repo_path.exists():
        logger.error(f"Sample C++ repository not found at {cpp_repo_path}")
        return 1

    # Get server URL from environment or use default
    base_url = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")

    # Run tests
    async with IntegrationTestRunner(base_url) as runner:
        success = await runner.run_all_tests(cpp_repo_path)
        return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
