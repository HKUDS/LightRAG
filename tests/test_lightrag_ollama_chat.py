"""
LightRAG Ollama Compatibility Interface Test Script

This script tests the LightRAG's Ollama compatibility interface, including:
1. Basic functionality tests (streaming and non-streaming responses)
2. Query mode tests (local, global, naive, hybrid)
3. Error handling tests (including streaming and non-streaming scenarios)

All responses use the JSON Lines format, complying with the Ollama API specification.
"""

import pytest
import requests
import json
import argparse
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum, auto


class ErrorCode(Enum):
    """Error codes for MCP errors"""

    InvalidRequest = auto()
    InternalError = auto()


class McpError(Exception):
    """Base exception class for MCP errors"""

    def __init__(self, code: ErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


DEFAULT_CONFIG = {
    "server": {
        "host": "localhost",
        "port": 9621,
        "model": "lightrag:latest",
        "timeout": 300,
        "max_retries": 1,
        "retry_delay": 1,
    },
    "test_cases": {
        "basic": {"query": "唐僧有几个徒弟"},
        "generate": {"query": "电视剧西游记导演是谁"},
    },
}

# Example conversation history for testing
EXAMPLE_CONVERSATION = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好!我是一个AI助手,很高兴为你服务。"},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I'm a Knowledge base query assistant."},
]


class OutputControl:
    """Output control class, manages the verbosity of test output"""

    _verbose: bool = False

    @classmethod
    def set_verbose(cls, verbose: bool) -> None:
        cls._verbose = verbose

    @classmethod
    def is_verbose(cls) -> bool:
        return cls._verbose


@dataclass
class TestResult:
    """Test result data class"""

    name: str
    success: bool
    duration: float
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TestStats:
    """Test statistics"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = datetime.now()

    def add_result(self, result: TestResult):
        self.results.append(result)

    def export_results(self, path: str = "test_results.json"):
        """Export test results to a JSON file
        Args:
            path: Output file path
        """
        results_data = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
                "total_duration": sum(r.duration for r in self.results),
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        print(f"\nTest results saved to: {path}")

    def print_summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        duration = sum(r.duration for r in self.results)

        print("\n=== Test Summary ===")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration:.2f} seconds")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for result in self.results:
                if not result.success:
                    print(f"- {result.name}: {result.error}")


def make_request(
    url: str, data: Dict[str, Any], stream: bool = False, check_status: bool = True
) -> requests.Response:
    """Send an HTTP request with retry mechanism
    Args:
        url: Request URL
        data: Request data
        stream: Whether to use streaming response
        check_status: Whether to check HTTP status code (default: True)
    Returns:
        requests.Response: Response object

    Raises:
        requests.exceptions.RequestException: Request failed after all retries
        requests.exceptions.HTTPError: HTTP status code is not 200 (when check_status is True)
    """
    server_config = CONFIG["server"]
    max_retries = server_config["max_retries"]
    retry_delay = server_config["retry_delay"]
    timeout = server_config["timeout"]

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, stream=stream, timeout=timeout)
            if check_status and response.status_code != 200:
                response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last retry
                raise
            print(f"\nRequest failed, retrying in {retry_delay} seconds: {str(e)}")
            time.sleep(retry_delay)


def load_config() -> Dict[str, Any]:
    """Load configuration file

    First try to load from config.json in the current directory,
    if it doesn't exist, use the default configuration
    Returns:
        Configuration dictionary
    """
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG


def print_json_response(data: Dict[str, Any], title: str = "", indent: int = 2) -> None:
    """Format and print JSON response data
    Args:
        data: Data dictionary to print
        title: Title to print
        indent: Number of spaces for JSON indentation
    """
    if OutputControl.is_verbose():
        if title:
            print(f"\n=== {title} ===")
        print(json.dumps(data, ensure_ascii=False, indent=indent))


# Global configuration
CONFIG = load_config()


def get_base_url(endpoint: str = "chat") -> str:
    """Return the base URL for specified endpoint
    Args:
        endpoint: API endpoint name (chat or generate)
    Returns:
        Complete URL for the endpoint
    """
    server = CONFIG["server"]
    return f"http://{server['host']}:{server['port']}/api/{endpoint}"


def create_chat_request_data(
    content: str,
    stream: bool = False,
    model: str = None,
    conversation_history: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create chat request data
    Args:
        content: User message content
        stream: Whether to use streaming response
        model: Model name
        conversation_history: List of previous conversation messages
        history_turns: Number of history turns to include
    Returns:
        Dictionary containing complete chat request data
    """
    messages = conversation_history or []
    messages.append({"role": "user", "content": content})

    return {
        "model": model or CONFIG["server"]["model"],
        "messages": messages,
        "stream": stream,
    }


def create_generate_request_data(
    prompt: str,
    system: str = None,
    stream: bool = False,
    model: str = None,
    options: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Create generate request data
    Args:
        prompt: Generation prompt
        system: System prompt
        stream: Whether to use streaming response
        model: Model name
        options: Additional options
    Returns:
        Dictionary containing complete generate request data
    """
    data = {
        "model": model or CONFIG["server"]["model"],
        "prompt": prompt,
        "stream": stream,
    }
    if system:
        data["system"] = system
    if options:
        data["options"] = options
    return data


# Global test statistics
STATS = TestStats()


def run_test(func: Callable, name: str) -> None:
    """Run a test and record the results
    Args:
        func: Test function
        name: Test name
    """
    start_time = time.time()
    try:
        func()
        duration = time.time() - start_time
        STATS.add_result(TestResult(name, True, duration))
    except Exception as e:
        duration = time.time() - start_time
        STATS.add_result(TestResult(name, False, duration, str(e)))
        raise


@pytest.mark.integration
@pytest.mark.requires_api
def test_non_stream_chat() -> None:
    """Test non-streaming call to /api/chat endpoint"""
    url = get_base_url()

    # Send request with conversation history
    data = create_chat_request_data(
        CONFIG["test_cases"]["basic"]["query"],
        stream=False,
        conversation_history=EXAMPLE_CONVERSATION,
    )
    response = make_request(url, data)

    # Print response
    if OutputControl.is_verbose():
        print("\n=== Non-streaming call response ===")
    response_json = response.json()

    # Print response content
    print_json_response(
        {"model": response_json["model"], "message": response_json["message"]},
        "Response content",
    )


@pytest.mark.integration
@pytest.mark.requires_api
def test_stream_chat() -> None:
    """Test streaming call to /api/chat endpoint

    Use JSON Lines format to process streaming responses, each line is a complete JSON object.
    Response format:
    {
        "model": "lightrag:latest",
        "created_at": "2024-01-15T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "Partial response content",
            "images": null
        },
        "done": false
    }

    The last message will contain performance statistics, with done set to true.
    """
    url = get_base_url()

    # Send request with conversation history
    data = create_chat_request_data(
        CONFIG["test_cases"]["basic"]["query"],
        stream=True,
        conversation_history=EXAMPLE_CONVERSATION,
    )
    response = make_request(url, data, stream=True)

    if OutputControl.is_verbose():
        print("\n=== Streaming call response ===")
    output_buffer = []
    try:
        for line in response.iter_lines():
            if line:  # Skip empty lines
                try:
                    # Decode and parse JSON
                    data = json.loads(line.decode("utf-8"))
                    if data.get("done", True):  # If it's the completion marker
                        if (
                            "total_duration" in data
                        ):  # Final performance statistics message
                            # print_json_response(data, "Performance statistics")
                            break
                    else:  # Normal content message
                        message = data.get("message", {})
                        content = message.get("content", "")
                        if content:  # Only collect non-empty content
                            output_buffer.append(content)
                            print(
                                content, end="", flush=True
                            )  # Print content in real-time
                except json.JSONDecodeError:
                    print("Error decoding JSON from response line")
    finally:
        response.close()  # Ensure the response connection is closed

    # Print a newline
    print()


@pytest.mark.integration
@pytest.mark.requires_api
def test_query_modes() -> None:
    """Test different query mode prefixes

    Supported query modes:
    - /local: Local retrieval mode, searches only in highly relevant documents
    - /global: Global retrieval mode, searches across all documents
    - /naive: Naive mode, does not use any optimization strategies
    - /hybrid: Hybrid mode (default), combines multiple strategies
    - /mix: Mix mode

    Each mode will return responses in the same format, but with different retrieval strategies.
    """
    url = get_base_url()
    modes = ["local", "global", "naive", "hybrid", "mix"]

    for mode in modes:
        if OutputControl.is_verbose():
            print(f"\n=== Testing /{mode} mode ===")
        data = create_chat_request_data(
            f"/{mode} {CONFIG['test_cases']['basic']['query']}", stream=False
        )

        # Send request
        response = make_request(url, data)
        response_json = response.json()

        # Print response content
        print_json_response(
            {"model": response_json["model"], "message": response_json["message"]}
        )


def create_error_test_data(error_type: str) -> Dict[str, Any]:
    """Create request data for error testing
    Args:
        error_type: Error type, supported:
            - empty_messages: Empty message list
            - invalid_role: Invalid role field
            - missing_content: Missing content field

    Returns:
        Request dictionary containing error data
    """
    error_data = {
        "empty_messages": {"model": "lightrag:latest", "messages": [], "stream": True},
        "invalid_role": {
            "model": "lightrag:latest",
            "messages": [{"invalid_role": "user", "content": "Test message"}],
            "stream": True,
        },
        "missing_content": {
            "model": "lightrag:latest",
            "messages": [{"role": "user"}],
            "stream": True,
        },
    }
    return error_data.get(error_type, error_data["empty_messages"])


@pytest.mark.integration
@pytest.mark.requires_api
def test_stream_error_handling() -> None:
    """Test error handling for streaming responses

    Test scenarios:
    1. Empty message list
    2. Message format error (missing required fields)

    Error responses should be returned immediately without establishing a streaming connection.
    The status code should be 4xx, and detailed error information should be returned.
    """
    url = get_base_url()

    if OutputControl.is_verbose():
        print("\n=== Testing streaming response error handling ===")

    # Test empty message list
    if OutputControl.is_verbose():
        print("\n--- Testing empty message list (streaming) ---")
    data = create_error_test_data("empty_messages")
    response = make_request(url, data, stream=True, check_status=False)
    print(f"Status code: {response.status_code}")
    if response.status_code != 200:
        print_json_response(response.json(), "Error message")
    response.close()

    # Test invalid role field
    if OutputControl.is_verbose():
        print("\n--- Testing invalid role field (streaming) ---")
    data = create_error_test_data("invalid_role")
    response = make_request(url, data, stream=True, check_status=False)
    print(f"Status code: {response.status_code}")
    if response.status_code != 200:
        print_json_response(response.json(), "Error message")
    response.close()

    # Test missing content field
    if OutputControl.is_verbose():
        print("\n--- Testing missing content field (streaming) ---")
    data = create_error_test_data("missing_content")
    response = make_request(url, data, stream=True, check_status=False)
    print(f"Status code: {response.status_code}")
    if response.status_code != 200:
        print_json_response(response.json(), "Error message")
    response.close()


@pytest.mark.integration
@pytest.mark.requires_api
def test_error_handling() -> None:
    """Test error handling for non-streaming responses

    Test scenarios:
    1. Empty message list
    2. Message format error (missing required fields)

    Error response format:
    {
        "detail": "Error description"
    }

    All errors should return appropriate HTTP status codes and clear error messages.
    """
    url = get_base_url()

    if OutputControl.is_verbose():
        print("\n=== Testing error handling ===")

    # Test empty message list
    if OutputControl.is_verbose():
        print("\n--- Testing empty message list ---")
    data = create_error_test_data("empty_messages")
    data["stream"] = False  # Change to non-streaming mode
    response = make_request(url, data, check_status=False)
    print(f"Status code: {response.status_code}")
    print_json_response(response.json(), "Error message")

    # Test invalid role field
    if OutputControl.is_verbose():
        print("\n--- Testing invalid role field ---")
    data = create_error_test_data("invalid_role")
    data["stream"] = False  # Change to non-streaming mode
    response = make_request(url, data, check_status=False)
    print(f"Status code: {response.status_code}")
    print_json_response(response.json(), "Error message")

    # Test missing content field
    if OutputControl.is_verbose():
        print("\n--- Testing missing content field ---")
    data = create_error_test_data("missing_content")
    data["stream"] = False  # Change to non-streaming mode
    response = make_request(url, data, check_status=False)
    print(f"Status code: {response.status_code}")
    print_json_response(response.json(), "Error message")


@pytest.mark.integration
@pytest.mark.requires_api
def test_non_stream_generate() -> None:
    """Test non-streaming call to /api/generate endpoint"""
    url = get_base_url("generate")
    data = create_generate_request_data(
        CONFIG["test_cases"]["generate"]["query"], stream=False
    )

    # Send request
    response = make_request(url, data)

    # Print response
    if OutputControl.is_verbose():
        print("\n=== Non-streaming generate response ===")
    response_json = response.json()

    # Print response content
    print(json.dumps(response_json, ensure_ascii=False, indent=2))


@pytest.mark.integration
@pytest.mark.requires_api
def test_stream_generate() -> None:
    """Test streaming call to /api/generate endpoint"""
    url = get_base_url("generate")
    data = create_generate_request_data(
        CONFIG["test_cases"]["generate"]["query"], stream=True
    )

    # Send request and get streaming response
    response = make_request(url, data, stream=True)

    if OutputControl.is_verbose():
        print("\n=== Streaming generate response ===")
    output_buffer = []
    try:
        for line in response.iter_lines():
            if line:  # Skip empty lines
                try:
                    # Decode and parse JSON
                    data = json.loads(line.decode("utf-8"))
                    if data.get("done", True):  # If it's the completion marker
                        if (
                            "total_duration" in data
                        ):  # Final performance statistics message
                            break
                    else:  # Normal content message
                        content = data.get("response", "")
                        if content:  # Only collect non-empty content
                            output_buffer.append(content)
                            print(
                                content, end="", flush=True
                            )  # Print content in real-time
                except json.JSONDecodeError:
                    print("Error decoding JSON from response line")
    finally:
        response.close()  # Ensure the response connection is closed

    # Print a newline
    print()


@pytest.mark.integration
@pytest.mark.requires_api
def test_generate_with_system() -> None:
    """Test generate with system prompt"""
    url = get_base_url("generate")
    data = create_generate_request_data(
        CONFIG["test_cases"]["generate"]["query"],
        system="你是一个知识渊博的助手",
        stream=False,
    )

    # Send request
    response = make_request(url, data)

    # Print response
    if OutputControl.is_verbose():
        print("\n=== Generate with system prompt response ===")
    response_json = response.json()

    # Print response content
    print_json_response(
        {
            "model": response_json["model"],
            "response": response_json["response"],
            "done": response_json["done"],
        },
        "Response content",
    )


@pytest.mark.integration
@pytest.mark.requires_api
def test_generate_error_handling() -> None:
    """Test error handling for generate endpoint"""
    url = get_base_url("generate")

    # Test empty prompt
    if OutputControl.is_verbose():
        print("\n=== Testing empty prompt ===")
    data = create_generate_request_data("", stream=False)
    response = make_request(url, data, check_status=False)
    print(f"Status code: {response.status_code}")
    print_json_response(response.json(), "Error message")

    # Test invalid options
    if OutputControl.is_verbose():
        print("\n=== Testing invalid options ===")
    data = create_generate_request_data(
        CONFIG["test_cases"]["basic"]["query"],
        options={"invalid_option": "value"},
        stream=False,
    )
    response = make_request(url, data, check_status=False)
    print(f"Status code: {response.status_code}")
    print_json_response(response.json(), "Error message")


@pytest.mark.integration
@pytest.mark.requires_api
def test_generate_concurrent() -> None:
    """Test concurrent generate requests"""
    import asyncio
    import aiohttp
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def get_session():
        async with aiohttp.ClientSession() as session:
            yield session

    async def make_request(session, prompt: str, request_id: int):
        url = get_base_url("generate")
        data = create_generate_request_data(prompt, stream=False)
        try:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    error_msg = (
                        f"Request {request_id} failed with status {response.status}"
                    )
                    if OutputControl.is_verbose():
                        print(f"\n{error_msg}")
                    raise McpError(ErrorCode.InternalError, error_msg)
                result = await response.json()
                if "error" in result:
                    error_msg = (
                        f"Request {request_id} returned error: {result['error']}"
                    )
                    if OutputControl.is_verbose():
                        print(f"\n{error_msg}")
                    raise McpError(ErrorCode.InternalError, error_msg)
                return result
        except Exception as e:
            error_msg = f"Request {request_id} failed: {str(e)}"
            if OutputControl.is_verbose():
                print(f"\n{error_msg}")
            raise McpError(ErrorCode.InternalError, error_msg)

    async def run_concurrent_requests():
        prompts = ["第一个问题", "第二个问题", "第三个问题", "第四个问题", "第五个问题"]

        async with get_session() as session:
            tasks = [
                make_request(session, prompt, i + 1) for i, prompt in enumerate(prompts)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_results = []
            error_messages = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_messages.append(f"Request {i+1} failed: {str(result)}")
                else:
                    success_results.append((i + 1, result))

            if error_messages:
                for req_id, result in success_results:
                    if OutputControl.is_verbose():
                        print(f"\nRequest {req_id} succeeded:")
                        print_json_response(result)

                error_summary = "\n".join(error_messages)
                raise McpError(
                    ErrorCode.InternalError,
                    f"Some concurrent requests failed:\n{error_summary}",
                )

            return results

    if OutputControl.is_verbose():
        print("\n=== Testing concurrent generate requests ===")

    # Run concurrent requests
    try:
        results = asyncio.run(run_concurrent_requests())
        # all success, print out results
        for i, result in enumerate(results, 1):
            print(f"\nRequest {i} result:")
            print_json_response(result)
    except McpError:
        # error message already printed
        raise


def get_test_cases() -> Dict[str, Callable]:
    """Get all available test cases
    Returns:
        A dictionary mapping test names to test functions
    """
    return {
        "non_stream": test_non_stream_chat,
        "stream": test_stream_chat,
        "modes": test_query_modes,
        "errors": test_error_handling,
        "stream_errors": test_stream_error_handling,
        "non_stream_generate": test_non_stream_generate,
        "stream_generate": test_stream_generate,
        "generate_with_system": test_generate_with_system,
        "generate_errors": test_generate_error_handling,
        "generate_concurrent": test_generate_concurrent,
    }


def create_default_config():
    """Create a default configuration file"""
    config_path = Path("config.json")
    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        print(f"Default configuration file created: {config_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LightRAG Ollama Compatibility Interface Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration file (config.json):
  {
    "server": {
      "host": "localhost",      # Server address
      "port": 9621,            # Server port
      "model": "lightrag:latest" # Default model name
    },
    "test_cases": {
      "basic": {
        "query": "Test query",      # Basic query text
        "stream_query": "Stream query" # Stream query text
      }
    }
  }
""",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Silent mode, only display test result summary",
    )
    parser.add_argument(
        "-a",
        "--ask",
        type=str,
        help="Specify query content, which will override the query settings in the configuration file",
    )
    parser.add_argument(
        "--init-config", action="store_true", help="Create default configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Test result output file path, default is not to output to a file",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=list(get_test_cases().keys()) + ["all"],
        default=["all"],
        help="Test cases to run, options: %(choices)s. Use 'all' to run all tests （except error tests)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set output mode
    OutputControl.set_verbose(not args.quiet)

    # If query content is specified, update the configuration
    if args.ask:
        CONFIG["test_cases"]["basic"]["query"] = args.ask

    # If specified to create a configuration file
    if args.init_config:
        create_default_config()
        exit(0)

    test_cases = get_test_cases()

    try:
        if "all" in args.tests:
            # Run all tests except error handling tests
            if OutputControl.is_verbose():
                print("\n【Chat API Tests】")
            run_test(test_non_stream_chat, "Non-streaming Chat Test")
            run_test(test_stream_chat, "Streaming Chat Test")
            run_test(test_query_modes, "Chat Query Mode Test")

            if OutputControl.is_verbose():
                print("\n【Generate API Tests】")
            run_test(test_non_stream_generate, "Non-streaming Generate Test")
            run_test(test_stream_generate, "Streaming Generate Test")
            run_test(test_generate_with_system, "Generate with System Prompt Test")
            run_test(test_generate_concurrent, "Generate Concurrent Test")
        else:
            # Run specified tests
            for test_name in args.tests:
                if OutputControl.is_verbose():
                    print(f"\n【Running Test: {test_name}】")
                run_test(test_cases[test_name], test_name)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Print test statistics
        STATS.print_summary()
        # If an output file path is specified, export the results
        if args.output:
            STATS.export_results(args.output)
