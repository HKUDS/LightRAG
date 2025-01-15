"""
LightRAG Ollama 兼容接口测试脚本

这个脚本测试 LightRAG 的 Ollama 兼容接口，包括：
1. 基本功能测试（流式和非流式响应）
2. 查询模式测试（local、global、naive、hybrid）
3. 错误处理测试（包括流式和非流式场景）

所有响应都使用 JSON Lines 格式，符合 Ollama API 规范。
"""

import requests
import json
import argparse
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

class OutputControl:
    """输出控制类，管理测试输出的详细程度"""
    _verbose: bool = False

    @classmethod
    def set_verbose(cls, verbose: bool) -> None:
        """设置输出详细程度
        
        Args:
            verbose: True 为详细模式，False 为静默模式
        """
        cls._verbose = verbose

    @classmethod
    def is_verbose(cls) -> bool:
        """获取当前输出模式
        
        Returns:
            当前是否为详细模式
        """
        return cls._verbose

@dataclass
class TestResult:
    """测试结果数据类"""
    name: str
    success: bool
    duration: float
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        """初始化后设置时间戳"""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class TestStats:
    """测试统计信息"""
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
    
    def add_result(self, result: TestResult):
        """添加测试结果"""
        self.results.append(result)
    
    def export_results(self, path: str = "test_results.json"):
        """导出测试结果到 JSON 文件
        
        Args:
            path: 输出文件路径
        """
        results_data = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
                "total_duration": sum(r.duration for r in self.results)
            }
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        print(f"\n测试结果已保存到: {path}")
    
    def print_summary(self):
        """打印测试统计摘要"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        duration = sum(r.duration for r in self.results)
        
        print("\n=== 测试结果摘要 ===")
        print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总用时: {duration:.2f}秒")
        print(f"总计: {total} 个测试")
        print(f"通过: {passed} 个")
        print(f"失败: {failed} 个")
        
        if failed > 0:
            print("\n失败的测试:")
            for result in self.results:
                if not result.success:
                    print(f"- {result.name}: {result.error}")

# 默认配置
DEFAULT_CONFIG = {
    "server": {
        "host": "localhost",
        "port": 9621,
        "model": "lightrag:latest",
        "timeout": 30,  # 请求超时时间（秒）
        "max_retries": 3,  # 最大重试次数
        "retry_delay": 1  # 重试间隔（秒）
    },
    "test_cases": {
        "basic": {
            "query": "唐僧有几个徒弟"
        }
    }
}

def make_request(url: str, data: Dict[str, Any], stream: bool = False) -> requests.Response:
    """发送 HTTP 请求，支持重试机制
    
    Args:
        url: 请求 URL
        data: 请求数据
        stream: 是否使用流式响应
        
    Returns:
        requests.Response 对象
        
    Raises:
        requests.exceptions.RequestException: 请求失败且重试次数用完
    """
    server_config = CONFIG["server"]
    max_retries = server_config["max_retries"]
    retry_delay = server_config["retry_delay"]
    timeout = server_config["timeout"]
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=data,
                stream=stream,
                timeout=timeout
            )
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # 最后一次重试
                raise
            print(f"\n请求失败，{retry_delay}秒后重试: {str(e)}")
            time.sleep(retry_delay)

def load_config() -> Dict[str, Any]:
    """加载配置文件
    
    首先尝试从当前目录的 config.json 加载，
    如果不存在则使用默认配置
    
    Returns:
        配置字典
    """
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG

def print_json_response(data: Dict[str, Any], title: str = "", indent: int = 2) -> None:
    """格式化打印 JSON 响应数据
    
    Args:
        data: 要打印的数据字典
        title: 打印的标题
        indent: JSON 缩进空格数
    """
    if OutputControl.is_verbose():
        if title:
            print(f"\n=== {title} ===")
        print(json.dumps(data, ensure_ascii=False, indent=indent))

# 全局配置
CONFIG = load_config()

def get_base_url() -> str:
    """返回基础 URL"""
    server = CONFIG["server"]
    return f"http://{server['host']}:{server['port']}/api/chat"

def create_request_data(
    content: str,
    stream: bool = False,
    model: str = None
) -> Dict[str, Any]:
    """创建基本的请求数据
    
    Args:
        content: 用户消息内容
        stream: 是否使用流式响应
        model: 模型名称
        
    Returns:
        包含完整请求数据的字典
    """
    return {
        "model": model or CONFIG["server"]["model"],
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "stream": stream
    }

# 全局测试统计
STATS = TestStats()

def run_test(func: Callable, name: str) -> None:
    """运行测试并记录结果
    
    Args:
        func: 测试函数
        name: 测试名称
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

def test_non_stream_chat():
    """测试非流式调用 /api/chat 接口"""
    url = get_base_url()
    data = create_request_data(
        CONFIG["test_cases"]["basic"]["query"],
        stream=False
    )
    
    # 发送请求
    response = make_request(url, data)
    
    # 打印响应
    if OutputControl.is_verbose():
        print("\n=== 非流式调用响应 ===")
    response_json = response.json()
    
    # 打印响应内容
    print_json_response({
        "model": response_json["model"],
        "message": response_json["message"]
    }, "响应内容")
    
    # # 打印性能统计
    # print_json_response({
    #     "total_duration": response_json["total_duration"],
    #     "load_duration": response_json["load_duration"],
    #     "prompt_eval_count": response_json["prompt_eval_count"],
    #     "prompt_eval_duration": response_json["prompt_eval_duration"],
    #     "eval_count": response_json["eval_count"],
    #     "eval_duration": response_json["eval_duration"]
    # }, "性能统计")

def test_stream_chat():
    """测试流式调用 /api/chat 接口
    
    使用 JSON Lines 格式处理流式响应，每行是一个完整的 JSON 对象。
    响应格式：
    {
        "model": "lightrag:latest",
        "created_at": "2024-01-15T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "部分响应内容",
            "images": null
        },
        "done": false
    }
    
    最后一条消息会包含性能统计信息，done 为 true。
    """
    url = get_base_url()
    data = create_request_data(
        CONFIG["test_cases"]["basic"]["query"],
        stream=True
    )
    
    # 发送请求并获取流式响应
    response = make_request(url, data, stream=True)
    
    if OutputControl.is_verbose():
        print("\n=== 流式调用响应 ===")
    output_buffer = []
    try:
        for line in response.iter_lines():
            if line:  # 跳过空行
                try:
                    # 解码并解析 JSON
                    data = json.loads(line.decode('utf-8'))
                    if data.get("done", True):  # 如果是完成标记
                        if "total_duration" in data:  # 最终的性能统计消息
                            # print_json_response(data, "性能统计")
                            break
                    else:  # 正常的内容消息
                        message = data.get("message", {})
                        content = message.get("content", "")
                        if content:  # 只收集非空内容
                            output_buffer.append(content)
                            print(content, end="", flush=True)  # 实时打印内容
                except json.JSONDecodeError:
                    print("Error decoding JSON from response line")
    finally:
        response.close()  # 确保关闭响应连接
        
    # 打印一个换行
    print()

def test_query_modes():
    """测试不同的查询模式前缀
    
    支持的查询模式：
    - /local: 本地检索模式，只在相关度高的文档中搜索
    - /global: 全局检索模式，在所有文档中搜索
    - /naive: 朴素模式，不使用任何优化策略
    - /hybrid: 混合模式（默认），结合多种策略
    
    每个模式都会返回相同格式的响应，但检索策略不同。
    """
    url = get_base_url()
    modes = ["local", "global", "naive", "hybrid"]  # 支持的查询模式
    
    for mode in modes:
        if OutputControl.is_verbose():
            print(f"\n=== 测试 /{mode} 模式 ===")
        data = create_request_data(
            f"/{mode} {CONFIG['test_cases']['basic']['query']}",
            stream=False
        )
        
        # 发送请求
        response = make_request(url, data)
        response_json = response.json()
        
        # 打印响应内容
        print_json_response({
            "model": response_json["model"],
            "message": response_json["message"]
        })

def create_error_test_data(error_type: str) -> Dict[str, Any]:
    """创建用于错误测试的请求数据
    
    Args:
        error_type: 错误类型，支持：
            - empty_messages: 空消息列表
            - invalid_role: 无效的角色字段
            - missing_content: 缺少内容字段
    
    Returns:
        包含错误数据的请求字典
    """
    error_data = {
        "empty_messages": {
            "model": "lightrag:latest",
            "messages": [],
            "stream": True
        },
        "invalid_role": {
            "model": "lightrag:latest",
            "messages": [
                {
                    "invalid_role": "user",
                    "content": "测试消息"
                }
            ],
            "stream": True
        },
        "missing_content": {
            "model": "lightrag:latest",
            "messages": [
                {
                    "role": "user"
                }
            ],
            "stream": True
        }
    }
    return error_data.get(error_type, error_data["empty_messages"])

def test_stream_error_handling():
    """测试流式响应的错误处理
    
    测试场景：
    1. 空消息列表
    2. 消息格式错误（缺少必需字段）
    
    错误响应会立即返回，不会建立流式连接。
    状态码应该是 4xx，并返回详细的错误信息。
    """
    url = get_base_url()
    
    if OutputControl.is_verbose():
        print("\n=== 测试流式响应错误处理 ===")
    
    # 测试空消息列表
    if OutputControl.is_verbose():
        print("\n--- 测试空消息列表（流式）---")
    data = create_error_test_data("empty_messages")
    response = make_request(url, data, stream=True)
    print(f"状态码: {response.status_code}")
    if response.status_code != 200:
        print_json_response(response.json(), "错误信息")
    response.close()
    
    # 测试无效角色字段
    if OutputControl.is_verbose():
        print("\n--- 测试无效角色字段（流式）---")
    data = create_error_test_data("invalid_role")
    response = make_request(url, data, stream=True)
    print(f"状态码: {response.status_code}")
    if response.status_code != 200:
        print_json_response(response.json(), "错误信息")
    response.close()

    # 测试缺少内容字段
    if OutputControl.is_verbose():
        print("\n--- 测试缺少内容字段（流式）---")
    data = create_error_test_data("missing_content")
    response = make_request(url, data, stream=True)
    print(f"状态码: {response.status_code}")
    if response.status_code != 200:
        print_json_response(response.json(), "错误信息")
    response.close()

def test_error_handling():
    """测试非流式响应的错误处理
    
    测试场景：
    1. 空消息列表
    2. 消息格式错误（缺少必需字段）
    
    错误响应格式：
    {
        "detail": "错误描述"
    }
    
    所有错误都应该返回合适的 HTTP 状态码和清晰的错误信息。
    """
    url = get_base_url()
    
    if OutputControl.is_verbose():
        print("\n=== 测试错误处理 ===")
    
    # 测试空消息列表
    if OutputControl.is_verbose():
        print("\n--- 测试空消息列表 ---")
    data = create_error_test_data("empty_messages")
    data["stream"] = False  # 修改为非流式模式
    response = make_request(url, data)
    print(f"状态码: {response.status_code}")
    print_json_response(response.json(), "错误信息")
    
    # 测试无效角色字段
    if OutputControl.is_verbose():
        print("\n--- 测试无效角色字段 ---")
    data = create_error_test_data("invalid_role")
    data["stream"] = False  # 修改为非流式模式
    response = make_request(url, data)
    print(f"状态码: {response.status_code}")
    print_json_response(response.json(), "错误信息")

    # 测试缺少内容字段
    if OutputControl.is_verbose():
        print("\n--- 测试缺少内容字段 ---")
    data = create_error_test_data("missing_content")
    data["stream"] = False  # 修改为非流式模式
    response = make_request(url, data)
    print(f"状态码: {response.status_code}")
    print_json_response(response.json(), "错误信息")

def get_test_cases() -> Dict[str, Callable]:
    """获取所有可用的测试用例
    
    Returns:
        测试名称到测试函数的映射字典
    """
    return {
        "non_stream": test_non_stream_chat,
        "stream": test_stream_chat,
        "modes": test_query_modes,
        "errors": test_error_handling,
        "stream_errors": test_stream_error_handling
    }

def create_default_config():
    """创建默认配置文件"""
    config_path = Path("config.json")
    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        print(f"已创建默认配置文件: {config_path}")

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LightRAG Ollama 兼容接口测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置文件 (config.json):
  {
    "server": {
      "host": "localhost",      # 服务器地址
      "port": 9621,            # 服务器端口
      "model": "lightrag:latest" # 默认模型名称
    },
    "test_cases": {
      "basic": {
        "query": "测试查询",      # 基本查询文本
        "stream_query": "流式查询" # 流式查询文本
      }
    }
  }
"""
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默模式，只显示测试结果摘要"
    )
    parser.add_argument(
        "-a", "--ask",
        type=str,
        help="指定查询内容，会覆盖配置文件中的查询设置"
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="创建默认配置文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_results.json",
        help="测试结果输出文件路径"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=list(get_test_cases().keys()) + ["all"],
        default=["all"],
        help="要运行的测试用例，可选: %(choices)s。使用 all 运行所有测试"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 设置输出模式
    OutputControl.set_verbose(not args.quiet)
    
    # 如果指定了查询内容，更新配置
    if args.ask:
        CONFIG["test_cases"]["basic"]["query"] = args.ask
    
    # 如果指定了创建配置文件
    if args.init_config:
        create_default_config()
        exit(0)
    
    test_cases = get_test_cases()
    
    try:
        if "all" in args.tests:
            # 运行所有测试
            if OutputControl.is_verbose():
                print("\n【基本功能测试】")
            run_test(test_non_stream_chat, "非流式调用测试")
            run_test(test_stream_chat, "流式调用测试")
            
            if OutputControl.is_verbose():
                print("\n【查询模式测试】")
            run_test(test_query_modes, "查询模式测试")
            
            if OutputControl.is_verbose():
                print("\n【错误处理测试】")
            run_test(test_error_handling, "错误处理测试")
            run_test(test_stream_error_handling, "流式错误处理测试")
        else:
            # 运行指定的测试
            for test_name in args.tests:
                if OutputControl.is_verbose():
                    print(f"\n【运行测试: {test_name}】")
                run_test(test_cases[test_name], test_name)
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
    finally:
        # 打印并导出测试统计
        STATS.print_summary()
        STATS.export_results(args.output)
