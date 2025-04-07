Okay, 这是一个非常好的起点，并且你已经提出了明确且重要的改进方向：**插件化、可测试性、独立性和文档化**。这正是将一个脚本转变为健壮、可维护工具的关键步骤。

让我们根据你的要求，一步步地设计和改进这个 Python 程序。

**核心思路：**

1.  **定义插件接口：** 规定每个插件必须实现的方法（例如 `process` 方法）。
2.  **插件加载机制：** 主程序能够发现、加载并按指定顺序执行 `plugins` 目录下的插件。
3.  **数据传递：** 主程序加载 JSON 数据后，将其依次传递给每个启用的插件进行处理。
4.  **Markdown 生成：** 经过所有插件处理后的最终数据，用于生成 Markdown 文件。
5.  **测试框架：** 使用 `pytest` 等框架为每个插件编写单元测试，并提供模拟 JSON 数据。
6.  **文档规范：** 为每个插件编写清晰的文档。

**项目结构建议：**

```
document_converter/
├── main_converter.py       # 主程序入口
├── plugins/                # 插件目录
│   ├── __init__.py
│   ├── base_plugin.py        # (可选) 插件基类
│   ├── heading_level_detector.py # 示例插件：标题级别检测
│   ├── article_level_adjuster.py # 示例插件：调整“条”的级别
│   ├── empty_line_remover.py   # 示例插件：移除空行
│   └── ... (其他插件)
├── tests/                  # 测试目录
│   ├── __init__.py
│   ├── test_heading_level_detector.py
│   ├── test_article_level_adjuster.py
│   ├── test_empty_line_remover.py
│   ├── data/                 # 测试用的模拟JSON文件
│   │   ├── heading_detector_input_1.json
│   │   ├── heading_detector_expected_1.json
│   │   ├── article_adjuster_input_1.json
│   │   ├── article_adjuster_expected_1.json
│   │   └── ...
│   └── conftest.py           # (可选) pytest 配置文件
├── config/                 # (可选) 配置文件目录
│   └── plugin_config.yaml    # (可选) 控制插件加载顺序和启用状态
├── utils/                  # (可选) 通用工具函数目录
│   ├── __init__.py
│   └── markdown_utils.py     # Markdown生成相关的辅助函数
├── README.md               # 项目说明
└── requirements.txt        # 依赖库
```

**步骤一：定义插件接口和基类 (可选)**

创建一个 `plugins/base_plugin.py` (可选，但推荐)，定义所有插件应遵循的接口。

```python
# plugins/base_plugin.py
from abc import ABC, abstractmethod

class BasePlugin(ABC):
    """
    插件基类，定义所有插件必须实现的接口。
    """
    # 插件元数据 (用于文档和选择)
    plugin_name = "Base Plugin"
    plugin_description = "This is a base class and should not be used directly."
    applicable_scenarios = "N/A"
    notes = "N/A"

    @abstractmethod
    def process(self, data: list) -> list:
        """
        处理文档数据列表。

        Args:
            data (list): 包含文档项（字典）的列表。

        Returns:
            list: 处理后的文档数据列表。
        """
        pass

    def get_metadata(self) -> dict:
        """返回插件的元数据"""
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "scenarios": self.applicable_scenarios,
            "notes": self.notes
        }

```

**步骤二：将现有逻辑拆分为插件**

将 `convert2md_new.py` 中的核心处理逻辑（如标题检测、级别调整等）移动到独立的插件文件中。

**示例插件 1：`plugins/heading_level_detector.py`**

```python
# plugins/heading_level_detector.py
import re
from .base_plugin import BasePlugin # 如果使用基类

# --- 从 convert2md_new.py 移过来的函数 ---
def determine_number_level(text):
    # ... (保持原样或稍作修改)
    number_pattern = r'^(\d+(\.\d+)*)\s+[\u4e00-\u9fa5]'
    match = re.match(number_pattern, text)
    if match:
        dots = match.group(1).count('.')
        return dots + 1
    return None

def determine_chinese_section_level(text):
    # ... (保持原样或稍作修改)
    CN_NUM = '一二三四五六七八九十'
    cn_num_pattern = f'[第]([{CN_NUM}]|[{CN_NUM}]?十[{CN_NUM}]?|[{CN_NUM}]百[{CN_NUM}]?十?[{CN_NUM}]?)'
    chapter_pattern = f'{cn_num_pattern}章\\s+'
    section_pattern = f'{cn_num_pattern}节\\s+'
    item_pattern = f'{cn_num_pattern}条\\s+'
    if re.match(chapter_pattern, text): return 1
    if re.match(section_pattern, text): return 2
    if re.match(item_pattern, text): return 2 # 初始判断为2级，后续插件可调整
    return None

def should_ignore_text_level(text):
    # ... (保持原样或稍作修改)
    cn_num = '一二三四五六七八九十'
    pattern = r'^[（\(]([0-9]+|[' + cn_num + r']+)[）\)]'
    return bool(re.match(pattern, text))

def determine_heading_level(text):
    # ... (保持原样或稍作修改)
    level = determine_number_level(text)
    if level is not None: return level
    level = determine_chinese_section_level(text)
    if level is not None: return level
    return None
# --- 结束移动的函数 ---

class HeadingLevelDetector(BasePlugin): # 继承基类
    """
    检测文本项的标题级别（基于数字和中文章节模式）。
    """
    plugin_name = "Heading Level Detector"
    plugin_description = "Detects heading levels based on numeric (e.g., '1.1 Text') and Chinese chapter/section/item patterns (e.g., '第一章', '第二节', '第三条'). Ignores parenthesized list items."
    applicable_scenarios = "Documents using standard numeric or Chinese chapter/section/item headings."
    notes = "Assigns an initial level to '条' items, which might be adjusted by other plugins like ArticleLevelAdjuster. Does not overwrite existing 'text_level' if present unless it's ignored."

    def process(self, data: list) -> list:
        """
        为文本项添加或更新 'text_level' 字段。
        """
        for item in data:
            if item.get('type') == 'text':
                text = item.get('text', '')
                # 只有在需要忽略现有级别，或者根本没有级别时，才进行检测
                if 'text_level' not in item or should_ignore_text_level(text):
                    if should_ignore_text_level(text):
                         # 如果是括号列表项，确保没有级别
                         if 'text_level' in item:
                             del item['text_level'] # 或者设置为 None
                    else:
                        level = determine_heading_level(text)
                        if level is not None:
                            item['text_level'] = level
                        elif 'text_level' in item:
                             # 如果检测不到级别，但之前有（且不是忽略类型），则移除
                             del item['text_level'] # 或者设置为 None

        return data

```

**示例插件 2：`plugins/article_level_adjuster.py`**

```python
# plugins/article_level_adjuster.py
import re
from .base_plugin import BasePlugin

class ArticleLevelAdjuster(BasePlugin):
    """
    调整“第X条”类型标题的级别。如果文档结构包含“章”和“节”，则“条”为3级；
    如果只有“章”没有“节”，则“条”为2级。
    """
    plugin_name = "Article Level Adjuster"
    plugin_description = "Adjusts the heading level of '第X条' items based on the presence of '第X节' within the same chapter."
    applicable_scenarios = "Documents structured with '章', '节', '条' hierarchy."
    notes = "Requires a previous plugin (like HeadingLevelDetector) to have assigned initial levels. Assumes '章' items are level 1 and '节' items are level 2."

    def process(self, data: list) -> list:
        CN_NUM = '一二三四五六七八九十'
        cn_num_pattern = f'[第]([{CN_NUM}]|[{CN_NUM}]?十[{CN_NUM}]?|[{CN_NUM}]百[{CN_NUM}]?十?[{CN_NUM}]?)'
        chapter_pattern = f'{cn_num_pattern}章\\s+'
        section_pattern = f'{cn_num_pattern}节\\s+'
        item_pattern = f'{cn_num_pattern}条\\s+'

        chapter_has_section = {}
        current_chapter_key = None

        # 1. 扫描确定哪些章包含节
        for item in data:
            if item.get('type') == 'text':
                text = item.get('text', '')
                level = item.get('text_level')

                if level == 1 and re.match(chapter_pattern, text):
                    current_chapter_key = text # 使用章节标题作为键
                    if current_chapter_key not in chapter_has_section:
                         chapter_has_section[current_chapter_key] = False
                elif level == 2 and re.match(section_pattern, text) and current_chapter_key:
                    chapter_has_section[current_chapter_key] = True

        # 2. 调整“条”的级别
        current_chapter_key = None
        for item in data:
             if item.get('type') == 'text':
                text = item.get('text', '')
                level = item.get('text_level')

                # 更新当前所在的章
                if level == 1 and re.match(chapter_pattern, text):
                     current_chapter_key = text

                # 检查是否是“条”并且需要调整
                # 注意：这里假设 HeadingLevelDetector 已将“条”设为 2 级
                if level == 2 and re.match(item_pattern, text) and current_chapter_key:
                    if chapter_has_section.get(current_chapter_key, False):
                        item['text_level'] = 3 # 如果当前章有节，条降为3级

        return data
```

**示例插件 3：`plugins/empty_line_remover.py`**

```python
# plugins/empty_line_remover.py
from .base_plugin import BasePlugin

class EmptyLineRemover(BasePlugin):
    """
    移除文档数据中空的文本项。
    """
    plugin_name = "Empty Line Remover"
    plugin_description = "Removes text items that contain only whitespace."
    applicable_scenarios = "General cleaning for most documents."
    notes = "Might remove intentional blank lines if they were represented as separate text items."

    def process(self, data: list) -> list:
        """
        过滤掉空的文本项。
        """
        processed_data = []
        for item in data:
            if item.get('type') == 'text':
                if item.get('text', '').strip(): # 检查文本去除空白后是否还有内容
                    processed_data.append(item)
            else:
                # 保留非文本项
                processed_data.append(item)
        return processed_data
```

**步骤三：修改主程序 (`main_converter.py`)**

主程序需要加载 JSON，动态加载并执行插件，最后生成 Markdown。

```python
# main_converter.py
import json
import os
import importlib
import sys
import yaml # 用于加载插件配置
from typing import List, Dict, Any

# (可以把 Markdown 生成逻辑移到 utils/markdown_utils.py)
def generate_markdown(data: List[Dict[str, Any]], markdown_file_path: str):
    """
    Generates a Markdown file from the processed data.
    """
    # --- 基本的 Markdown 生成逻辑 (从原 json_to_markdown 移来并简化) ---
    with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
        for item in data:
            item_type = item.get('type')
            page_idx = item.get('page_idx', None)
            text_level = item.get('text_level') # 直接使用插件处理后的级别

            page_info = f" *(p.{page_idx})*" if page_idx is not None else ""

            if item_type == 'text':
                text = item.get('text', '')
                if text_level is not None:
                    header_prefix = '#' * text_level
                    md_file.write(f"{header_prefix} {text}{page_info}\n\n")
                elif text.strip(): # 仅写入非空文本
                    md_file.write(f"{text}{page_info}\n\n")
                # 空文本行已被插件移除或在此处忽略

            elif item_type == 'image':
                img_path = item.get('img_path', '')
                img_caption = item.get('img_caption', [])
                md_file.write(f"![image]({img_path})\n\n")
                if img_caption:
                    caption_text = " ".join(img_caption)
                    md_file.write(f"{caption_text}{page_info}\n\n")
                elif page_info: # 如果没有标题但有页码
                     md_file.write(f"{page_info}\n\n")


            elif item_type == 'table':
                table_body = item.get('table_body', '')
                table_body = table_body.replace("<html><body><table>", "").replace("</table></body></html>", "")
                md_file.write(f"{table_body}\n\n")
                if page_info:
                    md_file.write(f"{page_info}\n\n")
            # else: # 可以选择性地处理未知类型
            #     print(f"Warning: Unknown item type encountered: {item_type}")

    print(f"Successfully generated Markdown file at {markdown_file_path}")
    # --- 结束 Markdown 生成逻辑 ---

def load_plugins(plugin_dir: str, config_path: str) -> List[Any]:
    """
    动态加载并实例化配置文件中指定的插件。
    """
    plugins = []
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            plugin_order = config.get('plugin_order', [])
            enabled_plugins = config.get('enabled_plugins', {})
    except FileNotFoundError:
        print(f"Warning: Plugin config file not found at {config_path}. No plugins will be loaded.")
        return []
    except Exception as e:
        print(f"Error loading plugin config {config_path}: {e}")
        return []

    print("Loading plugins...")
    for plugin_name in plugin_order:
        if enabled_plugins.get(plugin_name, False): # 检查插件是否启用
            try:
                module_name = f"plugins.{plugin_name}" # 假设文件名和插件名一致
                plugin_module = importlib.import_module(module_name)

                # 查找插件类 (假设类名是驼峰式，如 heading_level_detector -> HeadingLevelDetector)
                class_name = "".join(word.capitalize() for word in plugin_name.split('_'))
                plugin_class = getattr(plugin_module, class_name)
                plugin_instance = plugin_class()
                plugins.append(plugin_instance)
                print(f"  - Loaded plugin: {plugin_instance.plugin_name}") # 使用插件元数据
            except ModuleNotFoundError:
                print(f"  - Error: Plugin module '{module_name}.py' not found.")
            except AttributeError:
                 print(f"  - Error: Plugin class '{class_name}' not found in '{module_name}.py'.")
            except Exception as e:
                print(f"  - Error loading plugin '{plugin_name}': {e}")
        else:
             print(f"  - Skipping disabled plugin: {plugin_name}")

    return plugins

def process_with_plugins(data: List[Dict[str, Any]], plugins: List[Any]) -> List[Dict[str, Any]]:
    """
    按顺序将数据传递给插件处理。
    """
    processed_data = data
    print("\nProcessing data with plugins...")
    for plugin in plugins:
        print(f"  - Applying plugin: {plugin.plugin_name}")
        try:
            processed_data = plugin.process(processed_data)
        except Exception as e:
            print(f"    Error during processing with {plugin.plugin_name}: {e}")
            # 可以选择停止处理或继续下一个插件
            # raise # 重新抛出异常以停止
            continue # 继续下一个插件
    print("Plugin processing complete.")
    return processed_data

def convert_json_to_md(json_file_path: str, markdown_file_path: str, plugin_config_path: str):
    """
    主转换函数：加载JSON -> 加载插件 -> 应用插件 -> 生成Markdown。
    """
    # 1. 加载 JSON 数据
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"Successfully loaded JSON data from {json_file_path}")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading JSON: {e}")
        return

    # 2. 加载插件
    plugin_dir = os.path.join(os.path.dirname(__file__), 'plugins') # 获取插件目录路径
    loaded_plugins = load_plugins(plugin_dir, plugin_config_path)

    # 3. 应用插件处理数据
    processed_data = process_with_plugins(raw_data, loaded_plugins)

    # 4. 生成 Markdown 文件
    generate_markdown(processed_data, markdown_file_path)

# --- (保留 batch_process_directories 函数，但修改其调用) ---
def batch_process_directories(root_dir: str, plugin_config_path: str):
    """
    批量处理目录下的所有子目录中的content_list.json文件
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        json_files = [f for f in filenames if 'content_list' in f and f.endswith('.json')]
        if json_files:
            for json_file in json_files:
                json_path = os.path.join(dirpath, json_file)
                # 假设输出文件名基于输入文件名
                base_name = os.path.splitext(json_file)[0]
                md_path = os.path.join(dirpath, f'{base_name}_output.md') # 或者固定为 output.md

                print(f"\nProcessing {json_path} -> {md_path}")
                try:
                    # 传递插件配置文件路径
                    convert_json_to_md(json_path, md_path, plugin_config_path)
                except Exception as e:
                    print(f"  Error processing {json_path}: {str(e)}")


if __name__ == "__main__":
    # 确定插件配置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, 'config', 'plugin_config.yaml')

    if len(sys.argv) > 2:
        # 命令行提供： python main_converter.py <input_json> <output_markdown> [config_yaml]
        json_input = sys.argv[1]
        md_output = sys.argv[2]
        config_input = sys.argv[3] if len(sys.argv) > 3 else default_config_path
        if not os.path.exists(config_input):
             print(f"Warning: Specified config file '{config_input}' not found. Using default: '{default_config_path}'")
             config_input = default_config_path

        convert_json_to_md(json_input, md_output, config_input)

    elif len(sys.argv) == 2:
         # 命令行提供： python main_converter.py <root_directory> [config_yaml]
         # 认为是批量处理模式
         root_directory = sys.argv[1]
         config_input = default_config_path # 批量模式暂用默认配置，可后续扩展
         if not os.path.exists(config_input):
              print(f"Error: Default config file '{config_input}' not found for batch processing.")
         else:
              batch_process_directories(root_directory, config_input)

    else:
        # 使用默认示例进行单文件转换 (需要确保文件存在)
        print("Running default example...")
        example_json = '/path/to/your/example_content_list.json' # <--- 修改为你的示例JSON路径
        example_md = '/path/to/your/example_output.md'       # <--- 修改为你的示例输出MD路径
        example_config = default_config_path

        if not os.path.exists(example_json):
             print(f"Error: Default example JSON file not found at '{example_json}'. Please provide paths via command line or edit the script.")
        elif not os.path.exists(example_config):
             print(f"Error: Default config file not found at '{example_config}'.")
        else:
             convert_json_to_md(example_json, example_md, example_config)

```

**步骤四：创建插件配置文件 (`config/plugin_config.yaml`)**

这个文件控制哪些插件被加载以及它们的执行顺序。

```yaml
# config/plugin_config.yaml

# 定义插件的执行顺序
# 主程序将按照这个列表的顺序加载和执行启用的插件
plugin_order:
  - empty_line_remover        # 先移除空行
  - heading_level_detector    # 然后检测标题级别
  - article_level_adjuster    # 最后根据章节结构调整“条”的级别
  # - auto_sublevel_generator # (如果实现了这个插件) 可以加在这里
  # - other_plugin_name

# 控制每个插件是否启用
# 只有值为 true 的插件才会被加载和执行
enabled_plugins:
  empty_line_remover: true
  heading_level_detector: true
  article_level_adjuster: true
  # auto_sublevel_generator: false # 默认禁用，需要时再开启
  # other_plugin_name: true

```

**步骤五：编写测试 (`tests/`)**

使用 `pytest` 框架。为每个插件创建一个测试文件。

**示例测试：`tests/test_heading_level_detector.py`**

```python
# tests/test_heading_level_detector.py
import pytest
import json
import os

# 假设你的插件类在 'plugins.heading_level_detector' 模块中
from plugins.heading_level_detector import HeadingLevelDetector

# 获取测试数据目录的路径
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def load_json_data(filename):
    """辅助函数：加载测试用的JSON文件"""
    filepath = os.path.join(TEST_DATA_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"Test data file not found: {filepath}")
    except json.JSONDecodeError:
         pytest.fail(f"Invalid JSON in test data file: {filepath}")

# 参数化测试，可以添加更多测试用例
@pytest.mark.parametrize("input_filename, expected_filename", [
    ("heading_detector_input_1.json", "heading_detector_expected_1.json"),
    # ("heading_detector_input_2.json", "heading_detector_expected_2.json"), # 添加更多测试用例
])
def test_heading_detection(input_filename, expected_filename):
    """
    测试 HeadingLevelDetector 插件是否能正确检测标题级别。
    """
    # 1. 加载输入和预期输出数据
    input_data = load_json_data(input_filename)
    expected_data = load_json_data(expected_filename)

    # 2. 实例化并运行插件
    plugin = HeadingLevelDetector()
    processed_data = plugin.process(input_data) # 注意：插件可能会修改原始input_data，如果不想这样，传入副本

    # 3. 断言结果是否符合预期
    # 比较列表长度和每个字典的内容
    assert len(processed_data) == len(expected_data), "Number of items mismatch"
    for i, item in enumerate(processed_data):
        # 比较 text_level 字段，如果预期中没有，实际中也不应有
        processed_level = item.get('text_level')
        expected_level = expected_data[i].get('text_level')
        assert processed_level == expected_level, \
               f"Mismatch in text_level for item {i} ('{item.get('text', '')[:20]}...'): expected {expected_level}, got {processed_level}"
        # 可以选择性地比较其他关键字段是否被意外修改
        assert item.get('text') == expected_data[i].get('text'), f"Text content changed for item {i}"
        assert item.get('type') == expected_data[i].get('type'), f"Type changed for item {i}"

# 可以添加更多针对特定情况的测试函数
def test_ignore_parenthesized_items():
    """测试是否正确忽略括号列表项的级别设置"""
    plugin = HeadingLevelDetector()
    input_data = [
        {"type": "text", "text": "（一）第一点", "text_level": 1}, # 假设预处理错误地设置了级别
        {"type": "text", "text": "(2) 第二点"},
        {"type": "text", "text": "  (三) 第三点"},
    ]
    processed_data = plugin.process(input_data)
    assert 'text_level' not in processed_data[0] # 级别应被移除
    assert 'text_level' not in processed_data[1]
    assert 'text_level' not in processed_data[2]

def test_numeric_heading():
    """测试数字标题检测"""
    plugin = HeadingLevelDetector()
    input_data = [{"type": "text", "text": "1.1 小节标题"}]
    processed_data = plugin.process(input_data)
    assert processed_data[0].get('text_level') == 2

# --- 创建对应的模拟 JSON 文件 ---
# tests/data/heading_detector_input_1.json
# [
#   {"type": "text", "text": "第一章 总则"},
#   {"type": "text", "text": "第一条 定义"},
#   {"type": "text", "text": "（一）名词解释"},
#   {"type": "text", "text": "1.2 数值标题"},
#   {"type": "text", "text": "普通文本"}
# ]

# tests/data/heading_detector_expected_1.json
# [
#   {"type": "text", "text": "第一章 总则", "text_level": 1},
#   {"type": "text", "text": "第一条 定义", "text_level": 2},
#   {"type": "text", "text": "（一）名词解释"}, # 级别被忽略
#   {"type": "text", "text": "1.2 数值标题", "text_level": 2},
#   {"type": "text", "text": "普通文本"}
# ]

```

**运行测试:**

在项目根目录下运行 `pytest` 命令。

```bash
pip install pytest pyyaml # 安装测试和配置库
pytest tests/
```

**步骤六：编写文档**

1.  **插件文档 (Docstrings):** 如示例插件所示，在每个插件类或文件的开头使用清晰的 Docstring 描述其功能、适用场景和注意事项。`BasePlugin` 中的元数据字段可以帮助标准化。
2.  **项目 README.md:**
    *   项目简介和目标。
    *   如何安装和运行 (`pip install -r requirements.txt`, `python main_converter.py ...`)。
    *   项目结构说明。
    *   **插件系统说明:**
        *   如何创建新插件（继承 `BasePlugin`，实现 `process` 方法，添加元数据）。
        *   插件放置的位置 (`plugins/` 目录)。
        *   如何配置插件 (`config/plugin_config.yaml` 的用法，`plugin_order` 和 `enabled_plugins` 的含义)。
        *   现有插件列表及其简要说明（可以动态生成或手动维护）。
    *   **测试说明:**
        *   如何运行测试 (`pytest tests/`)。
        *   如何为新插件添加测试（创建 `test_*.py` 文件，准备模拟 JSON 数据）。
    *   依赖项 (`requirements.txt`)。
    *   贡献指南（如果需要）。

**总结与后续改进：**

*   **插件化:** 通过将逻辑拆分到 `plugins` 目录，实现了插件化。
*   **可测试性:** 使用 `pytest` 和模拟数据，可以对每个插件进行独立测试和回归测试。
*   **独立性:** 每个插件在自己的文件中，逻辑相对独立。
*   **模拟文档:** 测试数据 (`tests/data/`) 作为模拟文档。
*   **文档化:** 通过 Docstrings 和 README 实现文档。
*   **`process_document_for_auto_sublevels` 的处理:** 这个函数的逻辑比较复杂且基于启发式规则，容易出错。建议：
    *   将其也封装成一个插件 (`auto_sublevel_generator.py`)。
    *   在 `plugin_config.yaml` 中默认禁用它 (`enabled_plugins: auto_sublevel_generator: false`)。
    *   只在明确知道对某类文档有效时才启用它。
    *   或者，考虑是否可以用更明确、规则驱动的插件替代其功能。例如，一个专门识别 `1.`、`a)`、`(1)` 等列表项并添加缩进或标记的插件。
*   **错误处理:** 增强 `main_converter.py` 中的错误处理，例如插件加载失败、插件执行出错时的行为。
*   **日志记录:** 添加 `logging` 模块，记录插件加载、执行过程和潜在问题。
*   **Markdown 生成优化:** 可以将 `generate_markdown` 函数也移到 `utils/markdown_utils.py`，并可能进一步细化，例如提供不同的 Markdown 风格选项。

这个重构方案为你提供了一个灵活、可扩展且易于维护的基础。你可以根据遇到的新文档格式，不断添加和完善新的插件。