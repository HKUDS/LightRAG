# 文档处理系统

这是一个基于插件的文档处理系统，用于将JSON格式的文档内容转换为Markdown格式，同时提供了多种文档清洗和格式化功能。

## 功能特点

- **插件化架构**：允许用户自定义和扩展文档处理功能
- **模块化设计**：每个处理逻辑独立封装，易于测试和维护
- **批量处理**：支持处理单个文件或批量处理目录中的多个文件
- **灵活配置**：可通过配置文件控制插件的启用状态和执行顺序

## 安装与依赖

1. 确保已安装Python 3.7+
2. 安装依赖包：

```bash
pip install -r requirements.txt
```

## 使用方法

### 单文件转换

```bash
python main_converter.py <input_json_file> <output_markdown_file> [plugin_config_file]
```

示例：
```bash
python main_converter.py data/sample.json output.md
```

### 批量处理目录

```bash
python main_converter.py <directory_path> [plugin_config_file]
```

示例：
```bash
python main_converter.py data_directory/
```

## 可用插件

目前系统包含以下插件：

1. **空行移除器 (EmptyLineRemover)**：移除只包含空白字符的文本项
2. **标题级别检测器 (HeadingLevelDetector)**：检测基于数字和中文章节模式的标题级别
3. **条目级别调整器 (ArticleLevelAdjuster)**：根据文档结构调整"条"的级别

## 配置插件

可以通过编辑 `config/plugin_config.yaml` 来配置插件的启用状态和执行顺序。

```yaml
# 定义插件的执行顺序
plugin_order:
  - empty_line_remover
  - heading_level_detector
  - article_level_adjuster

# 控制每个插件是否启用
enabled_plugins:
  empty_line_remover: true
  heading_level_detector: true
  article_level_adjuster: true
```

## 开发新插件

1. 在 `plugins` 目录下创建新的Python文件，如 `my_plugin.py`
2. 定义一个继承自 `BasePlugin` 的类：

```python
from .base_plugin import BasePlugin

class MyPlugin(BasePlugin):
    """
    我的自定义插件
    """
    plugin_name = "我的插件名称"
    plugin_description = "插件功能描述"
    applicable_scenarios = "适用场景"
    notes = "使用注意事项"

    def process(self, data: list) -> list:
        # 实现文档处理逻辑
        # ...
        return processed_data
```

3. 在 `config/plugin_config.yaml` 中添加新插件：

```yaml
plugin_order:
  # 现有插件...
  - my_plugin
  
enabled_plugins:
  # 现有插件...
  my_plugin: true
```

## 测试

运行单元测试：

```bash
pytest tests/
```

添加新插件的测试用例：

1. 在 `tests` 目录下创建测试文件，如 `test_my_plugin.py`
2. 在 `tests/data` 目录下添加测试用的JSON数据
3. 编写测试函数

## 项目结构

```
document_process/
├── main_converter.py       # 主程序入口
├── plugins/                # 插件目录
│   ├── __init__.py
│   ├── base_plugin.py        # 插件基类
│   ├── heading_level_detector.py # 标题级别检测插件
│   ├── article_level_adjuster.py # 条目级别调整插件
│   └── empty_line_remover.py   # 空行移除插件
├── tests/                  # 测试目录
│   ├── __init__.py
│   ├── test_heading_level_detector.py
│   └── data/               # 测试数据
├── config/                 # 配置文件目录
│   └── plugin_config.yaml    # 插件配置文件
├── utils/                  # 工具函数目录
│   ├── __init__.py
│   └── markdown_utils.py     # Markdown生成相关函数
├── README.md               # 项目说明
└── requirements.txt        # 依赖库
``` 