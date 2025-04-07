import json
import os
import importlib
import sys
import yaml  # 用于加载插件配置
from typing import List, Dict, Any

from utils.markdown_utils import generate_markdown

def load_plugins(plugin_dir: str, config_path: str) -> List[Any]:
    """
    动态加载并实例化配置文件中指定的插件。
    
    Args:
        plugin_dir (str): 插件目录路径
        config_path (str): 插件配置文件路径
        
    Returns:
        List[Any]: 加载的插件实例列表
    """
    plugins = []
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            plugin_order = config.get('plugin_order', [])
            enabled_plugins = config.get('enabled_plugins', {})
    except FileNotFoundError:
        print(f"警告: 在 {config_path} 找不到插件配置文件。不会加载任何插件。")
        return []
    except Exception as e:
        print(f"加载插件配置 {config_path} 时出错: {e}")
        return []

    print("加载插件...")
    for plugin_name in plugin_order:
        if enabled_plugins.get(plugin_name, False):  # 检查插件是否启用
            try:
                module_name = f"plugins.{plugin_name}"  # 假设文件名和插件名一致
                plugin_module = importlib.import_module(module_name)

                # 查找插件类 (假设类名是驼峰式，如 heading_level_detector -> HeadingLevelDetector)
                class_name = "".join(word.capitalize() for word in plugin_name.split('_'))
                plugin_class = getattr(plugin_module, class_name)
                plugin_instance = plugin_class()
                plugins.append(plugin_instance)
                print(f"  - 已加载插件: {plugin_instance.plugin_name}")  # 使用插件元数据
            except ModuleNotFoundError:
                print(f"  - 错误: 找不到插件模块 '{module_name}.py'.")
            except AttributeError:
                print(f"  - 错误: 在 '{module_name}.py' 中找不到插件类 '{class_name}'.")
            except Exception as e:
                print(f"  - 加载插件 '{plugin_name}' 时出错: {e}")
        else:
            print(f"  - 跳过禁用的插件: {plugin_name}")

    return plugins

def process_with_plugins(data: List[Dict[str, Any]], plugins: List[Any]) -> List[Dict[str, Any]]:
    """
    按顺序将数据传递给插件处理。
    
    Args:
        data (List[Dict[str, Any]]): 原始文档数据
        plugins (List[Any]): 已加载的插件实例列表
        
    Returns:
        List[Dict[str, Any]]: 处理后的文档数据
    """
    processed_data = data
    print("\n使用插件处理数据...")
    for plugin in plugins:
        print(f"  - 应用插件: {plugin.plugin_name}")
        try:
            processed_data = plugin.process(processed_data)
        except Exception as e:
            print(f"    使用 {plugin.plugin_name} 处理时出错: {e}")
            # 可以选择停止处理或继续下一个插件
            # raise  # 重新抛出异常以停止
            continue  # 继续下一个插件
    print("插件处理完成。")
    return processed_data

def convert_json_to_md(json_file_path: str, markdown_file_path: str, plugin_config_path: str):
    """
    主转换函数：加载JSON -> 加载插件 -> 应用插件 -> 生成Markdown。
    
    Args:
        json_file_path (str): 输入JSON文件路径
        markdown_file_path (str): 输出Markdown文件路径
        plugin_config_path (str): 插件配置文件路径
    """
    # 1. 加载 JSON 数据
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"成功从 {json_file_path} 加载JSON数据")
    except FileNotFoundError:
        print(f"错误: 在 {json_file_path} 找不到JSON文件")
        return
    except json.JSONDecodeError:
        print(f"错误: {json_file_path} 中的JSON格式无效")
        return
    except Exception as e:
        print(f"加载JSON时发生意外错误: {e}")
        return

    # 2. 加载插件
    plugin_dir = os.path.join(os.path.dirname(__file__), 'plugins')  # 获取插件目录路径
    loaded_plugins = load_plugins(plugin_dir, plugin_config_path)

    # 3. 应用插件处理数据
    processed_data = process_with_plugins(raw_data, loaded_plugins)

    # 4. 生成 Markdown 文件
    generate_markdown(processed_data, markdown_file_path)

def batch_process_directories(root_dir: str, plugin_config_path: str):
    """
    批量处理目录下的所有子目录中的content_list.json文件
    
    Args:
        root_dir (str): 根目录路径
        plugin_config_path (str): 插件配置文件路径
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        json_files = [f for f in filenames if 'content_list' in f and f.endswith('.json')]
        if json_files:
            for json_file in json_files:
                json_path = os.path.join(dirpath, json_file)
                # 假设输出文件名基于输入文件名
                base_name = os.path.splitext(json_file)[0]
                md_path = os.path.join(dirpath, f'{base_name}_output.md')  # 或者固定为 output.md

                print(f"\n处理 {json_path} -> {md_path}")
                try:
                    # 传递插件配置文件路径
                    convert_json_to_md(json_path, md_path, plugin_config_path)
                except Exception as e:
                    print(f"  处理 {json_path} 时出错: {str(e)}")


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
            print(f"警告: 指定的配置文件 '{config_input}' 未找到。使用默认值: '{default_config_path}'")
            config_input = default_config_path

        convert_json_to_md(json_input, md_output, config_input)

    elif len(sys.argv) == 2:
        # 命令行提供： python main_converter.py <root_directory> [config_yaml]
        # 认为是批量处理模式
        root_directory = sys.argv[1]
        config_input = default_config_path  # 批量模式暂用默认配置，可后续扩展
        if not os.path.exists(config_input):
            print(f"错误: 批量处理的默认配置文件 '{config_input}' 未找到。")
        else:
            batch_process_directories(root_directory, config_input)

    else:
        # 显示使用说明
        print("使用方法:")
        print("  单文件转换:   python main_converter.py <input_json> <output_markdown> [config_yaml]")
        print("  批量目录处理: python main_converter.py <root_directory> [config_yaml]")
        print("\n示例:")
        print("  python main_converter.py data/sample.json output.md")
        print("  python main_converter.py data_directory/") 