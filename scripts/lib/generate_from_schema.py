#!/usr/bin/env python3
"""
配置生成器 - 从 Schema 生成本地配置

从 config/config.schema.yaml 读取配置元数据，生成 config/local.yaml。
支持深度合并、自动生成密钥、保留现有值。
"""

import sys
import secrets
import string
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def generate_secret(length: int = 32) -> str:
    """生成随机密钥"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def set_nested_value(data: Dict, path: str, value: Any) -> None:
    """
    设置嵌套字典的值

    Args:
        data: 目标字典
        path: 点分隔的路径，如 "trilingual.chinese.enabled"
        value: 要设置的值
    """
    keys = path.split('.')
    current = data

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def get_nested_value(data: Dict, path: str, default: Any = None) -> Any:
    """
    获取嵌套字典的值

    Args:
        data: 源字典
        path: 点分隔的路径
        default: 默认值

    Returns:
        找到的值或默认值
    """
    keys = path.split('.')
    current = data

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def deep_merge(base: Dict, overlay: Dict) -> Dict:
    """
    深度合并两个字典

    overlay 中的值会覆盖 base 中的值，但会保留 base 中 overlay 没有的键。

    Args:
        base: 基础字典
        overlay: 覆盖字典

    Returns:
        合并后的字典
    """
    result = base.copy()

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def infer_type(value: Any) -> Any:
    """
    推断并转换值的类型

    Args:
        value: 原始值

    Returns:
        转换后的值
    """
    if isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return value
    elif isinstance(value, float):
        return value
    elif isinstance(value, str):
        # 尝试转换为数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            # 尝试转换为布尔值
            if value.lower() in ('true', 'yes', 'on'):
                return True
            elif value.lower() in ('false', 'no', 'off'):
                return False
            return value
    else:
        return value


def load_schema(schema_path: Path) -> List[Dict]:
    """
    加载配置 Schema

    Args:
        schema_path: Schema 文件路径

    Returns:
        Schema 字段列表
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema 文件不存在: {schema_path}")

    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)

    if not isinstance(schema, list):
        raise ValueError("Schema 必须是列表格式")

    return schema


def load_existing_config(config_path: Path) -> Dict:
    """
    加载现有配置

    Args:
        config_path: 配置文件路径

    Returns:
        现有配置字典（如果文件不存在则返回空字典）
    """
    if not config_path.exists():
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config if config else {}


def generate_config(schema: List[Dict], existing_config: Dict) -> Dict:
    """
    从 Schema 生成配置

    Args:
        schema: Schema 字段列表
        existing_config: 现有配置

    Returns:
        生成的配置字典
    """
    config = {}

    for field in schema:
        section = field.get('section')
        if not section:
            continue

        # 检查是否有现有值
        existing_value = get_nested_value(existing_config, section)

        if existing_value is not None:
            # 保留现有值
            set_nested_value(config, section, existing_value)
        else:
            # 生成新值
            field_type = field.get('type', '')
            auto_generate = field.get('auto_generate', False)
            default_value = field.get('default')

            if field_type == 'secret' and auto_generate:
                # 自动生成密钥
                value = generate_secret(32)
            elif default_value is not None:
                # 使用默认值
                value = infer_type(default_value)
            else:
                # 跳过没有默认值的字段
                continue

            set_nested_value(config, section, value)

    return config


def save_config(config: Dict, config_path: Path) -> None:
    """
    保存配置到 YAML 文件

    Args:
        config: 配置字典
        config_path: 输出文件路径
    """
    # 确保目录存在
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=2
        )


def main():
    """主函数"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent

    # 文件路径
    schema_path = project_root / 'config' / 'config.schema.yaml'
    config_path = project_root / 'config' / 'local.yaml'

    print("=" * 70)
    print("  配置生成器")
    print("=" * 70)
    print()

    try:
        # 加载 Schema
        print(f"📖 读取 Schema: {schema_path.relative_to(project_root)}")
        schema = load_schema(schema_path)
        print(f"   找到 {len(schema)} 个配置字段")

        # 加载现有配置
        print(f"\n🔍 检查现有配置: {config_path.relative_to(project_root)}")
        existing_config = load_existing_config(config_path)

        if existing_config:
            print(f"   找到现有配置，将保留已有值")
        else:
            print(f"   未找到现有配置，将使用默认值")

        # 生成配置
        print(f"\n⚙️  生成配置...")
        config = generate_config(schema, existing_config)

        # 深度合并（保留现有配置中 schema 未定义的字段）
        if existing_config:
            config = deep_merge(existing_config, config)

        # 保存配置
        print(f"\n💾 保存配置: {config_path.relative_to(project_root)}")
        save_config(config, config_path)

        print()
        print("=" * 70)
        print("  ✅ 配置生成成功")
        print("=" * 70)
        print()
        print(f"配置文件: {config_path}")
        print()
        print("提示:")
        print("  - 配置文件已添加到 .gitignore，不会提交到 Git")
        print("  - 修改配置后重新运行此脚本可更新配置")
        print("  - 现有值会被保留，新字段会使用默认值")
        print()

    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
