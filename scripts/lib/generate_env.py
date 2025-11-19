#!/usr/bin/env python3
"""
环境变量生成器 - 将 YAML 配置转换为 .env 格式

从 config/local.yaml 读取配置，生成 .env 文件。
支持嵌套 YAML 扁平化为环境变量。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import yaml


def flatten_dict(data: Dict, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    扁平化嵌套字典

    Args:
        data: 嵌套字典
        parent_key: 父级键名
        sep: 分隔符

    Returns:
        扁平化后的字典

    Example:
        {'trilingual': {'enabled': True}} -> {'TRILINGUAL_ENABLED': True}
    """
    items = []

    for key, value in data.items():
        # 转换为大写并组合键名
        new_key = f"{parent_key}{sep}{key}".upper() if parent_key else key.upper()

        if isinstance(value, dict):
            # 递归处理嵌套字典
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))

    return dict(items)


def format_env_value(value: Any) -> str:
    """
    格式化环境变量值

    Args:
        value: 原始值

    Returns:
        格式化后的字符串

    Example:
        True -> 'true'
        123 -> '123'
        'hello world' -> 'hello world'
    """
    if isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # 如果字符串包含空格或特殊字符，添加引号
        if ' ' in value or any(c in value for c in ['#', '$', '&', '|', ';']):
            # 转义内部引号
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        return value
    elif value is None:
        return ''
    else:
        return str(value)


def load_config(config_path: Path) -> Dict:
    """
    加载 YAML 配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config if config else {}


def generate_env_content(config: Dict) -> str:
    """
    生成 .env 文件内容

    Args:
        config: 配置字典

    Returns:
        .env 格式的字符串
    """
    # 扁平化配置
    flat_config = flatten_dict(config)

    # 按键名排序
    sorted_items = sorted(flat_config.items())

    # 生成 .env 内容
    lines = [
        "# LightRAG 环境变量配置",
        "# 此文件由 scripts/setup.sh 自动生成，请勿手动编辑",
        "# 修改 config/local.yaml 后重新运行 ./scripts/setup.sh 更新此文件",
        "",
    ]

    current_section = None

    for key, value in sorted_items:
        # 提取顶级 section（第一个下划线之前的部分）
        section = key.split('_')[0]

        # 如果切换到新 section，添加分隔注释
        if section != current_section:
            if current_section is not None:
                lines.append("")  # 添加空行分隔
            lines.append(f"# {section}")
            current_section = section

        # 添加键值对
        formatted_value = format_env_value(value)
        lines.append(f"{key}={formatted_value}")

    return '\n'.join(lines) + '\n'


def save_env_file(content: str, env_path: Path) -> None:
    """
    保存 .env 文件

    Args:
        content: .env 文件内容
        env_path: 输出文件路径
    """
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    """主函数"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent

    # 文件路径
    config_path = project_root / 'config' / 'local.yaml'
    env_path = project_root / '.env'

    print("=" * 70)
    print("  环境变量生成器")
    print("=" * 70)
    print()

    try:
        # 加载配置
        print(f"📖 读取配置: {config_path.relative_to(project_root)}")
        config = load_config(config_path)
        print(f"   找到 {len(config)} 个顶级配置节")

        # 生成 .env 内容
        print(f"\n⚙️  生成环境变量...")
        env_content = generate_env_content(config)

        # 统计生成的环境变量数量
        env_count = len([line for line in env_content.split('\n') if '=' in line])
        print(f"   生成 {env_count} 个环境变量")

        # 保存 .env 文件
        print(f"\n💾 保存文件: {env_path.relative_to(project_root)}")
        save_env_file(env_content, env_path)

        print()
        print("=" * 70)
        print("  ✅ 环境变量生成成功")
        print("=" * 70)
        print()
        print(f"输出文件: {env_path}")
        print()
        print("提示:")
        print("  - .env 文件已添加到 .gitignore，不会提交到 Git")
        print("  - 修改 config/local.yaml 后重新运行此脚本更新 .env")
        print("  - 环境变量命名规则: 嵌套路径转大写并用下划线连接")
        print("    例如: trilingual.chinese.enabled -> TRILINGUAL_CHINESE_ENABLED")
        print()

    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
