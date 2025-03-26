#!/usr/bin/env python3
import os
import pathlib
from pathlib import Path

def process_markdown_files(base_dir):
    """
    处理指定目录下每个子目录中的markdown文件:
    1. 查找每个子目录中的两个md文件（其中一个是output.md）
    2. 将非output.md的文件扩展名改为.bak
    3. 将output.md重命名为之前记录的文件名
    """
    base_path = Path(base_dir)
    # 确保基础目录存在
    if not base_path.exists() or not base_path.is_dir():
        print(f"错误: 目录 {base_dir} 不存在或不是一个目录")
        return

    # 遍历所有子目录
    for subdir in [d for d in base_path.iterdir() if d.is_dir()]:
        print(f"处理子目录: {subdir}")
        
        # 获取所有md文件
        md_files = list(subdir.glob("*.md"))
        
        # 检查是否有正好两个md文件，且其中一个是output.md
        if len(md_files) == 2:
            output_md = subdir / "output.md"
            if output_md in md_files:
                # 找到另一个非output.md的文件
                other_md = [f for f in md_files if f != output_md][0]
                other_md_name = other_md.name
                
                print(f"  找到文件: output.md 和 {other_md_name}")
                
                # 将另一个文件改名为.bak
                bak_file = other_md.with_suffix(".bak")
                other_md.rename(bak_file)
                print(f"  已将 {other_md_name} 重命名为 {bak_file.name}")
                
                # 将output.md改名为另一个文件的原名
                output_md.rename(subdir / other_md_name)
                print(f"  已将 output.md 重命名为 {other_md_name}")
            else:
                print(f"  子目录 {subdir} 中没有找到output.md文件")
        else:
            print(f"  子目录 {subdir} 中没有找到正好两个md文件")

if __name__ == "__main__":
    target_dir = "/Users/llp/llp_experiments/management/test1"
    process_markdown_files(target_dir)
    print("处理完成") 