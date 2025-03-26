import os
import shutil
from pathlib import Path

def copy_md_files(source_dir, target_dir):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 计数器
    total_files = 0
    copied_files = 0
    
    # 遍历源目录及其所有子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.md'):
                total_files += 1
                # 构建源文件和目标文件的完整路径
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                
                # 如果目标文件已存在，添加数字后缀
                if os.path.exists(target_file):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_file):
                        target_file = os.path.join(target_dir, f"{base}_{counter}{ext}")
                        counter += 1
                
                # 拷贝文件
                shutil.copy2(source_file, target_file)
                copied_files += 1
                print(f"已拷贝: {source_file} -> {target_file}")
    
    print(f"\n拷贝完成！")
    print(f"总共找到 {total_files} 个md文件")
    print(f"成功拷贝 {copied_files} 个文件到目标目录")

if __name__ == "__main__":
    # 源目录和目标目录
    source_dir = "/Users/llp/llp_experiments/management/test1"
    target_dir = "/Users/llp/opensource/LightRAG/data/inputs/passenger"
    
    copy_md_files(source_dir, target_dir) 