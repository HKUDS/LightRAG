#!/usr/bin/env python3
import subprocess
import os
import sys
from openai import OpenAI

# 尝试加载 .env 文件的配置，若未安装 python-dotenv 则给出提示
try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    print("建议安装 python-dotenv 来加载 .env 文件的配置")

# 使用 .env 中的 LLM 配置
api_key = os.getenv("LLM_BINDING_API_KEY")
if not api_key:
    print("请设置 LLM_BINDING_API_KEY 环境变量")
    sys.exit(1)

# 限制 diff 的大小，防止超过模型的最大序列长度
def truncate_diff(diff, max_length=20000):
    """截断过长的 diff 内容"""
    if len(diff) <= max_length:
        return diff
    
    # 保留前10000和后10000个字符，确保包含首尾的关键信息
    half_length = max_length // 2
    return diff[:half_length] + "\n\n...[截断了中间部分的 diff]...\n\n" + diff[-half_length:]

api_base = os.getenv("LLM_BINDING_HOST", "https://api.siliconflow.cn/v1")
model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")

# 创建 OpenAI 客户端
client = OpenAI(api_key=api_key, base_url=api_base)

# 获取自上次提交以来的 diff
try:
    diff = subprocess.check_output(['git', 'diff', 'HEAD']).decode('utf-8')
except subprocess.CalledProcessError:
    print("无法获取 git diff。请确保当前目录是一个 git 仓库。")
    sys.exit(1)

if not diff.strip():
    print("没有需要提交的改动")
    sys.exit(0)

# 截断过长的 diff
truncated_diff = truncate_diff(diff)

# 构造 prompt
prompt = (
    "下面是一段 Git diff，描述了代码的修改。请生成一条详细且精确描述修改目的的 commit message。"
    "\n\nGit diff:\n"
    f"{truncated_diff}\n\nCommit message:"
)

# 调用 LLM 接口生成 commit message
try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=60
    )
    commit_message = response.choices[0].message.content.strip()
except Exception as e:
    print(f"调用 LLM 接口时发生错误: {e}")
    sys.exit(1)

print("生成的 commit message:", commit_message)

# 自动提交代码：添加改动、提交、推送
try:
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run(['git', 'commit', '-m', commit_message], check=True)
    subprocess.run(['git', 'push'], check=True)
    print("已成功提交并推送到远程仓库。")
except subprocess.CalledProcessError as e:
    print(f"执行 git 命令时出错: {e}")
    sys.exit(1)
