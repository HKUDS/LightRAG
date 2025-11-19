#!/bin/bash
# 三语言实体提取器安装脚本
# 自动安装 spaCy + HanLP 及相关模型

set -e  # 遇到错误立即退出

echo "=================================================="
echo "  三语言实体提取器安装"
echo "  支持: 中文 (HanLP) + 英文 (spaCy) + 瑞典语 (spaCy)"
echo "=================================================="
echo ""

# 检查 Python 版本
echo "🔍 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python 版本: $python_version"

# 安装 Python 依赖
echo ""
echo "📦 安装 Python 依赖包..."
echo "   - spaCy (英文 + 瑞典语)"
echo "   - HanLP (中文)"
pip install -r requirements-trilingual.txt

# 下载 spaCy 英文模型
echo ""
echo "⬇️  下载 spaCy 英文模型 (en_core_web_trf, ~440 MB)..."
python3 -m spacy download en_core_web_trf

# 下载 spaCy 瑞典语模型
echo ""
echo "⬇️  下载 spaCy 瑞典语模型 (sv_core_news_lg, ~545 MB)..."
python3 -m spacy download sv_core_news_lg

# HanLP 提示
echo ""
echo "ℹ️  HanLP 中文模型会在首次使用时自动下载 (~400 MB)"

# 完成
echo ""
echo "=================================================="
echo "  ✅ 安装完成！"
echo "=================================================="
echo ""
echo "磁盘空间使用:"
echo "  - spaCy 英文模型: ~440 MB"
echo "  - spaCy 瑞典语模型: ~545 MB"
echo "  - HanLP 中文模型: ~400 MB (首次使用时下载)"
echo "  - 总计: ~1.4 GB"
echo ""
echo "内存占用:"
echo "  - 按需加载: 同时只加载一个语言模型 (~1.5-1.8 GB)"
echo "  - 不会同时占用 4-5 GB 内存"
echo ""
echo "运行测试:"
echo "  python3 scripts/test_trilingual_extractor.py"
echo ""
