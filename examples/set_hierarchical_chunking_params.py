#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例：如何不修改 lightrag.py 的情况下设置 markdown 层级分块的参数
"""

from lightrag import LightRAG
from lightrag.chunking import set_hierarchical_chunking_config, get_hierarchical_chunking_config
from lightrag.base import ChunkingMode

def main():
    # 初始化 LightRAG 实例，指定使用层级分块策略
    rag = LightRAG(
        chunking_mode=ChunkingMode.HIREARCHIACL,
        # 其他参数...
    )
    
    # 查看当前层级分块参数
    print("默认参数配置：")
    print(get_hierarchical_chunking_config())
    
    # 设置新的层级分块参数
    # 例如：处理到 4 级标题，3 级标题为父文档，不进行附件标题预处理
    set_hierarchical_chunking_config(
        heading_levels=4,        # 处理到 #### 级别标题
        parent_level=3,          # ### 级别标题作为父文档
        preprocess_attachments=False,  # 不预处理附件标题
    )
    
    # 再次查看配置，确认更改已生效
    print("\n修改后参数配置：")
    print(get_hierarchical_chunking_config())
    
    # 现在，当您使用 LightRAG 实例进行文档处理时，会应用上述配置
    # 示例：处理文档
    # rag.insert_document("您的文档内容", file_path="example.md")
    # 这将使用新的参数设置进行分块
    
    # 您也可以临时为单个文档指定不同的设置 (如果需要的话)
    print("\n处理完成后，配置仍然保持不变：")
    print(get_hierarchical_chunking_config())
    

if __name__ == "__main__":
    main() 