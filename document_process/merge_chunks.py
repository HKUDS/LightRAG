#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import sys

def merge_chunks(json_file, output_file=None):
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 复制原始数据
    new_data = data.copy()
    new_chunks = []
    
    # 分类所有分块
    level1_chunks = []
    level2_chunks = {}  # 键为chunk_id，值为chunk对象
    level3_chunks = {}  # 键为parent_id，值为该parent_id下所有level3 chunk的列表
    
    for chunk in data['chunk']:
        chunk_id = chunk['chunk_id']
        
        # 根据chunk_id格式判断级别
        if re.search(r'_chunk_1_', chunk_id):
            level1_chunks.append(chunk)
        elif re.search(r'_chunk_2_', chunk_id):
            level2_chunks[chunk_id] = chunk
            # 初始化该level2 chunk下的level3 chunks列表
            level3_chunks[chunk_id] = []
        elif re.search(r'_chunk_3_', chunk_id):
            parent_id = chunk['parent_id']
            if parent_id in level3_chunks:
                level3_chunks[parent_id].append(chunk)
            else:
                level3_chunks[parent_id] = [chunk]
    
    # 合并内容
    for chunk_id, chunk in level2_chunks.items():
        # 获取该level2 chunk的所有level3 chunks
        children = level3_chunks.get(chunk_id, [])
        
        # 如果有level3 chunks，合并它们的内容到level2 chunk
        if children:
            # 先保存level2原始内容
            original_content = chunk['content']
            
            # 合并所有level3 chunks的content
            combined_content = original_content
            for child in sorted(children, key=lambda x: x['chunk_order_index']):
                # 提取level3 chunk的内容并合并
                child_content = child['content']
                combined_content += "\n\n" + child_content
            
            # 更新level2 chunk的content
            chunk['content'] = combined_content
            
            # 直接更新child_ids，移除所有level3的child_ids
            chunk['child_ids'] = []
        
        # 将更新后的level2 chunk添加到新的chunks列表
        new_chunks.append(chunk)
    
    # 添加level1 chunks到新的chunks列表
    for chunk in level1_chunks:
        # 更新level1 chunk的child_ids，仅保留level2 chunks
        child_ids = [child_id for child_id in chunk['child_ids'] if child_id in level2_chunks]
        chunk['child_ids'] = child_ids
        new_chunks.append(chunk)
    
    # 更新data的chunk字段
    new_data['chunk'] = new_chunks
    
    # 输出到文件
    if output_file is None:
        output_file = json_file.replace('.json', '_merged.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    
    print(f"已将三级分块合并为两级结构，合并后的文件已保存为: {output_file}")
    print(f"原始分块数: {len(data['chunk'])}, 合并后分块数: {len(new_chunks)}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        merge_chunks(json_file, output_file)
    else:
        print("用法: python merge_chunks.py <输入JSON文件> [输出JSON文件]") 