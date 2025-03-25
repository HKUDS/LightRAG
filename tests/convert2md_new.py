import json
import re
import os

def determine_number_level(text):
    """
    根据数字格式判断标题级别
    例如：
    "3 车体" -> 1 (只有当它是类似"3 车体"这样的格式时才匹配)
    "3.2 车体侧门" -> 2 (只有当它是类似"3.2 车体侧门"这样的格式时才匹配)
    "1．重大投诉问题" -> None (不应该匹配这种格式)
    """
    # 匹配类似 "3 车体" 或 "3.2 车体侧门" 的模式
    # 增加了更严格的匹配条件：数字后面必须跟着汉字，且不能是"．"这样的中文标点
    number_pattern = r'^(\d+(\.\d+)*)\s+[\u4e00-\u9fa5]'
    match = re.match(number_pattern, text)
    if match:
        # 计算点号的数量来确定层级
        dots = match.group(1).count('.')
        return dots + 1
    return None

def determine_chinese_section_level(text):
    """
    根据中文章节条判断标题级别
    例如：
    "第一章  总  则" -> 1
    "第十五章 总则" -> 1
    "第一节  方法" -> 2
    "第二十三节 方法" -> 2
    "第一条  为维护铁路旅客" -> 2（如果文档中存在"节"则为3）
    """
    # 定义中文数字
    CN_NUM = '一二三四五六七八九十'
    # 匹配章节条款的模式（支持更大的数字范围）
    # 可以匹配：一、二、三...九十、十一、二十一、三十、四十五等
    cn_num_pattern = f'[第]([{CN_NUM}]|[{CN_NUM}]?十[{CN_NUM}]?|[{CN_NUM}]百[{CN_NUM}]?十?[{CN_NUM}]?)'
    chapter_pattern = f'{cn_num_pattern}章\\s+'
    section_pattern = f'{cn_num_pattern}节\\s+'
    item_pattern = f'{cn_num_pattern}条\\s+'
    
    if re.match(chapter_pattern, text):
        return 1
    elif re.match(section_pattern, text):
        return 2
    elif re.match(item_pattern, text):
        return 2  # 默认条为2级，在json_to_markdown中会根据章节结构进行调整
    return None

def should_ignore_text_level(text):
    """
    判断是否应该忽略text_level字段
    例如：
    "（1）部门职责" -> True
    "（二）站段职责" -> True
    """
    # 匹配中文或阿拉伯数字的括号编号模式
    cn_num = '一二三四五六七八九十'
    pattern = r'^[（\(]([0-9]+|[' + cn_num + r']+)[）\)]'
    return bool(re.match(pattern, text))

def determine_heading_level(text):
    """
    根据文本内容判断标题级别，整合数字模式和中文章节条模式
    """
    # 首先尝试数字模式
    level = determine_number_level(text)
    if level is not None:
        return level
    
    # 然后尝试中文章节条模式
    level = determine_chinese_section_level(text)
    if level is not None:
        return level
    
    return None

def detect_sub_patterns(text_items):
    """
    检测文本中可能的子分级模式
    
    Args:
        text_items (list): 包含文本项的列表
        
    Returns:
        dict: 包含检测到的模式和对应的正则表达式
    """
    patterns = {}
    
    # 定义可能的模式
    number_pattern = r'^(\d+)[\.\、]'  # 数字标号模式：1. 或 1、
    bullet_pattern = r'^[•➢★\*\-]'  # 符号标记模式
    parenthesis_pattern = r'^[（\(]([0-9]+|[一二三四五六七八九十]+)[）\)]'  # 括号编号模式
    
    # 统计各种模式的出现次数
    number_count = 0
    bullet_count = 0
    parenthesis_count = 0
    
    for item in text_items:
        if item.get('type') != 'text':
            continue
            
        text = item.get('text', '')
        
        if re.match(number_pattern, text):
            number_count += 1
        elif re.match(bullet_pattern, text):
            bullet_count += 1
        elif re.match(parenthesis_pattern, text):
            parenthesis_count += 1
    
    # 如果某种模式出现次数超过阈值(3次)，则认为是有效的子分级模式
    threshold = 3
    if number_count >= threshold:
        patterns['number'] = number_pattern
    if bullet_count >= threshold:
        patterns['bullet'] = bullet_pattern
    if parenthesis_count >= threshold:
        patterns['parenthesis'] = parenthesis_pattern
    
    return patterns

def process_document_for_auto_sublevels(data):
    """
    处理文档，自动为需要的部分添加子分级
    
    Args:
        data (list): 原始文档数据
        
    Returns:
        list: 处理后的文档数据
    """
    # 第一步：标记所有节点的level
    for item in data:
        if item.get('type') == 'text':
            text = item.get('text', '')
            
            # 判断是否应该忽略text_level
            if not should_ignore_text_level(text):
                # 自动判断标题级别
                level = determine_heading_level(text)
                if level is not None:
                    item['text_level'] = level
    
    # 第二步：构建文档的层级结构
    hierarchy = []
    current_headings = [None] * 10  # 假设最多10个层级
    section_content = {}  # 每个标题对应的内容
    
    for item in data:
        if item.get('type') != 'text':
            # 非文本项直接添加到当前最低层级的内容中
            lowest_heading = None
            for heading in reversed(current_headings):
                if heading is not None:
                    lowest_heading = heading
                    break
                    
            if lowest_heading is not None:
                if lowest_heading not in section_content:
                    section_content[lowest_heading] = []
                section_content[lowest_heading].append(item)
            continue
            
        text = item.get('text', '')
        level = item.get('text_level')
        
        if level is not None:
            # 这是一个标题
            current_headings[level-1] = text
            # 清除此级别以下的所有级别
            for i in range(level, len(current_headings)):
                current_headings[i] = None
                
            # 将标题添加到层级结构
            hierarchy.append({
                'text': text,
                'level': level,
                'item': item
            })
        else:
            # 这是正文内容，添加到当前最低层级的内容中
            lowest_heading = None
            for heading in reversed(current_headings):
                if heading is not None:
                    lowest_heading = heading
                    break
                    
            if lowest_heading is not None:
                if lowest_heading not in section_content:
                    section_content[lowest_heading] = []
                section_content[lowest_heading].append(item)
    
    # 第三步：检查每个分级是否需要添加子分级
    result_data = []
    processed_items = set()  # 跟踪已处理的项
    
    for i, section in enumerate(hierarchy):
        heading_text = section['text']
        level = section['level']
        
        # 检查是否有下一级标题
        has_sublevel = False
        if i < len(hierarchy) - 1:
            next_section = hierarchy[i + 1]
            if next_section['level'] > level:
                has_sublevel = True
        
        # 如果没有下一级标题，并且内容超过512个字符
        if not has_sublevel and heading_text in section_content:
            content = section_content[heading_text]
            total_chars = sum(len(item.get('text', '')) for item in content if item.get('type') == 'text')
            
            if total_chars > 512:
                # 检测可能的子分级模式
                patterns = detect_sub_patterns(content)
                
                if patterns:
                    # 创建新的子分级
                    sub_level = level + 1
                    
                    # 根据检测到的模式处理内容
                    pattern_type, pattern_regex = next(iter(patterns.items()))
                    current_subheading = None
                    subheading_content = []
                    
                    # 添加标题到结果
                    result_data.append(section['item'])
                    processed_items.add(id(section['item']))
                    
                    for item in content:
                        if item.get('type') != 'text':
                            # 非文本项直接添加到当前子标题的内容中
                            if current_subheading is not None:
                                subheading_content.append(item)
                            else:
                                result_data.append(item)
                            continue
                            
                        text = item.get('text', '')
                        match = re.match(pattern_regex, text)
                        
                        if match:
                            # 如果之前有子标题内容，先添加到结果
                            if current_subheading is not None:
                                result_data.extend(subheading_content)
                            
                            # 创建新的子标题
                            current_subheading = text
                            new_item = item.copy()
                            new_item['text_level'] = sub_level
                            result_data.append(new_item)
                            processed_items.add(id(item))
                            subheading_content = []
                        else:
                            # 添加到当前子标题的内容中
                            if current_subheading is not None:
                                subheading_content.append(item)
                            else:
                                result_data.append(item)
                    
                    # 添加最后一个子标题的内容
                    if current_subheading is not None:
                        result_data.extend(subheading_content)
                    
                    continue  # 已处理此部分，继续下一个标题
        
        # 如果不需要添加子分级或无法添加，直接添加标题和内容
        if id(section['item']) not in processed_items:
            result_data.append(section['item'])
            processed_items.add(id(section['item']))
            
        if heading_text in section_content:
            for item in section_content[heading_text]:
                if id(item) not in processed_items:
                    result_data.append(item)
                    processed_items.add(id(item))
    
    # 确保没有遗漏任何项
    for item in data:
        if id(item) not in processed_items:
            result_data.append(item)
    
    return result_data

def json_to_markdown(json_file_path, markdown_file_path):
    """
    Converts a JSON file to a Markdown file, handling text, images, and tables.

    Args:
        json_file_path (str): The path to the JSON file.
        markdown_file_path (str): The path to the output Markdown file.
    """

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return

    # 定义中文数字和匹配模式
    CN_NUM = '一二三四五六七八九十'
    cn_num_pattern = f'[第]([{CN_NUM}]|[{CN_NUM}]?十[{CN_NUM}]?|[{CN_NUM}]百[{CN_NUM}]?十?[{CN_NUM}]?)'
    chapter_pattern = f'{cn_num_pattern}章\\s+'
    section_pattern = f'{cn_num_pattern}节\\s+'
    item_pattern = f'{cn_num_pattern}条\\s+'
    
    # 用于存储每个章节是否包含"节"的信息
    chapter_structure = {}
    current_chapter = None
    
    # 第一遍扫描：识别章节结构
    for item in data:
        if item.get('type') == 'text':
            text = item.get('text', '')
            
            # 识别章节
            chapter_match = re.match(chapter_pattern, text)
            if chapter_match:
                current_chapter = text
                chapter_structure[current_chapter] = {'has_section': False}
            
            # 识别节，并标记当前章节包含节
            section_match = re.match(section_pattern, text)
            if section_match and current_chapter:
                chapter_structure[current_chapter]['has_section'] = True
    
    # 处理文档以自动添加子分级
    processed_data = process_document_for_auto_sublevels(data)
    
    # 重置为第一章
    current_chapter = None
    
    with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
        for item in processed_data:
            item_type = item.get('type')
            page_idx = item.get('page_idx', None)

            if item_type == 'text':
                text = item.get('text', '')
                
                # 更新当前章节
                chapter_match = re.match(chapter_pattern, text)
                if chapter_match:
                    current_chapter = text
                
                # 判断是否应该忽略text_level
                if should_ignore_text_level(text):
                    text_level = None
                else:
                    # 自动判断标题级别，如果已经在process_document_for_auto_sublevels中设置了，使用设置的级别
                    text_level = item.get('text_level')
                    if text_level is None:
                        detected_level = determine_heading_level(text)
                        
                        # 特殊处理"条"级别
                        item_match = re.match(item_pattern, text)
                        if item_match and detected_level == 2 and current_chapter:
                            # 如果当前章节有"节"，则"条"为3级标题，否则为2级标题
                            if current_chapter in chapter_structure and chapter_structure[current_chapter]['has_section']:
                                detected_level = 3
                        
                        # 如果检测到标题级别，使用检测到的级别
                        text_level = detected_level
                
                if text_level is not None:
                    header_prefix = '#' * text_level
                    # 对于标题，在标题后添加页码信息
                    if page_idx is not None:
                        md_file.write(f"{header_prefix} {text} *(p.{page_idx})*\n\n")
                    else:
                        md_file.write(f"{header_prefix} {text}\n\n")
                else:
                    # 对于正文，在段落末尾添加页码信息
                    if page_idx is not None and text.strip():  # 只在非空文本后添加页码
                        md_file.write(f"{text} *(p.{page_idx})*\n\n")
                    else:
                        md_file.write(f"{text}\n\n")

            elif item_type == 'image':
                img_path = item.get('img_path', '')
                img_caption = item.get('img_caption', [])  # Get caption list

                md_file.write(f"![image]({img_path})\n\n") # Add a newline after the image

                if img_caption: # If there's a caption, use it
                    caption_text = " ".join(img_caption) # Join the list into a single string
                    if page_idx is not None:
                        md_file.write(f"{caption_text} *(p.{page_idx})*\n\n")
                    else:
                        md_file.write(f"{caption_text}\n\n")

            elif item_type == 'table':
                table_body = item.get('table_body', '')
                # Clean up the table_body to remove any surrounding <html><body><table>...</table></body></html>
                table_body = table_body.replace("<html><body><table>", "")
                table_body = table_body.replace("</table></body></html>", "")
                md_file.write(f"{table_body}\n\n")#Add a newline after the table
                if page_idx is not None:
                    md_file.write(f"*(p.{page_idx})*\n\n")
            else:
                print(f"Warning: Unknown item type: {item_type}")

    print(f"Successfully converted {json_file_path} to {markdown_file_path}")

def batch_process_directories(root_dir):
    """
    批量处理目录下的所有子目录中的content_list.json文件
    
    Args:
        root_dir (str): 根目录路径
    """
    # 遍历根目录下的所有子目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 在当前目录中查找包含content_list的json文件
        json_files = [f for f in filenames if 'content_list' in f and f.endswith('.json')]
        
        if json_files:  # 如果找到了符合条件的文件
            for json_file in json_files:
                json_path = os.path.join(dirpath, json_file)
                # 输出文件为当前目录下的output.md
                md_path = os.path.join(dirpath, 'output.md')
                
                print(f"Processing {json_path}")
                try:
                    json_to_markdown(json_path, md_path)
                    print(f"Successfully created {md_path}")
                except Exception as e:
                    print(f"Error processing {json_path}: {str(e)}")

# Example usage:
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了命令行参数，使用第一个参数作为根目录
        root_directory = sys.argv[1]
        batch_process_directories(root_directory)
    else:
        # 使用默认示例
        json_file = '/Users/llp/llp_experiments/management/383/383_content_list.json'
        markdown_file = '/Users/llp/llp_experiments/management/383/output.md'
        json_to_markdown(json_file, markdown_file)

# json_file = 'jna_content_list.json'  # Replace with the actual path to your JSON file
# markdown_file = 'output2.md'  # Replace with the desired path for the output Markdown file
# json_to_markdown(json_file, markdown_file)

