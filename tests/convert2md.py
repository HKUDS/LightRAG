import json
import re

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
    
    # 重置为第一章
    current_chapter = None
    
    with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
        for item in data:
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
                    # 自动判断标题级别
                    detected_level = determine_heading_level(text)
                    
                    # 特殊处理"条"级别
                    item_match = re.match(item_pattern, text)
                    if item_match and detected_level == 2 and current_chapter:
                        # 如果当前章节有"节"，则"条"为3级标题，否则为2级标题
                        if current_chapter in chapter_structure and chapter_structure[current_chapter]['has_section']:
                            detected_level = 3
                    
                    # 如果检测到标题级别，使用检测到的级别，否则使用原有级别
                    text_level = detected_level if detected_level is not None else item.get('text_level', None)
                
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

# Example usage:
json_file = '/Users/llp/llp_experiments/management/383/383_content_list.json'  # Replace with the actual path to your JSON file
markdown_file = '/Users/llp/llp_experiments/management/383/output.md'  # Replace with the desired path for the output Markdown file
json_to_markdown(json_file, markdown_file)

# json_file = 'jna_content_list.json'  # Replace with the actual path to your JSON file
# markdown_file = 'output2.md'  # Replace with the desired path for the output Markdown file
# json_to_markdown(json_file, markdown_file)

