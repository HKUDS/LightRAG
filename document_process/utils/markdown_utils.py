from typing import List, Dict, Any

def generate_markdown(data: List[Dict[str, Any]], markdown_file_path: str):
    """
    从处理后的数据生成Markdown文件。
    
    Args:
        data (List[Dict[str, Any]]): 已处理的文档数据列表
        markdown_file_path (str): 输出Markdown文件的路径
    """
    with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
        for item in data:
            item_type = item.get('type')
            page_idx = item.get('page_idx', None)
            text_level = item.get('text_level')  # 直接使用插件处理后的级别

            page_info = f" *(p.{page_idx})*" if page_idx is not None else ""

            if item_type == 'text':
                text = item.get('text', '')
                if text_level is not None:
                    header_prefix = '#' * text_level
                    md_file.write(f"{header_prefix} {text}{page_info}\n\n")
                elif text.strip():  # 仅写入非空文本
                    md_file.write(f"{text}{page_info}\n\n")
                # 空文本行已被插件移除或在此处忽略

            elif item_type == 'image':
                img_path = item.get('img_path', '')
                img_caption = item.get('img_caption', [])
                md_file.write(f"![image]({img_path})\n\n")
                if img_caption:
                    caption_text = " ".join(img_caption)
                    md_file.write(f"{caption_text}{page_info}\n\n")
                elif page_info:  # 如果没有标题但有页码
                    md_file.write(f"{page_info}\n\n")

            elif item_type == 'table':
                table_body = item.get('table_body', '')
                table_body = table_body.replace("<html><body><table>", "").replace("</table></body></html>", "")
                md_file.write(f"{table_body}\n\n")
                if page_info:
                    md_file.write(f"{page_info}\n\n")
            # else: # 可以选择性地处理未知类型
            #     print(f"Warning: Unknown item type encountered: {item_type}")

    print(f"成功生成Markdown文件: {markdown_file_path}") 