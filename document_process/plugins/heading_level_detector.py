import re
from .base_plugin import BasePlugin

# --- 从原始脚本移过来的函数 ---
def determine_number_level(text):
    number_pattern = r'^(\d+(\.\d+)*)\s+[\u4e00-\u9fa5]'
    match = re.match(number_pattern, text)
    if match:
        dots = match.group(1).count('.')
        return dots + 1
    return None

def determine_chinese_section_level(text):
    CN_NUM = '一二三四五六七八九十'
    cn_num_pattern = f'[第]([{CN_NUM}]|[{CN_NUM}]?十[{CN_NUM}]?|[{CN_NUM}]百[{CN_NUM}]?十?[{CN_NUM}]?)'
    chapter_pattern = f'{cn_num_pattern}章\\s+'
    section_pattern = f'{cn_num_pattern}节\\s+'
    item_pattern = f'{cn_num_pattern}条\\s+'
    if re.match(chapter_pattern, text): return 1
    if re.match(section_pattern, text): return 2
    if re.match(item_pattern, text): return 2  # 初始判断为2级，后续插件可调整
    return None

def should_ignore_text_level(text):
    cn_num = '一二三四五六七八九十'
    pattern = r'^[（\(]([0-9]+|[' + cn_num + r']+)[）\)]'
    return bool(re.match(pattern, text))

def determine_heading_level(text):
    level = determine_number_level(text)
    if level is not None: return level
    level = determine_chinese_section_level(text)
    if level is not None: return level
    return None
# --- 结束移动的函数 ---

class HeadingLevelDetector(BasePlugin):
    """
    检测文本项的标题级别（基于数字和中文章节模式）。
    """
    plugin_name = "标题级别检测器"
    plugin_description = "检测基于数字(例如'1.1 文本')和中文章节/节/条模式(例如'第一章', '第二节', '第三条')的标题级别。忽略带括号的列表项。"
    applicable_scenarios = "使用标准数字或中文章节/节/条标题的文档。"
    notes = "为'条'项分配初始级别，这些级别可能会被其他插件如ArticleLevelAdjuster调整。除非被忽略，否则不会覆盖已存在的'text_level'。"

    def process(self, data: list) -> list:
        """
        为文本项添加或更新 'text_level' 字段。
        """
        for item in data:
            if item.get('type') == 'text':
                text = item.get('text', '')
                # 只有在需要忽略现有级别，或者根本没有级别时，才进行检测
                if 'text_level' not in item or should_ignore_text_level(text):
                    if should_ignore_text_level(text):
                        # 如果是括号列表项，确保没有级别
                        if 'text_level' in item:
                            del item['text_level']  # 或者设置为 None
                    else:
                        level = determine_heading_level(text)
                        if level is not None:
                            item['text_level'] = level
                        elif 'text_level' in item:
                            # 如果检测不到级别，但之前有（且不是忽略类型），则移除
                            del item['text_level']  # 或者设置为 None

        return data 