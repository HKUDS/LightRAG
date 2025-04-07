import re
from .base_plugin import BasePlugin

class ArticleLevelAdjuster(BasePlugin):
    """
    调整"第X条"类型标题的级别。如果文档结构包含"章"和"节"，则"条"为3级；
    如果只有"章"没有"节"，则"条"为2级。
    """
    plugin_name = "条目级别调整器"
    plugin_description = "根据同一章中是否存在'第X节'调整'第X条'项的标题级别。"
    applicable_scenarios = "具有'章', '节', '条'层次结构的文档。"
    notes = "需要前置插件（如HeadingLevelDetector）已分配初始级别。假设'章'项为1级，'节'项为2级。"

    def process(self, data: list) -> list:
        CN_NUM = '一二三四五六七八九十'
        cn_num_pattern = f'[第]([{CN_NUM}]|[{CN_NUM}]?十[{CN_NUM}]?|[{CN_NUM}]百[{CN_NUM}]?十?[{CN_NUM}]?)'
        chapter_pattern = f'{cn_num_pattern}章\\s+'
        section_pattern = f'{cn_num_pattern}节\\s+'
        item_pattern = f'{cn_num_pattern}条\\s+'

        chapter_has_section = {}
        current_chapter_key = None

        # 1. 扫描确定哪些章包含节
        for item in data:
            if item.get('type') == 'text':
                text = item.get('text', '')
                level = item.get('text_level')

                if level == 1 and re.match(chapter_pattern, text):
                    current_chapter_key = text  # 使用章节标题作为键
                    if current_chapter_key not in chapter_has_section:
                        chapter_has_section[current_chapter_key] = False
                elif level == 2 and re.match(section_pattern, text) and current_chapter_key:
                    chapter_has_section[current_chapter_key] = True

        # 2. 调整"条"的级别
        current_chapter_key = None
        for item in data:
            if item.get('type') == 'text':
                text = item.get('text', '')
                level = item.get('text_level')

                # 更新当前所在的章
                if level == 1 and re.match(chapter_pattern, text):
                    current_chapter_key = text

                # 检查是否是"条"并且需要调整
                # 注意：这里假设 HeadingLevelDetector 已将"条"设为 2 级
                if level == 2 and re.match(item_pattern, text) and current_chapter_key:
                    if chapter_has_section.get(current_chapter_key, False):
                        item['text_level'] = 3  # 如果当前章有节，条降为3级

        return data 