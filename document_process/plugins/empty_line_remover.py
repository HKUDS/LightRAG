from .base_plugin import BasePlugin

class EmptyLineRemover(BasePlugin):
    """
    移除文档数据中空的文本项。
    """
    plugin_name = "空行移除器"
    plugin_description = "移除只包含空白字符的文本项。"
    applicable_scenarios = "大多数文档的通用清理。"
    notes = "如果有意的空行被表示为单独的文本项，可能会被移除。"

    def process(self, data: list) -> list:
        """
        过滤掉空的文本项。
        """
        processed_data = []
        for item in data:
            if item.get('type') == 'text':
                if item.get('text', '').strip():  # 检查文本去除空白后是否还有内容
                    processed_data.append(item)
            else:
                # 保留非文本项
                processed_data.append(item)
        return processed_data 