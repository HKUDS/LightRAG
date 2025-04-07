from abc import ABC, abstractmethod

class BasePlugin(ABC):
    """
    插件基类，定义所有插件必须实现的接口。
    """
    # 插件元数据 (用于文档和选择)
    plugin_name = "Base Plugin"
    plugin_description = "This is a base class and should not be used directly."
    applicable_scenarios = "N/A"
    notes = "N/A"

    @abstractmethod
    def process(self, data: list) -> list:
        """
        处理文档数据列表。

        Args:
            data (list): 包含文档项（字典）的列表。

        Returns:
            list: 处理后的文档数据列表。
        """
        pass

    def get_metadata(self) -> dict:
        """返回插件的元数据"""
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "scenarios": self.applicable_scenarios,
            "notes": self.notes
        } 