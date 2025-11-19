"""
三语言实体提取器（中文/英文/瑞典语）

使用最佳工具组合：
- 中文: HanLP (F1 95%)
- 英文: spaCy (F1 90%)
- 瑞典语: spaCy (F1 80-85%)

特点：
- 延迟加载（按需加载模型，节省内存）
- 高质量（每种语言使用最佳工具）
- 简单易用
"""

from typing import List, Dict, Literal, Optional
import logging

logger = logging.getLogger(__name__)


class TrilingualEntityExtractor:
    """三语言实体提取器（中/英/瑞典）"""

    def __init__(self):
        """初始化（延迟加载模型）"""
        self._spacy_en = None
        self._spacy_sv = None
        self._hanlp = None

    @property
    def spacy_en(self):
        """延迟加载英文模型"""
        if self._spacy_en is None:
            logger.info("Loading English spaCy model (en_core_web_trf)...")
            try:
                import spacy
                self._spacy_en = spacy.load("en_core_web_trf")
                logger.info("✓ English model loaded successfully")
            except OSError:
                logger.error(
                    "English model not found. Please run: "
                    "python -m spacy download en_core_web_trf"
                )
                raise
        return self._spacy_en

    @property
    def spacy_sv(self):
        """延迟加载瑞典语模型"""
        if self._spacy_sv is None:
            logger.info("Loading Swedish spaCy model (sv_core_news_lg)...")
            try:
                import spacy
                self._spacy_sv = spacy.load("sv_core_news_lg")
                logger.info("✓ Swedish model loaded successfully")
            except OSError:
                logger.error(
                    "Swedish model not found. Please run: "
                    "python -m spacy download sv_core_news_lg"
                )
                raise
        return self._spacy_sv

    @property
    def hanlp(self):
        """延迟加载中文模型"""
        if self._hanlp is None:
            logger.info("Loading Chinese HanLP model...")
            try:
                import hanlp
                self._hanlp = hanlp.load(
                    hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
                )
                logger.info("✓ Chinese model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load HanLP model: {e}")
                raise
        return self._hanlp

    def extract(
        self, text: str, language: Literal["zh", "en", "sv"]
    ) -> List[Dict[str, any]]:
        """提取实体

        Args:
            text: 文本内容
            language: 'zh' (中文), 'en' (英文), 'sv' (瑞典语)

        Returns:
            [{'entity': '...', 'type': '...', 'score': 0.9, 'start': 0, 'end': 5}, ...]

        Raises:
            ValueError: 如果语言不支持
        """
        if language == "zh":
            return self._extract_chinese(text)
        elif language == "en":
            return self._extract_english(text)
        elif language == "sv":
            return self._extract_swedish(text)
        else:
            raise ValueError(
                f"Unsupported language: {language}. " f"Supported: 'zh', 'en', 'sv'"
            )

    def _extract_chinese(self, text: str) -> List[Dict]:
        """提取中文实体（使用 HanLP）

        HanLP 输出格式：
        {
            'tok': [['苹果', '公司'], ...],
            'ner': [['B-ORG', 'I-ORG'], ...]
        }
        """
        result = self.hanlp(text, tasks="ner")

        entities = []
        current_entity = []
        current_type = None
        current_start = 0
        char_position = 0

        # 遍历 token 和 NER 标签
        for tokens, labels in zip(result["tok"], result["ner"]):
            for token, label in zip(tokens, labels):
                if label.startswith("B-"):  # Begin of entity
                    # 保存之前的实体
                    if current_entity:
                        entities.append(
                            {
                                "entity": "".join(current_entity),
                                "type": current_type,
                                "score": 1.0,
                                "start": current_start,
                                "end": char_position,
                            }
                        )

                    # 开始新实体
                    current_entity = [token]
                    current_type = label[2:]  # 去掉 'B-' 前缀
                    current_start = char_position

                elif label.startswith("I-") and current_entity:  # Inside entity
                    current_entity.append(token)

                else:  # O (Outside) or 结束当前实体
                    if current_entity:
                        entities.append(
                            {
                                "entity": "".join(current_entity),
                                "type": current_type,
                                "score": 1.0,
                                "start": current_start,
                                "end": char_position,
                            }
                        )
                        current_entity = []
                        current_type = None

                char_position += len(token)

        # 处理最后一个实体
        if current_entity:
            entities.append(
                {
                    "entity": "".join(current_entity),
                    "type": current_type,
                    "score": 1.0,
                    "start": current_start,
                    "end": char_position,
                }
            )

        return entities

    def _extract_english(self, text: str) -> List[Dict]:
        """提取英文实体（使用 spaCy）"""
        doc = self.spacy_en(text)
        return [
            {
                "entity": ent.text,
                "type": ent.label_,
                "score": 1.0,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    def _extract_swedish(self, text: str) -> List[Dict]:
        """提取瑞典语实体（使用 spaCy）"""
        doc = self.spacy_sv(text)
        return [
            {
                "entity": ent.text,
                "type": ent.label_,
                "score": 1.0,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    def unload_all(self):
        """卸载所有模型（释放内存）"""
        logger.info("Unloading all models to free memory...")
        self._spacy_en = None
        self._spacy_sv = None
        self._hanlp = None
        logger.info("✓ All models unloaded")

    def get_loaded_models(self) -> List[str]:
        """获取当前已加载的模型列表"""
        loaded = []
        if self._spacy_en is not None:
            loaded.append("English (spaCy)")
        if self._spacy_sv is not None:
            loaded.append("Swedish (spaCy)")
        if self._hanlp is not None:
            loaded.append("Chinese (HanLP)")
        return loaded


# 便捷函数
def create_extractor() -> TrilingualEntityExtractor:
    """创建三语言实体提取器实例

    Returns:
        TrilingualEntityExtractor 实例

    Example:
        >>> extractor = create_extractor()
        >>> entities = extractor.extract("Apple Inc. was founded in 1976.", language='en')
        >>> print(entities)
        [{'entity': 'Apple Inc.', 'type': 'ORG', 'score': 1.0, ...}, ...]
    """
    return TrilingualEntityExtractor()
