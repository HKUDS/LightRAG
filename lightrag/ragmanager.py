class RAGManager:
    _instance = None
    _rag = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_rag(cls, rag_instance):
        cls._rag = rag_instance

    @classmethod
    def get_rag(cls):
        if cls._rag is None:
            raise ValueError("RAG instance not initialized!")
        return cls._rag
