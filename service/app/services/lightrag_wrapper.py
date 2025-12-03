import os
import json
from typing import Dict, Any
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from app.core.config import settings

class LightRAGWrapper:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LightRAGWrapper, cls).__new__(cls)
            cls._instance.rag = None
            cls._instance.initialized = False
        return cls._instance

    async def initialize(self):
        """Initialize LightRAG engine"""
        if self.initialized:
            return

        if not os.path.exists(settings.LIGHTRAG_WORKING_DIR):
            os.makedirs(settings.LIGHTRAG_WORKING_DIR)
            
        self.rag = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,
            # Add other configurations as needed
        )
        # await self.rag.initialize_storages() # Uncomment if needed based on LightRAG version
        self.initialized = True
        print("LightRAG Initialized Successfully")

    async def query(self, query_text: str, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Execute query against LightRAG.
        """
        if not self.rag:
            await self.initialize()

        param = QueryParam(
            mode=mode,
            only_need_context=False,
            response_type="Multiple Paragraphs" 
        )
        
        # Execute query
        # Note: Depending on LightRAG version, this might be sync or async. 
        # Assuming async based on plan.
        try:
            result = await self.rag.aquery(query_text, param=param)
        except AttributeError:
             # Fallback to sync if aquery not available
            result = self.rag.query(query_text, param=param)
        
        return self._parse_lightrag_response(result)

    def _parse_lightrag_response(self, raw_response: Any) -> Dict[str, Any]:
        """
        Parse raw response from LightRAG into a structured format.
        """
        # This logic depends heavily on the actual return format of LightRAG.
        # Assuming it returns a string or a specific object.
        # For now, we'll assume it returns a string that might contain the answer.
        # In a real scenario, we'd inspect 'raw_response' type.
        
        answer = str(raw_response)
        references = [] # Placeholder for references extraction logic
        
        # If LightRAG returns an object with context, extract it here.
        # For example:
        # if isinstance(raw_response, dict):
        #     answer = raw_response.get("response", "")
        #     references = raw_response.get("context", [])
            
        return {
            "answer": answer,
            "references": references
        }
