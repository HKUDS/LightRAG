# pip install -q -U google-genai to use gemini as a client

import os
from typing import Optional
import dataclasses
from pathlib import Path
import hashlib
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc, Tokenizer
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status
import sentencepiece as spm
import requests

import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

WORKING_DIR = "./dickens"

if os.path.exists(WORKING_DIR):
    import shutil

    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)


class GemmaTokenizer(Tokenizer):
    # adapted from google-cloud-aiplatform[tokenization]

    @dataclasses.dataclass(frozen=True)
    class _TokenizerConfig:
        tokenizer_model_url: str
        tokenizer_model_hash: str

    _TOKENIZERS = {
        "google/gemma2": _TokenizerConfig(
            tokenizer_model_url="https://raw.githubusercontent.com/google/gemma_pytorch/33b652c465537c6158f9a472ea5700e5e770ad3f/tokenizer/tokenizer.model",
            tokenizer_model_hash="61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
        ),
        "google/gemma3": _TokenizerConfig(
            tokenizer_model_url="https://raw.githubusercontent.com/google/gemma_pytorch/cb7c0152a369e43908e769eb09e1ce6043afe084/tokenizer/gemma3_cleaned_262144_v2.spiece.model",
            tokenizer_model_hash="1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        ),
    }

    def __init__(
        self, model_name: str = "gemini-2.0-flash", tokenizer_dir: Optional[str] = None
    ):
        # https://github.com/google/gemma_pytorch/tree/main/tokenizer
        if "1.5" in model_name or "1.0" in model_name:
            # up to gemini 1.5 gemma2 is a comparable local tokenizer
            # https://github.com/googleapis/python-aiplatform/blob/main/vertexai/tokenization/_tokenizer_loading.py
            tokenizer_name = "google/gemma2"
        else:
            # for gemini > 2.0 gemma3 was used
            tokenizer_name = "google/gemma3"

        file_url = self._TOKENIZERS[tokenizer_name].tokenizer_model_url
        tokenizer_model_name = file_url.rsplit("/", 1)[1]
        expected_hash = self._TOKENIZERS[tokenizer_name].tokenizer_model_hash

        tokenizer_dir = Path(tokenizer_dir)
        if tokenizer_dir.is_dir():
            file_path = tokenizer_dir / tokenizer_model_name
            model_data = self._maybe_load_from_cache(
                file_path=file_path, expected_hash=expected_hash
            )
        else:
            model_data = None
        if not model_data:
            model_data = self._load_from_url(
                file_url=file_url, expected_hash=expected_hash
            )
            self.save_tokenizer_to_cache(cache_path=file_path, model_data=model_data)

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.LoadFromSerializedProto(model_data)
        super().__init__(model_name=model_name, tokenizer=tokenizer)

    def _is_valid_model(self, model_data: bytes, expected_hash: str) -> bool:
        """Returns true if the content is valid by checking the hash."""
        return hashlib.sha256(model_data).hexdigest() == expected_hash

    def _maybe_load_from_cache(self, file_path: Path, expected_hash: str) -> bytes:
        """Loads the model data from the cache path."""
        if not file_path.is_file():
            return
        with open(file_path, "rb") as f:
            content = f.read()
        if self._is_valid_model(model_data=content, expected_hash=expected_hash):
            return content

        # Cached file corrupted.
        self._maybe_remove_file(file_path)

    def _load_from_url(self, file_url: str, expected_hash: str) -> bytes:
        """Loads model bytes from the given file url."""
        resp = requests.get(file_url)
        resp.raise_for_status()
        content = resp.content

        if not self._is_valid_model(model_data=content, expected_hash=expected_hash):
            actual_hash = hashlib.sha256(content).hexdigest()
            raise ValueError(
                f"Downloaded model file is corrupted."
                f" Expected hash {expected_hash}. Got file hash {actual_hash}."
            )
        return content

    @staticmethod
    def save_tokenizer_to_cache(cache_path: Path, model_data: bytes) -> None:
        """Saves the model data to the cache path."""
        try:
            if not cache_path.is_file():
                cache_dir = cache_path.parent
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(model_data)
        except OSError:
            # Don't raise if we cannot write file.
            pass

    @staticmethod
    def _maybe_remove_file(file_path: Path) -> None:
        """Removes the file if exists."""
        if not file_path.is_file():
            return
        try:
            file_path.unlink()
        except OSError:
            # Don't raise if we cannot remove file.
            pass

    # def encode(self, content: str) -> list[int]:
    #     return self.tokenizer.encode(content)

    # def decode(self, tokens: list[int]) -> str:
    #     return self.tokenizer.decode(tokens)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 1. Initialize the GenAI Client with your Gemini API Key
    client = genai.Client(api_key=gemini_api_key)

    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
    )

    # 4. Return the response text
    return response.text


async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        # tiktoken_model_name="gpt-4o-mini",
        tokenizer=GemmaTokenizer(
            tokenizer_dir=(Path(WORKING_DIR) / "vertexai_tokenizer_model"),
            model_name="gemini-2.0-flash",
        ),
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    file_path = "story.txt"
    with open(file_path, "r") as file:
        text = file.read()

    rag.insert(text)

    response = rag.query(
        query="What is the main theme of the story?",
        param=QueryParam(mode="hybrid", top_k=5, response_type="single line"),
    )

    print(response)


if __name__ == "__main__":
    main()
