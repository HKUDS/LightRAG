import pipmaster as pm  # Pipmaster for dynamic library install

if not pm.is_installed("sentence_transformers"):
    pm.install("sentence_transformers")
if not pm.is_installed("numpy"):
    pm.install("numpy")

import numpy as np
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer


async def sentence_transformers_embed(
    texts: list[str], model: SentenceTransformer
) -> np.ndarray:
    async def inner_encode(
        texts: list[str], model: SentenceTransformer, embedding_dim: int = 1024
    ):
        return model.encode(
            texts,
            truncate_dim=embedding_dim,
            convert_to_numpy=True,
            convert_to_tensor=False,
            show_progress_bar=False,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=model.get_sentence_embedding_dimension(),
        func=inner_encode,
        max_token_size=model.get_max_seq_length(),
    )
    return await embedding_func(texts, model=model)
