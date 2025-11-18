import pytest
from lightrag.utils import EmbeddingFunc

def dummy_func(*args, **kwargs):
    pass

def test_embedding_func_with_model_name():
    func = EmbeddingFunc(
        embedding_dim=1536,
        func=dummy_func,
        model_name="text-embedding-ada-002"
    )
    assert func.get_model_identifier() == "text_embedding_ada_002_1536d"

def test_embedding_func_without_model_name():
    func = EmbeddingFunc(
        embedding_dim=768,
        func=dummy_func
    )
    assert func.get_model_identifier() == "unknown_768d"

def test_model_name_sanitization():
    func = EmbeddingFunc(
        embedding_dim=1024,
        func=dummy_func,
        model_name="models/text-embedding-004"  # Contains special chars
    )
    assert func.get_model_identifier() == "models_text_embedding_004_1024d"

def test_model_name_with_uppercase():
    func = EmbeddingFunc(
        embedding_dim=512,
        func=dummy_func,
        model_name="My-Model-V1"
    )
    assert func.get_model_identifier() == "my_model_v1_512d"

