"""
Test script for Vietnamese Embedding Integration

This script tests the Vietnamese_Embedding model integration with LightRAG.

Usage:
    export HUGGINGFACE_API_KEY="your_token"
    export OPENAI_API_KEY="your_openai_key"
    python tests/test_vietnamese_embedding_integration.py
"""

import os
import sys
import asyncio
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.llm.vietnamese_embed import vietnamese_embed, vietnamese_embedding_func
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc


def test_environment_setup():
    """Test 1: Check environment variables"""
    print("\n" + "="*60)
    print("Test 1: Environment Setup")
    print("="*60)
    
    hf_token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    assert hf_token is not None, "❌ HUGGINGFACE_API_KEY or HF_TOKEN not set"
    print(f"✓ HuggingFace token found: {hf_token[:10]}...")
    
    assert openai_key is not None, "❌ OPENAI_API_KEY not set"
    print(f"✓ OpenAI API key found: {openai_key[:10]}...")
    
    print("✓ Environment setup complete")


async def test_basic_embedding():
    """Test 2: Basic embedding generation"""
    print("\n" + "="*60)
    print("Test 2: Basic Embedding Generation")
    print("="*60)
    
    # Test with Vietnamese text
    texts_vi = ["Xin chào", "Việt Nam", "Trí tuệ nhân tạo"]
    embeddings_vi = await vietnamese_embed(texts_vi)
    
    assert embeddings_vi.shape == (3, 1024), f"❌ Wrong shape: {embeddings_vi.shape}"
    print(f"✓ Vietnamese embeddings shape: {embeddings_vi.shape}")
    
    # Test with English text
    texts_en = ["Hello", "Vietnam", "Artificial Intelligence"]
    embeddings_en = await vietnamese_embed(texts_en)
    
    assert embeddings_en.shape == (3, 1024), f"❌ Wrong shape: {embeddings_en.shape}"
    print(f"✓ English embeddings shape: {embeddings_en.shape}")
    
    # Verify embeddings are normalized (for dot product similarity)
    norms = np.linalg.norm(embeddings_vi, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "❌ Embeddings not normalized"
    print(f"✓ Embeddings are normalized (norms: {norms})")
    
    print("✓ Basic embedding generation test passed")


async def test_convenience_function():
    """Test 3: Convenience function"""
    print("\n" + "="*60)
    print("Test 3: Convenience Function")
    print("="*60)
    
    texts = ["Test text 1", "Test text 2"]
    embeddings = await vietnamese_embedding_func(texts)
    
    assert embeddings.shape == (2, 1024), f"❌ Wrong shape: {embeddings.shape}"
    print(f"✓ Convenience function works: {embeddings.shape}")
    
    print("✓ Convenience function test passed")


async def test_lightrag_integration():
    """Test 4: LightRAG integration"""
    print("\n" + "="*60)
    print("Test 4: LightRAG Integration")
    print("="*60)
    
    # Create temporary working directory
    working_dir = "./test_vietnamese_rag_storage"
    os.makedirs(working_dir, exist_ok=True)
    
    try:
        # Get token
        hf_token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
        
        # Initialize LightRAG
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=2048,
                func=lambda texts: vietnamese_embed(
                    texts,
                    token=hf_token
                )
            ),
        )
        
        print("✓ LightRAG instance created")
        
        # Initialize storage
        await rag.initialize_storages()
        print("✓ Storage initialized")
        
        # Initialize pipeline
        await initialize_pipeline_status()
        print("✓ Pipeline initialized")
        
        # Insert test text
        test_text = """
        Việt Nam là một quốc gia ở Đông Nam Á.
        Thủ đô của Việt Nam là Hà Nội.
        Thành phố lớn nhất là Thành phố Hồ Chí Minh.
        """
        
        await rag.ainsert(test_text)
        print("✓ Text inserted successfully")
        
        # Query test
        query = "Thủ đô của Việt Nam là gì?"
        result = await rag.aquery(
            query,
            param=QueryParam(mode="naive")
        )
        
        assert result is not None and len(result) > 0, "❌ Empty query result"
        print("✓ Query executed successfully")
        print(f"  Query: {query}")
        print(f"  Result length: {len(result)} chars")
        
        # Clean up
        await rag.finalize_storages()
        print("✓ Storage finalized")
        
        print("✓ LightRAG integration test passed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up test directory
        import shutil
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
            print("✓ Test directory cleaned up")


async def test_batch_processing():
    """Test 5: Batch processing"""
    print("\n" + "="*60)
    print("Test 5: Batch Processing")
    print("="*60)
    
    # Test with different batch sizes
    batch_sizes = [1, 5, 10, 20]
    
    for batch_size in batch_sizes:
        texts = [f"Sample text {i}" for i in range(batch_size)]
        embeddings = await vietnamese_embed(texts)
        
        assert embeddings.shape == (batch_size, 1024), \
            f"❌ Wrong shape for batch size {batch_size}: {embeddings.shape}"
        
        print(f"✓ Batch size {batch_size}: {embeddings.shape}")
    
    print("✓ Batch processing test passed")


async def test_long_text_handling():
    """Test 6: Long text handling"""
    print("\n" + "="*60)
    print("Test 6: Long Text Handling")
    print("="*60)
    
    # Generate a long text (exceeding 2048 tokens)
    long_text = " ".join(["Việt Nam là một quốc gia đẹp."] * 500)
    
    try:
        embeddings = await vietnamese_embed([long_text])
        assert embeddings.shape == (1, 1024), f"❌ Wrong shape: {embeddings.shape}"
        print(f"✓ Long text handled (truncated): {embeddings.shape}")
        print("✓ Long text handling test passed")
    except Exception as e:
        print(f"❌ Long text handling failed: {e}")
        raise


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Vietnamese Embedding Integration Test Suite")
    print("="*60)
    
    try:
        # Test 1: Environment
        test_environment_setup()
        
        # Test 2: Basic embedding
        await test_basic_embedding()
        
        # Test 3: Convenience function
        await test_convenience_function()
        
        # Test 4: LightRAG integration
        await test_lightrag_integration()
        
        # Test 5: Batch processing
        await test_batch_processing()
        
        # Test 6: Long text handling
        await test_long_text_handling()
        
        print("\n" + "="*60)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌❌❌ TESTS FAILED ❌❌❌")
        print(f"Error: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
