"""
LightRAG Demo with Vietnamese Embedding Model
==============================================

This example demonstrates how to use LightRAG with the Vietnamese_Embedding model
from AITeamVN (https://huggingface.co/AITeamVN/Vietnamese_Embedding)

Model Details:
- Base: BAAI/bge-m3
- Max Sequence Length: 2048 tokens
- Output Dimensions: 1024
- Similarity Function: Dot product
- Language: Vietnamese (also works with other languages)

Setup:
1. Set your HuggingFace token:
   export HUGGINGFACE_API_KEY="your_hf_token"
   or
   export HF_TOKEN="your_hf_token"

2. Set your LLM API key (OpenAI example):
   export OPENAI_API_KEY="your_openai_key"

3. Run the script:
   python examples/vietnamese_embedding_demo.py
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.llm.vietnamese_embed import vietnamese_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger

# Setup logger
setup_logger("lightrag", level="INFO")

# Working directory for storing RAG data
WORKING_DIR = "./vietnamese_rag_storage"

# Ensure working directory exists
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)
    print(f"Created working directory: {WORKING_DIR}")


async def initialize_rag():
    """
    Initialize LightRAG with Vietnamese Embedding model.
    """
    print("\n" + "="*60)
    print("Initializing LightRAG with Vietnamese Embedding Model")
    print("="*60)
    
    # Check for required environment variables
    hf_token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  Warning: HUGGINGFACE_API_KEY or HF_TOKEN not set!")
        print("   The model may still work if it's publicly accessible.")
    else:
        print(f"✓ HuggingFace token found: {hf_token[:10]}...")
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it with: export OPENAI_API_KEY='your-key'"
        )
    print(f"✓ OpenAI API key found: {openai_key[:10]}...")
    
    # Create LightRAG instance with Vietnamese embedding
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,  # Using GPT-4o-mini for text generation
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,  # Vietnamese_Embedding outputs 1024 dimensions
            max_token_size=2048,  # Model was trained with max_length=2048
            func=lambda texts: vietnamese_embed(
                texts,
                model_name="AITeamVN/Vietnamese_Embedding",
                token=hf_token
            )
        ),
    )
    
    print("\n✓ Initializing storage backends...")
    await rag.initialize_storages()
    
    print("✓ Initializing processing pipeline...")
    await initialize_pipeline_status()
    
    print("✓ LightRAG initialization complete!\n")
    return rag


async def demo_vietnamese_text():
    """
    Demo with Vietnamese text content.
    """
    print("\n" + "="*60)
    print("DEMO 1: Vietnamese Text Processing")
    print("="*60 + "\n")
    
    # Initialize RAG
    rag = await initialize_rag()
    
    # Sample Vietnamese text about Vietnam
    vietnamese_text = """
    Việt Nam, tên chính thức là Cộng hòa Xã hội chủ nghĩa Việt Nam, là một quốc gia nằm ở phía đông bán đảo Đông Dương thuộc khu vực Đông Nam Á.
    
    Thủ đô của Việt Nam là Hà Nội, trong khi thành phố lớn nhất là Thành phố Hồ Chí Minh (Sài Gòn).
    
    Việt Nam có dân số khoảng 100 triệu người, là quốc gia đông dân thứ 15 trên thế giới. 
    Việt Nam có nền kinh tế phát triển nhanh, với các ngành công nghiệp chủ đạo bao gồm:
    - Sản xuất điện tử và công nghệ thông tin
    - Du lịch và dịch vụ
    - Nông nghiệp (đặc biệt là xuất khẩu gạo và cà phê)
    - Dệt may và giày da
    
    Văn hóa Việt Nam rất đa dạng với nhiều di sản thế giới được UNESCO công nhận như:
    Vịnh Hạ Long, Phố cổ Hội An, Cố đô Huế, và Thánh địa Mỹ Sơn.
    """
    
    print("Inserting Vietnamese text into RAG system...")
    await rag.ainsert(vietnamese_text)
    print("✓ Text inserted successfully!\n")
    
    # Query in Vietnamese
    queries_vi = [
        "Thủ đô của Việt Nam là gì?",
        "Việt Nam có bao nhiêu dân số?",
        "Những ngành kinh tế chủ đạo của Việt Nam là gì?",
        "Kể tên một số di sản thế giới của Việt Nam?"
    ]
    
    print("Querying in Vietnamese:")
    print("-" * 60)
    
    for query in queries_vi:
        print(f"\n❓ Query: {query}")
        result = await rag.aquery(
            query,
            param=QueryParam(mode="hybrid")
        )
        print(f"💡 Answer: {result}\n")
    
    # Clean up
    await rag.finalize_storages()


async def demo_english_text():
    """
    Demo with English text (the model also works with other languages).
    """
    print("\n" + "="*60)
    print("DEMO 2: English Text Processing (Multilingual Support)")
    print("="*60 + "\n")
    
    # Initialize RAG
    rag = await initialize_rag()
    
    # Sample English text about AI
    english_text = """
    Artificial Intelligence (AI) is transforming the world in unprecedented ways. 
    
    Machine Learning, a subset of AI, enables computers to learn from data without explicit programming.
    Deep Learning uses neural networks with multiple layers to process complex patterns.
    
    Natural Language Processing (NLP) allows machines to understand and generate human language.
    Popular NLP models include:
    - GPT (Generative Pre-trained Transformer) for text generation
    - BERT (Bidirectional Encoder Representations from Transformers) for understanding
    - T5 (Text-to-Text Transfer Transformer) for various NLP tasks
    
    Computer Vision enables machines to interpret visual information from the world.
    Applications include facial recognition, object detection, and image classification.
    
    AI is being applied in various domains such as healthcare, finance, transportation, and education.
    """
    
    print("Inserting English text into RAG system...")
    await rag.ainsert(english_text)
    print("✓ Text inserted successfully!\n")
    
    # Query in English
    queries_en = [
        "What is Machine Learning?",
        "Name some popular NLP models.",
        "What are the applications of Computer Vision?",
        "In which domains is AI being applied?"
    ]
    
    print("Querying in English:")
    print("-" * 60)
    
    for query in queries_en:
        print(f"\n❓ Query: {query}")
        result = await rag.aquery(
            query,
            param=QueryParam(mode="hybrid")
        )
        print(f"💡 Answer: {result}\n")
    
    # Clean up
    await rag.finalize_storages()


async def demo_mixed_languages():
    """
    Demo with mixed Vietnamese and English content.
    """
    print("\n" + "="*60)
    print("DEMO 3: Mixed Language Processing")
    print("="*60 + "\n")
    
    # Initialize RAG
    rag = await initialize_rag()
    
    # Mixed content
    mixed_text = """
    # Công nghệ AI tại Việt Nam (AI Technology in Vietnam)
    
    Việt Nam đang phát triển mạnh mẽ trong lĩnh vực Trí tuệ Nhân tạo (AI).
    Vietnam is rapidly developing in the field of Artificial Intelligence.
    
    ## Các công ty AI hàng đầu (Leading AI Companies):
    
    1. VinAI Research
       - Nghiên cứu về Machine Learning và Computer Vision
       - Research in Machine Learning and Computer Vision
       - Phát triển các mô hình ngôn ngữ tiếng Việt
       - Developing Vietnamese language models
    
    2. FPT AI Center
       - Tập trung vào AI applications cho doanh nghiệp
       - Focus on AI applications for enterprises
       - Smart city solutions và IoT
    
    3. VNG AI Center
       - Phát triển chatbots và virtual assistants
       - Natural Language Processing for Vietnamese
       - Game AI và recommendation systems
    
    ## Ứng dụng AI (AI Applications):
    - Healthcare: Chẩn đoán hình ảnh y khoa / Medical image diagnosis
    - Education: Personalized learning và adaptive testing
    - Finance: Fraud detection và risk assessment
    - E-commerce: Product recommendations và customer service bots
    """
    
    print("Inserting mixed language text into RAG system...")
    await rag.ainsert(mixed_text)
    print("✓ Text inserted successfully!\n")
    
    # Query in both languages
    queries_mixed = [
        "Những công ty AI hàng đầu tại Việt Nam là gì?",  # Vietnamese
        "What does VinAI Research focus on?",  # English
        "AI được ứng dụng trong lĩnh vực nào?",  # Vietnamese
        "Which company develops chatbots and virtual assistants?"  # English
    ]
    
    print("Querying in mixed languages:")
    print("-" * 60)
    
    for query in queries_mixed:
        print(f"\n❓ Query: {query}")
        result = await rag.aquery(
            query,
            param=QueryParam(mode="hybrid")
        )
        print(f"💡 Answer: {result}\n")
    
    # Clean up
    await rag.finalize_storages()


async def main():
    """
    Main function to run all demos.
    """
    print("\n" + "="*60)
    print("Vietnamese Embedding Model Demo for LightRAG")
    print("Model: AITeamVN/Vietnamese_Embedding")
    print("="*60)
    
    try:
        # Run Demo 1: Vietnamese text
        await demo_vietnamese_text()
        
        # Run Demo 2: English text
        await demo_english_text()
        
        # Run Demo 3: Mixed languages
        await demo_mixed_languages()
        
        print("\n" + "="*60)
        print("✓ All demos completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
