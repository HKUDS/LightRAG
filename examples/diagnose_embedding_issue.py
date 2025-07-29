#!/usr/bin/env python3
"""
Diagnostic script to identify embedding dimension issues in LightRAG.

This script helps diagnose the exact cause of embedding dimension conflicts
without requiring API keys or running full demos.

Usage:
    python examples/diagnose_embedding_issue.py [working_directory]
"""

import sys
import json
from pathlib import Path
import asyncio


def check_working_directory(working_dir="./dickens_xai"):
    """Check the working directory for existing data."""
    working_path = Path(working_dir)
    
    print(f"🔍 Checking working directory: {working_dir}")
    
    if not working_path.exists():
        print("✅ Working directory doesn't exist - no conflicts possible")
        return True
    
    # Check for vector database files
    vector_files = []
    for pattern in ["*vector*", "*embed*", "*.npy", "*.json"]:
        vector_files.extend(working_path.glob(pattern))
    
    if not vector_files:
        print("✅ No vector/embedding files found")
        return True
    
    print(f"📁 Found {len(vector_files)} potential data files:")
    for file in vector_files:
        print(f"  - {file.name} ({file.stat().st_size} bytes)")
    
    # Check if there's JSON data that might contain dimension info
    for file in vector_files:
        if file.suffix == '.json':
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        print(f"  📄 {file.name} contains {len(data)} entries")
                        # Look for dimension clues
                        for key, value in list(data.items())[:3]:  # Check first 3 entries
                            if isinstance(value, dict) and 'embedding' in str(value).lower():
                                print(f"    - Found embedding-related data in key: {key}")
            except Exception as e:
                print(f"  ⚠️  Could not read {file.name}: {e}")
    
    return False


async def test_ollama_embedding():
    """Test Ollama embedding to get actual dimensions."""
    try:
        print("\n🧪 Testing Ollama embedding dimensions...")
        
        # Import here to avoid issues if not available
        from lightrag.llm.ollama import ollama_embed
        
        models_to_test = [
            ("bge-m3:latest", "http://localhost:11434"),
            ("nomic-embed-text:latest", "http://localhost:11434"),
        ]
        
        results = {}
        
        for model, host in models_to_test:
            try:
                print(f"  Testing {model}...")
                result = await ollama_embed(
                    ["test text for dimension check"], 
                    embed_model=model,
                    host=host
                )
                dimension = result.shape[1]
                results[model] = dimension
                print(f"  ✅ {model}: {dimension} dimensions")
                
            except Exception as e:
                print(f"  ❌ {model}: {str(e)}")
                results[model] = f"Error: {str(e)}"
        
        return results
        
    except ImportError as e:
        print(f"❌ Cannot import embedding functions: {e}")
        return {}
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
        return {}


def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    try:
        import httpx
        print("\n🌐 Checking Ollama connection...")
        
        response = httpx.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"✅ Ollama is running with {len(models)} models:")
            
            embedding_models = []
            for model in models:
                name = model.get("name", "unknown")
                if any(embed_name in name.lower() for embed_name in ["embed", "bge", "nomic"]):
                    embedding_models.append(name)
                    print(f"  📊 {name}")
            
            if not embedding_models:
                print("  ⚠️  No embedding models found")
                print("  💡 Install one: ollama pull bge-m3:latest")
            
            return True, embedding_models
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False, []
            
    except ImportError:
        print("❌ httpx not available - cannot check Ollama")
        return False, []
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure Ollama is running: systemctl start ollama")
        return False, []


def provide_recommendations(working_dir, has_data, embedding_results):
    """Provide specific recommendations based on findings."""
    print("\n" + "="*60)
    print("🎯 RECOMMENDATIONS")
    print("="*60)
    
    if has_data:
        print("🧹 ISSUE: Existing data detected in working directory")
        print("   This is likely causing the dimension conflict.")
        print()
        print("✅ SOLUTION: Clean the working directory")
        print(f"   rm -rf {working_dir}")
        print("   Then run your LightRAG script again.")
        print()
        print("💡 Why this works:")
        print("   - Removes old embeddings with different dimensions")
        print("   - Forces fresh start with consistent model")
        print("   - Eliminates any cached incompatible data")
    
    if embedding_results:
        print("\n📊 EMBEDDING MODEL INFO:")
        for model, result in embedding_results.items():
            if isinstance(result, int):
                print(f"   - {model}: {result} dimensions")
            else:
                print(f"   - {model}: {result}")
        
        # Check for dimension consistency
        dims = [r for r in embedding_results.values() if isinstance(r, int)]
        if len(set(dims)) > 1:
            print("\n⚠️  WARNING: Multiple models with different dimensions detected!")
            print("   Make sure to use the same model consistently.")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Clean working directory (if needed)")
    print("2. Set your API key: export XAI_API_KEY='your-key'")
    print("3. Run: python examples/lightrag_xai_demo_robust.py")
    print()
    print("📚 For more help: cat TROUBLESHOOTING_XAI.md")


async def main():
    """Main diagnostic function."""
    print("🔧 LightRAG Embedding Dimension Diagnostic")
    print("="*50)
    
    # Get working directory from command line or use default
    working_dir = sys.argv[1] if len(sys.argv) > 1 else "./dickens_xai"
    
    # Check working directory
    has_data = not check_working_directory(working_dir)
    
    # Check Ollama connection
    ollama_ok, models = check_ollama_connection()
    
    # Test embedding dimensions if Ollama is available
    embedding_results = {}
    if ollama_ok:
        embedding_results = await test_ollama_embedding()
    
    # Provide recommendations
    provide_recommendations(working_dir, has_data, embedding_results)


if __name__ == "__main__":
    asyncio.run(main())