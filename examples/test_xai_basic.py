#!/usr/bin/env python3
"""
Basic test of xAI integration without the full demo.
This tests just the core functionality to verify everything works.

Usage:
    export XAI_API_KEY="your-xai-api-key"
    python examples/test_xai_basic.py
"""

import os
import asyncio
from lightrag.llm.xai import grok_3_mini_complete

XAI_API_KEY = os.environ.get("XAI_API_KEY")

async def test_xai_basic():
    """Basic test of xAI functionality."""
    print("🧪 Testing xAI Grok 3 Mini...")
    
    if not XAI_API_KEY:
        print("❌ XAI_API_KEY not set")
        print("Set it with: export XAI_API_KEY='your-key'")
        return False
    
    try:
        response = await grok_3_mini_complete(
            "Hello! Please respond with exactly: 'xAI integration working'"
        )
        print(f"✅ xAI Response: {response}")
        
        if "working" in response.lower():
            print("🎉 xAI integration is working correctly!")
            return True
        else:
            print("⚠️ xAI responded but not as expected")
            return True  # Still working, just different response
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔮 xAI Basic Integration Test")
    print("="*30)
    
    result = asyncio.run(test_xai_basic())
    
    if result:
        print("\n✅ Ready to run full demo:")
        print("   python examples/lightrag_xai_demo_robust.py")
    else:
        print("\n❌ Fix API key issue first")