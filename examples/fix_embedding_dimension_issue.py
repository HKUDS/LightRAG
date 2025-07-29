#!/usr/bin/env python3
"""
Fix embedding dimension mismatch issues in LightRAG.

This script helps resolve the common error:
"ValueError: all the input array dimensions except for the concatenation axis must match exactly"

This happens when you switch embedding models with different dimensions.

Usage:
    python examples/fix_embedding_dimension_issue.py [working_dir]
"""

import os
import sys
import shutil
from pathlib import Path


def main():
    """Main function to fix embedding dimension issues."""
    # Get working directory from argument or use default
    if len(sys.argv) > 1:
        working_dir = sys.argv[1]
    else:
        working_dir = "./dickens_xai"  # Default from demo
    
    working_path = Path(working_dir)
    
    print("🔧 LightRAG Embedding Dimension Fix")
    print("="*50)
    
    if not working_path.exists():
        print(f"✅ Working directory '{working_dir}' doesn't exist. No action needed.")
        return
    
    print(f"📁 Working directory: {working_dir}")
    
    # Check for vector database files that might have dimension conflicts
    vector_files = list(working_path.glob("*vector*")) + list(working_path.glob("*embedding*"))
    
    if vector_files:
        print(f"🔍 Found {len(vector_files)} vector/embedding files:")
        for file in vector_files:
            print(f"  - {file.name}")
    
    # Ask user for confirmation
    print("\n⚠️  This will delete the existing working directory and all cached data.")
    print("   You will need to re-process your documents, but this fixes dimension conflicts.")
    
    response = input("\nDo you want to continue? (yes/no): ").lower().strip()
    
    if response == "yes":
        try:
            print(f"\n🧹 Removing working directory: {working_dir}")
            shutil.rmtree(working_dir)
            print("✅ Successfully removed working directory")
            print("\n💡 Next steps:")
            print("   1. Run your LightRAG script again")
            print("   2. Make sure your embedding model configuration is consistent")
            print("   3. The system will rebuild the knowledge graph with correct dimensions")
            
        except Exception as e:
            print(f"❌ Error removing directory: {e}")
            return 1
    else:
        print("❌ Operation cancelled")
        print("\n💡 Alternative solutions:")
        print("   1. Use the same embedding model as before")
        print("   2. Or manually backup important data before cleaning")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())