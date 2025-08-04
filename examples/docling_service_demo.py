#!/usr/bin/env python3
"""
LightRAG Docling Service Integration Demo

This script demonstrates the new Docling service integration capabilities.
"""

import asyncio
import os
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_service_discovery():
    """Demonstrate service discovery functionality."""
    print("\n🔍 Testing Service Discovery...")
    
    try:
        from lightrag.docling_client.service_discovery import service_discovery
        
        # Get service information
        service_info = await service_discovery.get_service_info()
        
        print(f"Service configured: {service_info['configured']}")
        print(f"Service URL: {service_info['url']}")
        print(f"Service available: {service_info['available']}")
        
        if service_info.get('error'):
            print(f"Error: {service_info['error']}")
        
        if service_info['available'] and service_info.get('config'):
            config = service_info['config']
            print(f"Service version: {config.get('version', 'unknown')}")
            print(f"Supported formats: {config.get('supported_formats', [])}")
    
    except Exception as e:
        print(f"Error testing service discovery: {e}")


async def demo_fallback_processing():
    """Demonstrate fallback processing."""
    print("\n🔄 Testing Fallback Processing...")
    
    try:
        from lightrag.docling_client.fallback import fallback_processor
        
        # Create a test text file
        test_content = "This is a test document for fallback processing.\n\nIt demonstrates basic text extraction."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = Path(f.name)
        
        try:
            # Process with fallback
            result = await fallback_processor.process_document_with_fallback(temp_path)
            
            print(f"Processing success: {result['success']}")
            print(f"Content length: {len(result['content'])}")
            print(f"Processor used: {result['metadata'].get('processor', 'unknown')}")
            print(f"Processing time: {result['metadata'].get('processing_time_seconds', 0):.2f}s")
            
            if result['success']:
                print("✅ Fallback processing working correctly")
            else:
                print(f"❌ Fallback processing failed: {result.get('error', 'Unknown error')}")
        
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
    
    except Exception as e:
        print(f"Error testing fallback processing: {e}")


async def demo_docling_client():
    """Demonstrate Docling client functionality."""
    print("\n🤖 Testing Docling Client...")
    
    try:
        from lightrag.docling_client import DoclingClient
        
        client = DoclingClient()
        
        try:
            # Check if service is available
            available = await client.is_service_available()
            print(f"Service available: {available}")
            
            if available:
                # Get supported formats
                formats = await client.get_supported_formats()
                print(f"Supported formats: {formats}")
                
                # Get cache stats
                cache_stats = await client.get_cache_stats()
                print(f"Cache enabled: {cache_stats.get('cache_enabled', False)}")
                
                print("✅ Docling client working correctly")
            else:
                print("⚠️  Docling service not available")
        
        finally:
            await client.close()
    
    except Exception as e:
        print(f"Error testing Docling client: {e}")


async def demo_enhanced_processing():
    """Demonstrate enhanced document processing."""
    print("\n⚡ Testing Enhanced Processing...")
    
    try:
        from lightrag.api.routers.document_processing import docling_processor
        
        # Create test files
        test_files = []
        
        # Text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Sample text document for testing.")
            test_files.append(Path(f.name))
        
        # PDF file (minimal)
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n32\n%%EOF"
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(pdf_content)
            test_files.append(Path(f.name))
        
        try:
            for test_file in test_files:
                print(f"\nProcessing {test_file.suffix} file...")
                
                # Check processing strategy
                use_service, reason = await docling_processor.should_use_docling_service(test_file)
                print(f"Use service: {use_service} (reason: {reason})")
                
                # Process document
                result = await docling_processor.process_document(test_file)
                
                print(f"Success: {result['success']}")
                if result['success']:
                    print(f"Content length: {len(result['content'])}")
                    metadata = result.get('metadata', {})
                    print(f"Processor: {metadata.get('processor', 'unknown')}")
                    print(f"Service used: {metadata.get('docling_service_used', False)}")
                    print(f"Processing time: {metadata.get('processing_time_seconds', 0):.2f}s")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
        
        finally:
            # Cleanup
            for test_file in test_files:
                if test_file.exists():
                    test_file.unlink()
    
    except Exception as e:
        print(f"Error testing enhanced processing: {e}")


async def demo_configuration():
    """Show configuration options."""
    print("\n⚙️  Configuration Options...")
    
    config_vars = [
        "DOCLING_SERVICE_URL",
        "DOCLING_SERVICE_MODE", 
        "DOCLING_FALLBACK_ENABLED",
        "LIGHTRAG_ENHANCED_PROCESSING",
        "DOCLING_SERVICE_TIMEOUT",
        "DOCLING_SERVICE_RETRIES"
    ]
    
    print("Current environment configuration:")
    for var in config_vars:
        value = os.getenv(var, "not set")
        print(f"  {var}: {value}")


async def main():
    """Run all demonstrations."""
    print("🚀 LightRAG Docling Service Integration Demo")
    print("=" * 50)
    
    await demo_configuration()
    await demo_service_discovery()
    await demo_docling_client()  
    await demo_fallback_processing()
    await demo_enhanced_processing()
    
    print("\n✨ Demo completed!")
    print("\n🚀 Quick Setup Guide:")
    print("=" * 50)
    print("\n📋 Option 1: Automated startup (Recommended)")
    print("   ./scripts/start-enhanced-processing.sh")
    print("\n📋 Option 2: Manual startup")
    print("   1. Start services: docker compose -f docker-compose.yml -f docker-compose.enhanced.yml --profile enhanced-processing up -d")
    print("   2. Wait for services to be ready (may take several minutes for first-time build)")
    print("\n📋 Option 3: Basic LightRAG (without Docling)")
    print("   docker compose up lightrag postgres -d")
    print("\n🔧 Environment Configuration:")
    print("   DOCLING_SERVICE_URL=http://localhost:8080")
    print("   LIGHTRAG_ENHANCED_PROCESSING=true")
    print("   DOCLING_SERVICE_MODE=auto")
    print("\n🧪 Test the integration:")
    print("   python examples/docling_service_demo.py")
    print("\n📊 Monitor services:")
    print("   docker compose logs -f lightrag")
    print("   docker compose logs -f docling-service")


if __name__ == "__main__":
    asyncio.run(main())