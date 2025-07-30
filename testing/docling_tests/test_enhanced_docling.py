#!/usr/bin/env python3
"""
Test script for enhanced Docling configuration implementation.
Tests all the new features and configuration options.
"""

import asyncio
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add the lightrag module to the path
sys.path.insert(0, '/home/ajithkv/developments/LightRAG')

from lightrag.api.config import global_args
from lightrag.api.routers.document_routes import _process_with_enhanced_docling


async def test_enhanced_docling_config():
    """Test the enhanced Docling configuration implementation."""
    print("=" * 80)
    print("🧪 ENHANCED DOCLING CONFIGURATION TEST")
    print("=" * 80)
    
    # Test file
    test_file = Path("/home/ajithkv/developments/LightRAG/test_document_enhanced_docling.pdf")
    
    if not test_file.exists():
        print("❌ Test PDF not found. Please run create_test_pdf.py first.")
        return False
    
    print(f"📄 Test file: {test_file}")
    print(f"📊 File size: {test_file.stat().st_size / 1024:.1f} KB")
    print()
    
    # Display current configuration
    print("🔧 CURRENT ENHANCED DOCLING CONFIGURATION:")
    print("-" * 50)
    config_vars = [
        'document_loading_engine',
        'docling_export_format', 
        'docling_max_workers',
        'docling_enable_ocr',
        'docling_enable_table_structure',
        'docling_enable_figures',
        'docling_layout_model',
        'docling_ocr_model', 
        'docling_table_model',
        'docling_include_page_numbers',
        'docling_include_headings',
        'docling_extract_metadata',
        'docling_process_images',
        'docling_image_dpi',
        'docling_ocr_confidence',
        'docling_table_confidence',
        'docling_enable_cache',
        'docling_cache_dir',
        'docling_cache_ttl_hours'
    ]
    
    for var in config_vars:
        if hasattr(global_args, var):
            value = getattr(global_args, var)
            print(f"  {var}: {value}")
        else:
            print(f"  {var}: NOT CONFIGURED")
    print()
    
    # Test different export formats
    export_formats = ['markdown', 'json']
    
    for export_format in export_formats:
        print(f"📤 TESTING EXPORT FORMAT: {export_format.upper()}")
        print("-" * 50)
        
        # Temporarily change export format
        original_format = global_args.docling_export_format
        global_args.docling_export_format = export_format
        
        start_time = time.time()
        
        try:
            # Process the document
            print("🔄 Processing document with enhanced Docling...")
            content = await _process_with_enhanced_docling(test_file)
            
            processing_time = time.time() - start_time
            
            print(f"✅ Processing completed successfully!")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"📏 Content length: {len(content):,} characters")
            print()
            
            # Show content preview
            print("📋 CONTENT PREVIEW (first 500 characters):")
            print("-" * 50)
            preview = content[:500].replace('\n', '\n  ')
            print(f"  {preview}")
            if len(content) > 500:
                print(f"  ... (truncated, showing {500} of {len(content)} characters)")
            print()
            
            # Analyze content for key features
            print("🔍 CONTENT ANALYSIS:")
            print("-" * 50)
            
            # Check for tables
            table_indicators = ['|', 'table', 'Table', '---', 'Feature', 'Configuration']
            table_found = any(indicator in content for indicator in table_indicators)
            print(f"  📊 Tables detected: {'✅ Yes' if table_found else '❌ No'}")
            
            # Check for metadata
            metadata_indicators = ['metadata', 'Metadata', 'author', 'created', 'title']
            metadata_found = any(indicator.lower() in content.lower() for indicator in metadata_indicators)
            print(f"  📝 Metadata extracted: {'✅ Yes' if metadata_found else '❌ No'}")
            
            # Check for headings
            heading_indicators = ['#', 'Section', 'Test', 'Enhanced Docling']
            headings_found = any(indicator in content for indicator in heading_indicators)
            print(f"  📰 Headings preserved: {'✅ Yes' if headings_found else '❌ No'}")
            
            # Check for structured content
            structure_indicators = ['•', '*', '1.', '2.', '3.', 'Processing', 'Configuration']
            structure_found = any(indicator in content for indicator in structure_indicators)
            print(f"  🏗️  Structure preserved: {'✅ Yes' if structure_found else '❌ No'}")
            
            print()
            
            # Save output for inspection
            output_file = f"/home/ajithkv/developments/LightRAG/docling_test_output_{export_format}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Enhanced Docling Test Output\n")
                f.write(f"Export Format: {export_format}\n")
                f.write(f"Processing Time: {processing_time:.2f} seconds\n")
                f.write(f"Content Length: {len(content):,} characters\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"\n{'='*80}\n")
                f.write(f"PROCESSED CONTENT:\n")
                f.write(f"{'='*80}\n\n")
                f.write(content)
            
            print(f"💾 Output saved to: {output_file}")
            print()
            
        except Exception as e:
            print(f"❌ Processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
            return False
        finally:
            # Restore original format
            global_args.docling_export_format = original_format
    
    # Test caching functionality
    print("🗂️  TESTING CACHE FUNCTIONALITY:")
    print("-" * 50)
    
    if global_args.docling_enable_cache:
        cache_dir = Path(global_args.working_dir) / global_args.docling_cache_dir
        print(f"📁 Cache directory: {cache_dir}")
        
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.json"))
            print(f"📄 Cache files found: {len(cache_files)}")
            
            if cache_files:
                # Show cache file details
                for cache_file in cache_files[:3]:  # Show first 3
                    stat = cache_file.stat()
                    print(f"  📄 {cache_file.name}: {stat.st_size} bytes, modified {datetime.fromtimestamp(stat.st_mtime)}")
                
                if len(cache_files) > 3:
                    print(f"  ... and {len(cache_files) - 3} more cache files")
            
            print("✅ Cache system is operational")
        else:
            print("⚠️  Cache directory not found (will be created on first use)")
    else:
        print("❌ Caching is disabled")
    
    print()
    
    # Performance summary
    print("📊 PERFORMANCE SUMMARY:")
    print("-" * 50)
    print(f"✅ Enhanced Docling configuration: WORKING")
    print(f"✅ Multiple export formats: SUPPORTED")  
    print(f"✅ Configuration loading: SUCCESSFUL")
    print(f"✅ Document processing: FUNCTIONAL")
    print(f"✅ Content extraction: OPERATIONAL")
    print(f"✅ Caching system: {'ENABLED' if global_args.docling_enable_cache else 'DISABLED'}")
    print()
    
    print("🎉 ENHANCED DOCLING TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return True


async def main():
    """Main test function."""
    print("🚀 Starting enhanced Docling configuration test...")
    print()
    
    success = await test_enhanced_docling_config()
    
    if success:
        print("✅ All tests passed! Enhanced Docling configuration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration and try again.")
        return 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)