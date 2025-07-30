#!/usr/bin/env python3
"""
Check the current Docling API to understand available classes and methods.
"""

try:
    from docling.document_converter import DocumentConverter
    print("✅ DocumentConverter imported successfully")
    
    # Check what's available in DocumentConverter
    converter = DocumentConverter()
    print(f"✅ DocumentConverter created: {type(converter)}")
    
    # Check available methods and attributes
    print("\n📋 Available methods and attributes:")
    for attr in sorted(dir(converter)):
        if not attr.startswith('_'):
            attr_type = type(getattr(converter, attr))
            print(f"  {attr}: {attr_type}")
    
    # Check the convert method signature
    import inspect
    convert_signature = inspect.signature(converter.convert)
    print(f"\n🔍 convert method signature: {convert_signature}")
    
    # Try a simple conversion to see what's returned
    print("\n🧪 Testing simple conversion...")
    test_file = "/home/ajithkv/developments/LightRAG/test_document_enhanced_docling.pdf"
    result = converter.convert(test_file)
    print(f"✅ Conversion result type: {type(result)}")
    
    # Check result attributes
    print("\n📋 Result attributes:")
    for attr in sorted(dir(result)):
        if not attr.startswith('_'):
            attr_type = type(getattr(result, attr))
            print(f"  {attr}: {attr_type}")
    
    # Check document attributes
    if hasattr(result, 'document'):
        doc = result.document
        print(f"\n📄 Document type: {type(doc)}")
        print("📋 Document methods:")
        for attr in sorted(dir(doc)):
            if not attr.startswith('_') and 'export' in attr.lower():
                attr_type = type(getattr(doc, attr))
                print(f"  {attr}: {attr_type}")
    
    # Test export methods
    if hasattr(result, 'document'):
        doc = result.document
        if hasattr(doc, 'export_to_markdown'):
            print("\n🧪 Testing export_to_markdown...")
            try:
                markdown_content = doc.export_to_markdown()
                print(f"✅ Markdown export successful, length: {len(markdown_content)}")
                print(f"📋 Preview: {markdown_content[:200]}...")
            except Exception as e:
                print(f"❌ Markdown export failed: {e}")
        
        if hasattr(doc, 'export_to_json'):
            print("\n🧪 Testing export_to_json...")
            try:
                json_content = doc.export_to_json()
                print(f"✅ JSON export successful, length: {len(json_content)}")
            except Exception as e:
                print(f"❌ JSON export failed: {e}")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()