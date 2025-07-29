# Testing Directory

This directory contains test files and utilities for LightRAG development and validation.

## Structure

- `docling_tests/` - Enhanced Docling configuration tests and utilities
- Individual test files should be placed in appropriate subdirectories

## Test Files

### Docling Tests (`docling_tests/`)

- `test_enhanced_docling.py` - Comprehensive test suite for enhanced Docling configuration
- `test_api_integration.py` - API integration testing for Docling features
- `create_test_pdf.py` - Utility to generate test PDF documents
- `check_docling_api.py` - Docling API compatibility checker
- `test_document_enhanced_docling.pdf` - Test PDF document with complex content

## Usage

Run tests from the project root directory:

```bash
# Run enhanced Docling configuration tests
python testing/docling_tests/test_enhanced_docling.py

# Test API integration
python testing/docling_tests/test_api_integration.py

# Generate test documents
python testing/docling_tests/create_test_pdf.py
```

## Test Outputs

Test outputs are automatically saved to `docs/test_outputs/` to keep them organized and accessible for review.