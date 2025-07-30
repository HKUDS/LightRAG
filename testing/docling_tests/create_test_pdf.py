#!/usr/bin/env python3
"""
Create a test PDF document for testing enhanced Docling configuration.
This PDF will contain various elements to test different Docling features.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, red, blue, green
from reportlab.lib import colors
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.utils import ImageReader
import os

def create_test_pdf():
    """Create a comprehensive test PDF with various content types."""
    filename = "/home/ajithkv/developments/LightRAG/test_document_enhanced_docling.pdf"
    
    # Create the PDF document
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=1*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=blue,
        alignment=1  # Center alignment
    )
    
    # Title page
    story.append(Paragraph("Enhanced Docling Test Document", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph("Testing Advanced Document Processing Features", styles['Heading2']))
    story.append(Spacer(1, 0.25*inch))
    
    # Document metadata in content
    metadata_content = """
    <b>Document Metadata:</b><br/>
    • Title: Enhanced Docling Test Document<br/>
    • Author: LightRAG Test Suite<br/>
    • Subject: Document Processing Testing<br/>
    • Keywords: Docling, OCR, Tables, Images, Text Extraction<br/>
    • Created: 2025-01-29<br/>
    • Version: 1.0<br/>
    """
    story.append(Paragraph(metadata_content, styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Page break
    story.append(PageBreak())
    
    # Section 1: Plain Text Content
    story.append(Paragraph("1. Plain Text Processing Test", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    plain_text = """
    This section contains plain text content to test basic text extraction capabilities.
    The enhanced Docling configuration should properly extract this content while
    preserving formatting and structure. This paragraph includes various types of
    content including <b>bold text</b>, <i>italic text</i>, and <u>underlined text</u>.
    
    Additionally, this section tests paragraph separation and line spacing. The Docling
    processor should maintain proper text flow and formatting when converting to the
    specified output format (markdown, JSON, HTML, or DocTags).
    """
    story.append(Paragraph(plain_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Section 2: Table Content
    story.append(Paragraph("2. Table Structure Recognition Test", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("The following table tests table structure recognition capabilities:", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    # Create a test table
    table_data = [
        ['Feature', 'Configuration Variable', 'Default Value', 'Description'],
        ['Export Format', 'DOCLING_EXPORT_FORMAT', 'markdown', 'Output format selection'],
        ['OCR Processing', 'DOCLING_ENABLE_OCR', 'true', 'Enable OCR for images'],
        ['Table Recognition', 'DOCLING_ENABLE_TABLE_STRUCTURE', 'true', 'Advanced table processing'],
        ['Figure Extraction', 'DOCLING_ENABLE_FIGURES', 'true', 'Extract images and figures'],
        ['Cache System', 'DOCLING_ENABLE_CACHE', 'true', 'Enable intelligent caching'],
        ['Max Workers', 'DOCLING_MAX_WORKERS', '2', 'Parallel processing workers'],
    ]
    
    table = Table(table_data, colWidths=[1.2*inch, 2*inch, 1*inch, 2.3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Section 3: Complex table with merged cells
    story.append(Paragraph("Complex Table with Merged Cells:", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    complex_table_data = [
        ['Processing Stage', 'Input', 'Output', 'Configuration'],
        ['Document Loading', 'PDF/DOCX/PPTX/XLSX', 'Raw Document Object', 'DOCUMENT_LOADING_ENGINE'],
        ['Content Extraction', 'Document Object', 'Structured Content', 'Export Format Settings'],
        ['OCR Processing', 'Images in Document', 'Extracted Text', 'OCR Confidence & DPI'],
        ['Table Recognition', 'Table Structures', 'Structured Table Data', 'Table Confidence'],
        ['Caching', 'Processed Content', 'Cached Results', 'Cache TTL & Directory'],
    ]
    
    complex_table = Table(complex_table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    complex_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.white]),
    ]))
    
    story.append(complex_table)
    story.append(PageBreak())
    
    # Section 4: Lists and structured content
    story.append(Paragraph("3. Structured Content and Lists", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    list_content = """
    <b>Enhanced Docling Features:</b>
    
    <b>1. Processing Options:</b>
    • Multiple export formats (markdown, JSON, HTML, DocTags)
    • Configurable OCR with confidence thresholds
    • Advanced table structure recognition
    • Figure and image extraction capabilities
    
    <b>2. Performance Features:</b>
    • Parallel processing with configurable workers
    • Intelligent caching system with TTL control
    • Optimized processing pipeline
    • Memory-efficient content handling
    
    <b>3. Quality Controls:</b>
    • OCR confidence thresholds (0.0-1.0)
    • Table detection confidence settings
    • Image DPI configuration for OCR
    • Content validation and error handling
    
    <b>4. Output Customization:</b>
    • Include/exclude page numbers
    • Section heading preservation
    • Metadata extraction and inclusion
    • Image processing controls
    """
    story.append(Paragraph(list_content, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Section 5: Code-like content
    story.append(Paragraph("4. Configuration Example", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        backgroundColor=colors.lightgrey,
        leftIndent=10,
        rightIndent=10,
        spaceAfter=6,
        spaceBefore=6
    )
    
    config_example = """
# Enhanced Docling Configuration Example
DOCUMENT_LOADING_ENGINE=DOCLING
DOCLING_EXPORT_FORMAT=markdown
DOCLING_MAX_WORKERS=2
DOCLING_ENABLE_OCR=true
DOCLING_ENABLE_TABLE_STRUCTURE=true
DOCLING_ENABLE_FIGURES=true
DOCLING_INCLUDE_PAGE_NUMBERS=true
DOCLING_INCLUDE_HEADINGS=true
DOCLING_EXTRACT_METADATA=true
DOCLING_OCR_CONFIDENCE=0.7
DOCLING_TABLE_CONFIDENCE=0.8
DOCLING_ENABLE_CACHE=true
DOCLING_CACHE_TTL_HOURS=168
    """
    story.append(Paragraph(config_example, code_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Section 6: Summary
    story.append(Paragraph("5. Testing Summary", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    summary_text = """
    This test document contains various content types to validate the enhanced Docling
    configuration implementation:
    
    ✓ Plain text with formatting
    ✓ Structured tables with headers and data
    ✓ Complex tables with different styling
    ✓ Lists and hierarchical content
    ✓ Code blocks and configuration examples
    ✓ Multiple sections with headings
    ✓ Page breaks and spacing
    ✓ Metadata embedded in content
    
    The enhanced Docling processor should extract all this content accurately while
    preserving structure and applying the configured processing options such as
    OCR settings, table recognition, and output formatting.
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    
    # Build the PDF
    doc.build(story)
    print(f"Test PDF created: {filename}")
    return filename

if __name__ == "__main__":
    create_test_pdf()