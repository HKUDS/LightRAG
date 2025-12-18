#!/usr/bin/env python3
"""
Build HTML documentation from FEATURES markdown files.

This script scans all subfolders under FEATURES/ and converts markdown files
to HTML pages with navigation and styling.

Usage:
    python build_feature_docs.py [--output-dir OUTPUT_DIR]

Options:
    --output-dir    Output directory for HTML files (default: docs/features)
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict
import markdown
from markdown.extensions.toc import TocExtension
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        .header p {{
            opacity: 0.9;
            font-size: 1.1rem;
        }}

        .nav {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        .nav h2 {{
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: #667eea;
        }}

        .nav ul {{
            list-style: none;
        }}

        .nav li {{
            margin: 0.5rem 0;
        }}

        .nav a {{
            color: #667eea;
            text-decoration: none;
            padding: 0.3rem 0.5rem;
            display: inline-block;
            border-radius: 4px;
            transition: background-color 0.2s;
        }}

        .nav a:hover {{
            background-color: #f0f0ff;
        }}

        .content {{
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }}

        .content h1 {{
            color: #667eea;
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-size: 2rem;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }}

        .content h1:first-child {{
            margin-top: 0;
        }}

        .content h2 {{
            color: #764ba2;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }}

        .content h3 {{
            color: #555;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
            font-size: 1.2rem;
        }}

        .content h4 {{
            color: #666;
            margin-top: 1rem;
            margin-bottom: 0.6rem;
            font-size: 1.1rem;
        }}

        .content p {{
            margin-bottom: 1rem;
            line-height: 1.8;
        }}

        .content ul, .content ol {{
            margin-left: 2rem;
            margin-bottom: 1rem;
        }}

        .content li {{
            margin-bottom: 0.5rem;
        }}

        .content code {{
            background-color: #f4f4f4;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            color: #e83e8c;
        }}

        .content pre {{
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 1rem;
            border-radius: 5px;
            overflow-x: auto;
            margin-bottom: 1rem;
        }}

        .content pre code {{
            background: none;
            color: inherit;
            padding: 0;
            font-size: 0.9rem;
        }}

        .content blockquote {{
            border-left: 4px solid #667eea;
            padding-left: 1rem;
            margin: 1rem 0;
            color: #555;
            font-style: italic;
            background-color: #f9f9ff;
            padding: 1rem;
            border-radius: 4px;
        }}

        .content table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1rem;
        }}

        .content table th {{
            background-color: #667eea;
            color: white;
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
        }}

        .content table td {{
            border: 1px solid #ddd;
            padding: 0.75rem;
        }}

        .content table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        .content img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin: 1rem 0;
        }}

        .content a {{
            color: #667eea;
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-color 0.2s;
        }}

        .content a:hover {{
            border-bottom-color: #667eea;
        }}

        .footer {{
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
        }}

        .toc {{
            background-color: #f9f9ff;
            border: 1px solid #e0e0ff;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}

        .toc h2 {{
            color: #667eea;
            margin-bottom: 1rem;
        }}

        .toc ul {{
            list-style: none;
            margin-left: 0;
        }}

        .toc li {{
            margin: 0.3rem 0;
        }}

        .toc a {{
            color: #555;
        }}

        .breadcrumb {{
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }}

        .breadcrumb a {{
            color: #667eea;
            text-decoration: none;
        }}

        .breadcrumb a:hover {{
            text-decoration: underline;
        }}

        .breadcrumb span {{
            margin: 0 0.5rem;
            color: #999;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}

            .header {{
                padding: 1.5rem;
            }}

            .header h1 {{
                font-size: 1.5rem;
            }}

            .content {{
                padding: 1rem;
            }}

            .content h1 {{
                font-size: 1.5rem;
            }}

            .content h2 {{
                font-size: 1.3rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>

        {breadcrumb}

        {nav}

        <div class="content">
            {content}
        </div>

        <div class="footer">
            <p>Generated from markdown files in FEATURES/{feature_id}/</p>
            <p>LightRAG Feature Documentation • {date}</p>
        </div>
    </div>
</body>
</html>
"""

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LightRAG Features Documentation</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            color: white;
            margin-bottom: 3rem;
        }}

        .header h1 {{
            font-size: 3rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }}

        .header p {{
            font-size: 1.3rem;
            opacity: 0.9;
        }}

        .features {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 2rem;
        }}

        .feature-card {{
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s, box-shadow 0.3s;
            text-decoration: none;
            color: inherit;
            display: block;
        }}

        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }}

        .feature-card h2 {{
            color: #667eea;
            margin-bottom: 0.5rem;
            font-size: 1.5rem;
        }}

        .feature-card .feature-id {{
            color: #999;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            font-family: monospace;
        }}

        .feature-card p {{
            color: #666;
            margin-bottom: 1rem;
        }}

        .feature-card .meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
            font-size: 0.9rem;
            color: #999;
        }}

        .feature-card .docs-count {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .feature-card .view-link {{
            color: #667eea;
            font-weight: 600;
        }}

        .footer {{
            text-align: center;
            margin-top: 3rem;
            color: white;
            opacity: 0.8;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 2rem;
            }}

            .features {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 LightRAG Features</h1>
            <p>Technical Documentation & Implementation Plans</p>
        </div>

        <div class="features">
            {feature_cards}
        </div>

        <div class="footer">
            <p>Generated on {date}</p>
        </div>
    </div>
</body>
</html>
"""


def extract_title_from_content(content: str) -> str:
    """Extract first H1 heading from markdown content."""
    lines = content.split('\n')
    for line in lines:
        if line.startswith('# '):
            return line[2:].strip()
    return "Documentation"


def extract_description_from_content(content: str) -> str:
    """Extract first paragraph after title."""
    lines = content.split('\n')
    found_title = False
    for line in lines:
        if line.startswith('# '):
            found_title = True
            continue
        if found_title and line.strip():
            return line.strip()[:200] + ('...' if len(line.strip()) > 200 else '')
    return "Feature documentation"


def convert_markdown_to_html(md_content: str) -> str:
    """Convert markdown to HTML with extensions."""
    md = markdown.Markdown(
        extensions=[
            TocExtension(title='Table of Contents'),
            CodeHiliteExtension(linenums=False),
            FencedCodeExtension(),
            TableExtension(),
            'nl2br',
            'sane_lists'
        ]
    )
    return md.convert(md_content)


def build_feature_page(feature_dir: Path, output_dir: Path) -> Dict:
    """Build HTML page for a feature from markdown files."""
    feature_id = feature_dir.name

    # Find all markdown files
    md_files = sorted(feature_dir.glob('*.md'))

    if not md_files:
        print(f"Warning: No markdown files found in {feature_dir}")
        return None

    # Read and combine content
    combined_content = []
    nav_items = []

    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title for navigation
        title = extract_title_from_content(content)
        file_id = md_file.stem
        nav_items.append({
            'title': title,
            'file': file_id,
            'filename': md_file.name
        })

        # Add separator between files
        if combined_content:
            combined_content.append('\n\n---\n\n')

        combined_content.append(content)

    # Convert to HTML
    combined_md = '\n'.join(combined_content)
    html_content = convert_markdown_to_html(combined_md)

    # Build navigation
    nav_html = '<div class="nav"><h2>📑 Documents in this Feature</h2><ul>'
    for item in nav_items:
        nav_html += f'<li><strong>{item["filename"]}</strong> - {item["title"]}</li>'
    nav_html += '</ul></div>'

    # Build breadcrumb
    breadcrumb = f'''
    <div class="breadcrumb">
        <a href="index.html">🏠 Features</a>
        <span>›</span>
        <span>{feature_id}</span>
    </div>
    '''

    # Get feature title and description
    first_file = md_files[0]
    with open(first_file, 'r', encoding='utf-8') as f:
        first_content = f.read()

    feature_title = extract_title_from_content(first_content)
    feature_subtitle = f"{len(md_files)} documentation files"

    # Generate HTML
    from datetime import datetime
    html = HTML_TEMPLATE.format(
        title=f"{feature_id}: {feature_title}",
        subtitle=feature_subtitle,
        breadcrumb=breadcrumb,
        nav=nav_html,
        content=html_content,
        feature_id=feature_id,
        date=datetime.now().strftime('%Y-%m-%d %H:%M')
    )

    # Write output
    output_file = output_dir / f"{feature_id}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Generated {output_file}")

    # Return metadata for index
    description = extract_description_from_content(first_content)
    return {
        'id': feature_id,
        'title': feature_title,
        'description': description,
        'docs_count': len(md_files),
        'html_file': f"{feature_id}.html"
    }


def build_index_page(features: List[Dict], output_dir: Path):
    """Build index page listing all features."""

    feature_cards_html = []
    for feature in features:
        card = f'''
        <a href="{feature['html_file']}" class="feature-card">
            <h2>{feature['title']}</h2>
            <div class="feature-id">{feature['id']}</div>
            <p>{feature['description']}</p>
            <div class="meta">
                <div class="docs-count">📄 {feature['docs_count']} documents</div>
                <div class="view-link">View docs →</div>
            </div>
        </a>
        '''
        feature_cards_html.append(card)

    from datetime import datetime
    html = INDEX_TEMPLATE.format(
        feature_cards='\n'.join(feature_cards_html),
        date=datetime.now().strftime('%Y-%m-%d %H:%M')
    )

    output_file = output_dir / 'index.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Generated {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Build HTML documentation from FEATURES markdown files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='docs/features',
        help='Output directory for HTML files (default: docs/features)'
    )

    args = parser.parse_args()

    # Setup paths
    features_dir = Path('FEATURES')
    output_dir = Path(args.output_dir)

    if not features_dir.exists():
        print(f"Error: FEATURES directory not found at {features_dir}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building feature documentation...")
    print(f"Source: {features_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    print()

    # Find all feature subdirectories
    feature_dirs = [d for d in features_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not feature_dirs:
        print(f"Error: No feature directories found in {features_dir}")
        return 1

    print(f"Found {len(feature_dirs)} feature(s):\n")

    # Build pages for each feature
    features_metadata = []
    for feature_dir in sorted(feature_dirs):
        metadata = build_feature_page(feature_dir, output_dir)
        if metadata:
            features_metadata.append(metadata)

    print()

    # Build index page
    if features_metadata:
        build_index_page(features_metadata, output_dir)

    print()
    print(f"✅ Documentation build complete!")
    print(f"📂 Output directory: {output_dir.absolute()}")
    print(f"🌐 Open {output_dir.absolute()}/index.html in your browser")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
