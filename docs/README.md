# LightRAG Documentation

This directory contains generated documentation for LightRAG features.

## 📚 Feature Documentation

Feature documentation is generated from markdown files in the `FEATURES/` directory.

### Quick Start

1. **Build the documentation:**
   ```bash
   # From repository root
   python3 build_feature_docs.py
   ```

2. **View the documentation:**
   ```bash
   # Serve with Python HTTP server
   ./serve_docs.sh

   # Or open directly
   open docs/features/index.html
   ```

3. **Browse to:** http://localhost:8000

## 📖 What Gets Generated

The build script (`build_feature_docs.py`) scans the `FEATURES/` directory and:

- Converts all `.md` files in each feature subfolder to HTML
- Creates a comprehensive HTML page per feature with:
  - Beautiful styling and responsive design
  - Navigation between documents
  - Table of contents
  - Code syntax highlighting
  - Mobile-friendly layout
- Generates an index page listing all features

## 🔄 Workflow

### Adding New Features

1. Create a new folder under `FEATURES/` (e.g., `FEATURES/FR02-new-feature/`)
2. Add markdown documentation files
3. Run `python3 build_feature_docs.py`
4. New feature appears in the documentation automatically

### Updating Existing Features

1. Edit markdown files in `FEATURES/FR*-*/`
2. Run `python3 build_feature_docs.py`
3. HTML files are regenerated with latest content

## 🎨 Features of Generated Docs

- **Clean, Modern Design**: Professional styling with gradients and shadows
- **Responsive**: Works on desktop, tablet, and mobile
- **Navigation**: Easy navigation between documents in a feature
- **Code Highlighting**: Syntax highlighting for code blocks
- **Tables**: Beautiful table formatting
- **Links**: Internal and external linking
- **Breadcrumbs**: Easy navigation back to index

## 📁 Directory Structure

```
docs/
├── README.md (this file)
└── features/
    ├── index.html              # Main index listing all features
    ├── FR01-*.html             # Feature-specific pages
    └── README.md               # Instructions for viewing docs
```

## 🛠️ Advanced Usage

### Custom Output Directory

```bash
python3 build_feature_docs.py --output-dir custom/path
```

### Serve on Different Port

```bash
cd docs/features
python3 -m http.server 9000
```

### Deploy to Web Server

The generated HTML files are static and can be deployed to any web server:

- Copy `docs/features/` to your web server
- Or use GitHub Pages, Netlify, Vercel, etc.
- No server-side processing required

## 🔧 Requirements

The build script requires Python 3.6+ and the `markdown` library:

```bash
pip install markdown
```

## 📝 Markdown Support

The documentation builder supports:

- **Standard Markdown**: Headings, paragraphs, lists, links, images
- **Tables**: GitHub-flavored table syntax
- **Code Blocks**: Fenced code blocks with syntax highlighting
- **Blockquotes**: For callouts and notes
- **HTML**: You can embed HTML if needed

## 🚀 Integration with Documentation Frameworks

While the generated HTML is standalone, you can integrate it with documentation frameworks:

### MkDocs

```bash
# Copy generated HTML to MkDocs site
cp docs/features/*.html mkdocs/docs/features/
```

### Docusaurus

```bash
# Copy to Docusaurus static directory
cp docs/features/*.html docusaurus/static/features/
```

### Sphinx

```bash
# Copy to Sphinx _static directory
cp docs/features/*.html sphinx/_static/features/
```

## 🎯 Tips

- **Run the build script** after any documentation changes
- **Use descriptive filenames** in FEATURES (e.g., `00-OVERVIEW.md`, `01-ARCHITECTURE.md`)
- **Keep markdown clean** - the script preserves your structure
- **Test locally** before deploying with `./serve_docs.sh`

## 🤝 Contributing

When contributing feature documentation:

1. Create markdown files in appropriate `FEATURES/FR*-*/` directory
2. Follow existing naming conventions
3. Run build script to verify rendering
4. Commit both markdown and generated HTML

## 📞 Support

For issues with the documentation build process:

1. Check `build_feature_docs.py` has execute permissions
2. Verify `markdown` library is installed
3. Ensure `FEATURES/` directory exists
4. Check markdown syntax is valid
