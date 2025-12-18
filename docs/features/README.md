# Feature Documentation

This directory contains HTML documentation generated from markdown files in `FEATURES/`.

## Viewing the Documentation

### Option 1: Open Directly in Browser

Simply open `index.html` in your web browser:

```bash
# On macOS
open index.html

# On Linux
xdg-open index.html

# On Windows
start index.html

# Or just drag the file to your browser
```

### Option 2: Serve with Python HTTP Server

For a better experience (especially with relative links), serve the docs with a local HTTP server:

```bash
# From the docs/features directory
python3 -m http.server 8000

# Then open in browser:
# http://localhost:8000
```

### Option 3: Serve with Node.js

If you have Node.js installed:

```bash
# Install http-server globally (one time)
npm install -g http-server

# Serve the docs
http-server -p 8000

# Then open: http://localhost:8000
```

## Regenerating the Documentation

To rebuild the HTML files from the latest markdown in `FEATURES/`:

```bash
# From the repository root
python3 build_feature_docs.py

# Or specify a custom output directory
python3 build_feature_docs.py --output-dir path/to/output
```

The script will:
1. Scan all subdirectories in `FEATURES/`
2. Convert all `.md` files to HTML
3. Generate an index page listing all features
4. Create a beautiful, styled HTML page for each feature

## Files

- `index.html` - Main index page listing all features
- `FR01-*.html` - Individual feature documentation pages
- (More will appear as features are added)

## Updating Documentation

Whenever you update markdown files in `FEATURES/`, simply run:

```bash
python3 build_feature_docs.py
```

This will regenerate all HTML pages with the latest content.
