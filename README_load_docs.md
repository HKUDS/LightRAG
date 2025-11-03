# LightRAG Documentation Loader

Advanced script to load markdown documentation into LightRAG with flexible reference modes.

## Quick Start

```bash
# Default mode (file path references)
python load_docs.py /path/to/your/docs

# URL mode (website link references)
python load_docs.py /path/to/docs --mode urls --base-url https://docs.example.com/
```

## Reference Modes

### Files Mode (Default)
Uses local file paths in query response citations:
```bash
python load_docs.py docs/
python load_docs.py docs/ --mode files
```

**Query Response Example:**
```
### References
- [DC] getting-started/installation.md
- [KG] administration/setup.md
```

### URLs Mode
Uses website URLs in query response citations:
```bash
python load_docs.py docs/ --mode urls --base-url https://docs.apolo.us/index/
python load_docs.py docs/ --mode urls --base-url https://my-docs.com/v1/
```

**Query Response Example:**
```
### References
- [DC] https://docs.apolo.us/index/getting-started/installation
- [KG] https://docs.apolo.us/index/administration/setup
```

**⚠️ Important for URLs Mode**: Your local file structure must match your documentation site's URL structure for proper link generation.

**File Structure Requirements:**
```
docs/
├── getting-started/
│   ├── installation.md          → https://docs.example.com/getting-started/installation
│   └── first-steps.md           → https://docs.example.com/getting-started/first-steps
├── administration/
│   ├── README.md                → https://docs.example.com/administration
│   └── setup.md                 → https://docs.example.com/administration/setup
└── README.md                    → https://docs.example.com/
```

**URL Mapping Rules:**
- `.md` extension is removed from URLs
- `README.md` files map to their directory URL
- Subdirectories become URL path segments
- Hyphens and underscores in filenames are preserved

### Organizing Docs for URL Mode

**Step 1: Analyze Your Documentation Site Structure**
```bash
# Visit your docs site and note the URL patterns:
# https://docs.example.com/getting-started/installation
# https://docs.example.com/api/authentication
# https://docs.example.com/guides/deployment
```

**Step 2: Create Matching Directory Structure**
```bash
mkdir -p docs/{getting-started,api,guides}
```

**Step 3: Organize Your Markdown Files**
```bash
# Match each URL to a file path:
docs/getting-started/installation.md     # → /getting-started/installation
docs/api/authentication.md              # → /api/authentication
docs/guides/deployment.md               # → /guides/deployment
docs/guides/README.md                   # → /guides (overview page)
```

**Step 4: Verify URL Mapping**
```bash
# Test a few URLs manually to ensure they work:
curl -I https://docs.example.com/getting-started/installation
curl -I https://docs.example.com/api/authentication
```

**Common Documentation Site Patterns:**

| Site Type | File Structure | URL Structure |
|-----------|---------------|---------------|
| **GitBook** | `docs/section/page.md` | `/section/page` |
| **Docusaurus** | `docs/section/page.md` | `/docs/section/page` |
| **MkDocs** | `docs/section/page.md` | `/section/page/` |
| **Custom** | Varies | Match your site's pattern |

**Real Example: Apolo Documentation**
```bash
# Apolo docs site: https://docs.apolo.us/index/
# Your local structure should match:
apolo-docs/
├── getting-started/
│   ├── first-steps/
│   │   ├── getting-started.md   → /index/getting-started/first-steps/getting-started
│   │   └── README.md           → /index/getting-started/first-steps
│   ├── apolo-base-docker-image.md → /index/getting-started/apolo-base-docker-image
│   └── faq.md                  → /index/getting-started/faq
├── apolo-console/
│   └── getting-started/
│       └── sign-up-login.md    → /index/apolo-console/getting-started/sign-up-login
└── README.md                   → /index/

# Load with correct base URL:
python load_docs.py apolo-docs/ --mode urls --base-url https://docs.apolo.us/index/
```

## Complete Usage Examples

```bash
# Load Apolo documentation with URL references
python load_docs.py ../apolo-copilot/docs/official-apolo-documentation/docs \
  --mode urls --base-url https://docs.apolo.us/index/

# Load with custom LightRAG endpoint
python load_docs.py docs/ --endpoint https://lightrag.example.com

# Load to local instance, skip test query
python load_docs.py docs/ --no-test

# Files mode with custom endpoint
python load_docs.py docs/ --mode files --endpoint http://localhost:9621
```

## Features

- **Dual Reference Modes**: File paths or live website URLs in citations
- **Flexible Base URL**: Works with any documentation site structure
- **Simple dependency**: Only requires `httpx` and Python standard library
- **Automatic discovery**: Finds all `.md` files recursively
- **Smart metadata**: Adds appropriate title, path/URL, and source information
- **Progress tracking**: Shows loading progress with success/failure counts
- **Health checks**: Verifies LightRAG connectivity before loading
- **Test queries**: Validates functionality after loading
- **Error handling**: Clear validation and error messages

## Requirements

```bash
pip install httpx
```

## Use Cases

This loader is perfect for:
- **Kubernetes deployments**: Self-contained with minimal dependencies
- **Quick testing**: Immediate setup without complex environments
- **Documentation loading**: Any markdown-based documentation
- **Development workflows**: Fast iteration and testing

## Requirements

```bash
pip install httpx
```

**Note**: This script is included with LightRAG deployments and provides a simple way to load any markdown documentation into your LightRAG instance.
