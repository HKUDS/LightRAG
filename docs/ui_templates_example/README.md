# Example UI customization bundle

A ready-to-copy example for `UI_TEMPLATES_DIR` (the workspace entry's
customizable welcome page and query empty state). This directory is
**documentation only** — the server never loads or bundles it.

## Usage

```yaml
services:
  lightrag:
    environment:
      UI_TEMPLATES_DIR: /app/ui_templates
    volumes:
      - ./ui_templates:/app/ui_templates:ro   # read-only mount recommended
```

Copy this directory, replace the texts and the logo, then restart the server
(all workers). There is no hot reload.

## Rules

- `manifest.json` is the only index; files it does not reference are never
  served.
- Every declared locale must provide `welcome`, `query_empty` and a non-empty
  `logo_alt`. Locale keys use BCP 47 hyphen form (`zh-TW`, not `zh_TW`).
- `brand.logo` is REQUIRED: a path, or an explicit `null` for "no logo".
  Omitting it fails startup (a missing logo must never silently fall back to
  the LightRAG logo under customer texts).
- A locale entry may override the logo with its own `logo` path (or `null`).
- `fallbacks` maps uncovered locales to an ordered list of DECLARED locales;
  resolution is single-level (uncovered locale → first declared target →
  `default_locale`). Content is never mixed field-by-field with the
  frontend's default branding.
- Content format is Markdown without raw HTML (HTML is dropped at render
  time); links and images work normally. Logos may be PNG, JPEG, WebP or SVG
  (checked by content, not extension). Limits: 64 KiB per template, 2 MiB per
  logo.
- The bundle is public content served without authentication — never put
  secrets or internal paths in it.
- If anything in the bundle is invalid, the server FAILS TO START with a
  descriptive error (a misconfigured bundle is never silently ignored).
