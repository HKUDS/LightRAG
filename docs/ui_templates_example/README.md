# Example UI customization bundle

A ready-to-copy example for `UI_TEMPLATES_DIR` (the customizable welcome
page, query empty state and login page). This directory is
**documentation only** — the server never loads or bundles it.

For the complete guide — deployment from source / Docker / Kubernetes, the
full `manifest.json` reference, verification and every startup error — see
[UserDefinedUI.md](../UserDefinedUI.md) ([中文](../UserDefinedUI-zh.md)).

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
- `login` and `agreements` are OPTIONAL per locale, and together they switch
  on the **login consent gate**: when a locale declares both, the login page
  shows the `login` text plus a checkbox ("I agree to the Privacy Policy and
  Model Service Agreement"), and sign-in stays disabled until it is ticked.
  The checkbox carries ONE link, which opens `agreements` in a dialog — so
  write the privacy policy and the model service agreement into that single
  file (headings are the way to separate them) rather than expecting the
  reader to find two documents.
  - Declaring only one of the two leaves the gate OFF: a branded login page
    with nothing to agree to, or an agreement document no page links to, is
    a half-finished configuration and is treated as such.
  - Both are per LOCALE. A visitor resolving to a locale that declares
    neither sees no checkbox, so declare them for every locale the gate must
    cover (or route uncovered locales there through `fallbacks`).
  - A declared file that is empty fails startup — the gate must never point
    at a blank document.
  - The gate covers **credentialed sign-in only**. A deployment with
    authentication disabled (`AUTH_ACCOUNTS` unset) admits visitors as
    guests without it, and that is deliberate rather than a gap: with no
    authentication there is no identified user to hold to an agreement, and
    auth-disabled is a development and demo posture, not a production one.
    Configure `AUTH_ACCOUNTS` if the agreement has to be accepted.
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
- **The bundle's languages and the WebUI's are separate sets.** Any valid
  BCP 47 locale is accepted here, but the surrounding interface (buttons,
  settings, login) exists only in the languages the WebUI ships: `en`, `zh`,
  `zh-TW`, `fr`, `ar`, `ru`, `ja`, `de`, `uk`, `ko`, `vi`. A locale outside
  that set renders its own content correctly, text direction included, while
  the controls around it stay in the visitor's resolved UI language — there
  is no interface translation to switch to. Startup logs a warning naming
  such locales. Declare a locale from the list above whenever you want the
  whole page in one language.
- If anything in the bundle is invalid, the server FAILS TO START with a
  descriptive error (a misconfigured bundle is never silently ignored).
