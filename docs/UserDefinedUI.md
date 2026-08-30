# User-Defined UI Content (`UI_TEMPLATES_DIR`)

LightRAG can serve **your own** welcome page, login-page text, user agreement,
query empty state and brand logo — in several languages — without rebuilding
the WebUI. You write a small directory of Markdown files plus a
`manifest.json`, point the `UI_TEMPLATES_DIR` environment variable at it, and
restart the server.

This document is the complete guide: what can be customized, the exact bundle
format, how to deploy it from source / Docker / Kubernetes, how to verify it,
and what every startup error means.

A ready-to-copy bundle lives in [`docs/ui_templates_example/`](./ui_templates_example/).

---

## 1. What you can customize

| Surface | Where the visitor sees it | Bundle field |
|---|---|---|
| **Welcome page** | `/workspace` entry, before sign-in (`/workspace/#/welcome`) | `welcome` (required) |
| **Query empty state** | `/workspace` query page, while the conversation is empty (and again after *Clear*) | `query_empty` (required) |
| **Login page text** | Above the username/password form, on **both** entries (`/webui/#/login` and `/workspace/#/login`) | `login` (optional) |
| **User agreement + consent checkbox** | A checkbox on the login page, whose link opens the document in a dialog; sign-in is blocked until it is ticked | `agreements` (optional, pairs with `login`) |
| **Brand logo** | Welcome page and query empty state | `brand.logo`, per-locale `logo`, `logo_alt` |

What you **cannot** set from the bundle:

- The browser tab title and the subtitle under the LightRAG wordmark on the
  login card. Those come from the `WEBUI_TITLE` / `WEBUI_DESCRIPTION`
  environment variables and the manifest deliberately cannot override them.
- The buttons, menus and settings around your content — see
  [§5.3 Interface languages](#53-interface-languages-vs-bundle-locales).
- Text direction. It is derived from the locale (see [§5.2](#52-text-direction-rtl)),
  never taken from bundle markup.

**All-or-nothing per locale.** A visitor either sees your bundle's
representation of a locale *entirely*, or LightRAG's built-in branding
*entirely*. Fields are never mixed one by one, so you cannot supply only a
logo and inherit LightRAG's welcome text.

---

## 2. Quick start

### 2.1 From source

```bash
# The conventional location for a source checkout; it is git-ignored.
cp -r docs/ui_templates_example lightrag_webui/ui_templates

# Edit the texts and drop in your own logo
$EDITOR lightrag_webui/ui_templates/locales/en/welcome.md
cp /path/to/your-logo.svg lightrag_webui/ui_templates/assets/logo.svg
```

Then add to `.env`:

```ini
UI_TEMPLATES_DIR=./lightrag_webui/ui_templates
```

and start the server:

```bash
lightrag-server
```

The startup log tells you which state you are in:

```
INFO: UI customization bundle active: bundle_revision=1f0c… locales=['en', 'zh', 'zh-TW']
```

### 2.2 With Docker Compose

The bundled compose files already mount the bundle directory read-only:

```yaml
    volumes:
      - ./data/ui_templates:/app/data/ui_templates:ro
```

So:

```bash
mkdir -p ./data/ui_templates
cp -r docs/ui_templates_example/* ./data/ui_templates/
```

then uncomment one line in the `lightrag` service's `environment:` block:

```yaml
      UI_TEMPLATES_DIR: "/app/data/ui_templates"
```

and restart:

```bash
docker compose up -d --force-recreate lightrag
```

See [§7 Deployment](#7-deployment) for the full picture, including the
wizard-generated `docker-compose.final.yml` and Kubernetes.

---

## 3. Bundle layout

```
ui_templates/
├── manifest.json            # the only index — nothing else is discovered
├── assets/
│   └── logo.svg             # PNG / JPEG / WebP / SVG
└── locales/
    ├── en/
    │   ├── welcome.md       # required
    │   ├── query_empty.md   # required
    │   ├── login.md         # optional
    │   └── agreements.md    # optional
    ├── zh/
    │   └── …
    └── zh-TW/
        └── …
```

The directory names above are a convention, not a rule: every file is located
through `manifest.json`, and a file the manifest does not reference is never
read and never served. Paths in the manifest are relative to the bundle root;
absolute paths, `..` segments and symlinks pointing outside the bundle are all
rejected at startup.

---

## 4. `manifest.json` reference

```jsonc
{
  "schema_version": 1,
  "default_locale": "en",
  "fallbacks": {
    "ko": ["en"],
    "ja": ["en"]
  },
  "brand": {
    "logo": "assets/logo.svg"
  },
  "locales": {
    "en": {
      "welcome": "locales/en/welcome.md",
      "query_empty": "locales/en/query_empty.md",
      "login": "locales/en/login.md",
      "agreements": "locales/en/agreements.md",
      "logo_alt": "Example Corp"
    },
    "zh": {
      "welcome": "locales/zh/welcome.md",
      "query_empty": "locales/zh/query_empty.md",
      "login": "locales/zh/login.md",
      "agreements": "locales/zh/agreements.md",
      "logo_alt": "示例公司"
    }
  }
}
```

> JSON has no comments — the `jsonc` above is for reading only. Keep your real
> manifest as plain JSON.

### 4.1 Top-level fields

| Field | Required | Type | Notes |
|---|---|---|---|
| `schema_version` | yes | number | Must be `1`. |
| `default_locale` | yes | string | Must be one of the declared `locales` keys. Used when a visitor's locale matches nothing. |
| `brand` | yes | object | Exactly one key: `logo`. |
| `brand.logo` | yes | string \| `null` | Path to the default logo, or an explicit `null` for "this deployment shows no logo". **Omitting the key fails startup** — a missing logo must never silently fall back to the LightRAG logo underneath your texts. |
| `locales` | yes | object | Non-empty map of locale → entry. |
| `fallbacks` | no | object \| `null` | Maps *uncovered* locales onto declared ones. See [§5.1](#51-locale-resolution). |

Unknown top-level fields are an error, so a typo (`defaultLocale`) is reported
at startup rather than silently ignored.

### 4.2 Locale entries

| Field | Required | Type | Notes |
|---|---|---|---|
| `welcome` | yes | string | Path to the welcome-page Markdown. |
| `query_empty` | yes | string | Path to the query-empty-state Markdown. |
| `logo_alt` | yes | string | Non-empty alt text for the logo, in this locale's language. |
| `logo` | no | string \| `null` | Overrides `brand.logo` for this locale (`null` = no logo here). When the key is **absent**, the locale inherits `brand.logo`. |
| `login` | no | string \| `null` | Path to the login-page blurb. |
| `agreements` | no | string \| `null` | Path to the single user-agreement document. |

`login` and `agreements` together switch on the login consent gate — see
[§6](#6-the-login-consent-gate).

### 4.3 Locale keys

Keys are BCP 47 tags in **hyphen** form, written exactly in their normalized
shape:

- lowercase language (`zh`, `en`, `ar`);
- Titlecase 4-letter script (`Hant`, `Arab`);
- UPPERCASE 2-letter region (`TW`, `CN`);
- `zh_TW` (underscore) is rejected — write `zh-TW`;
- `zh-tw` is rejected too: the key must already be normalized (`zh-TW`).

Any valid BCP 47 tag is accepted, not only the languages the WebUI itself
ships — but read [§5.3](#53-interface-languages-vs-bundle-locales) before
declaring one.

### 4.4 Limits and formats

| Rule | Value |
|---|---|
| Markdown template size | ≤ 64 KiB per file |
| `manifest.json` size | ≤ 64 KiB |
| Logo size | ≤ 2 MiB per file |
| Template encoding | UTF-8 (invalid UTF-8 fails startup) |
| Logo formats | PNG, JPEG, WebP, SVG — **detected from the file's bytes, not its extension** |

For SVG, the file must actually open an `<svg>` root element (an XML
declaration, comments, a DOCTYPE or processing instructions may precede it). A
namespace-prefixed root (`<s:svg>`) or a UTF-16/32-encoded file is not
recognized — no SVG tool emits those.

---

## 5. Languages

### 5.1 Locale resolution

The WebUI asks for one locale: the interface language it resolved for this
visitor (explicit setting in the UI > browser language > `en`). The server then
resolves it in a **single hop**:

1. **Exact match** against a declared locale → used.
2. Otherwise the first *declared* target listed under `fallbacks.<requested>`.
3. Otherwise `default_locale`.

Every `fallbacks` target must be a declared locale, which is what makes
resolution single-hop and cycles impossible. The *source* side may be any
locale — pointing uncovered languages somewhere sensible is the entire purpose
of the map:

```json
"fallbacks": {
  "ko": ["en"],
  "ja": ["en"],
  "de": ["en"]
}
```

Without an entry, an uncovered locale simply lands on `default_locale`, so
`fallbacks` only matters when different uncovered languages should land in
*different* places (e.g. `zh-HK` → `zh-TW`, everything else → `en`).

### 5.2 Text direction (RTL)

Direction is derived from the resolved locale against a CLDR-derived registry
and sent to the browser; the bundle cannot set it. An explicit script subtag
wins over the language, which is the escape hatch when the default is wrong:

- `ar`, `he`, `fa`, `ur`, `ps`, `ckb`, `dv`… → right-to-left;
- `ku` → left-to-right, but `ku-Arab` → right-to-left;
- `az-Arab`, `pa-Arab`, `ha-Arab` likewise.

### 5.3 Interface languages vs bundle locales

The bundle's set of languages and the WebUI's are **independent**. The WebUI
ships interface translations (buttons, settings, login labels, the consent
checkbox wording) for:

`en`, `zh`, `zh-TW`, `fr`, `ar`, `ru`, `ja`, `de`, `uk`, `ko`, `vi`

A bundle may declare a locale outside that list — say `nl`. Its content will
render correctly, direction included, but the controls around it stay in the
visitor's resolved interface language, because no Dutch interface translation
exists to switch to. Startup logs a warning naming such locales:

```
WARNING: UI customization: the WebUI ships no interface translation for ['nl'] …
```

Declare a locale from the supported list whenever you want the whole page in
one language.

---

## 6. The login consent gate

When a locale declares **both** `login` and `agreements`, the login page for
that locale shows:

- your `login` Markdown above the form, and
- a checkbox reading (English UI) *"I agree to the Privacy Policy and Model
  Service Agreement"*, whose single link opens your `agreements` document in a
  dialog.

The **Login** button stays disabled until the box is ticked, and pressing
Enter in the form is refused the same way.

### 6.1 One document, not two

The checkbox carries exactly **one** link. Write both the privacy policy and
the model service agreement into that single `agreements.md`, separating them
with headings:

```markdown
## Privacy Policy

…

## Model Service Agreement

…
```

The checkbox label itself is not customizable — it comes from the WebUI's own
translations, so it already reads naturally in each interface language
(`同意《用户隐私协议》和《模型服务协议》` in Chinese, and so on). Keep your
document's headings consistent with that wording.

### 6.2 Rules to know

- **Both or neither.** Declaring only `login` gives a branded login page with
  no gate; declaring only `agreements` gives a document nothing links to.
  Neither turns the gate on — a half-configuration is treated as a
  half-configuration, not as consent.
- **Per locale.** A visitor resolving to a locale that declares neither field
  sees no checkbox. Declare the pair for every locale that must be gated, or
  route the uncovered ones there with `fallbacks`.
- **A declared-but-empty file fails startup.** The gate must never point at a
  blank document.
- **The tick is not remembered.** It lives only as long as the login page and
  is bound to the exact document text on screen, so it is asked for on every
  visit. Switching the interface language mid-page replaces the document and
  clears the tick, unless the new locale's text is byte-for-byte identical.
- **Credentialed sign-in only.** A deployment with authentication disabled
  (`AUTH_ACCOUNTS` unset) admits visitors as guests without the gate. That is
  deliberate rather than a gap: with no authentication there is no identified
  user to bind an agreement to, and auth-disabled is a development/demo
  posture. **If the agreement must be accepted, configure `AUTH_ACCOUNTS`**
  (with `TOKEN_SECRET`).
- **Fail-open on an unreachable server.** If the customization endpoint cannot
  be loaded after its retries, the gate stays off rather than locking everyone
  out of the deployment.

---

## 7. Deployment

### 7.1 Where the bundle lives

| Deployment | Bundle directory | `UI_TEMPLATES_DIR` |
|---|---|---|
| From source | `lightrag_webui/ui_templates/` (git-ignored) | `./lightrag_webui/ui_templates` |
| Docker / Compose | host `./data/ui_templates/` → container `/app/data/ui_templates` | `/app/data/ui_templates` |
| Kubernetes | ConfigMap or PVC mounted at `/app/data/ui_templates` | `/app/data/ui_templates` |

Relative paths are resolved against the server's working directory, so an
absolute path is the safer choice whenever you are not sure where the process
starts.

### 7.2 Docker Compose

`docker-compose.yml`, `docker-compose-full.yml` and `docker-compose.podman.yml`
all ship the mount already, plus the activation line commented out:

```yaml
services:
  lightrag:
    volumes:
      - ./data/ui_templates:/app/data/ui_templates:ro
    environment:
      # UI_TEMPLATES_DIR: "/app/data/ui_templates"
```

The mount is inert on its own: with `UI_TEMPLATES_DIR` unset, an absent or
empty `./data/ui_templates` changes nothing. Uncommenting the environment line
is the single step that activates the feature.

Two practical notes:

- **Create the directory before the first `up`.** If Docker creates it for you
  it will be owned by `root`, and you will need `sudo` to copy files into it.
- `:ro` is intentional — the server only ever reads the bundle.

Setting `UI_TEMPLATES_DIR` in `.env` also works (the file is mounted into the
container), but the compose `environment:` block is the better home for it: the
value is a *container* path, and keeping container paths out of `.env` is what
lets the same `.env` stay usable when running from source. Note that a compose
`environment:` entry wins over the same key in `.env`.

### 7.3 The wizard-generated compose file

`make env-base` / `make env-storage` / `make env-server` generate
`docker-compose.final.yml`. The generator now adds the same read-only mount if
it is not already present, so regenerating an existing file picks it up:

```bash
make env-server        # or any other make env-* target
grep ui_templates docker-compose.final.yml
```

The wizard does **not** write `UI_TEMPLATES_DIR` for you — add it yourself to
the `lightrag` service's `environment:` block in `docker-compose.final.yml`.
The wizard preserves user-added environment keys and bind mounts in that
service across regenerations.

### 7.4 Kubernetes

The bundled Helm chart has no dedicated value for this yet, so mount the
bundle by patching the deployment. Text files fit a ConfigMap; a binary logo
goes in the same ConfigMap's `binaryData` (or a PVC, if the bundle is large):

```bash
kubectl create configmap lightrag-ui-templates \
  --from-file=manifest.json=./ui_templates/manifest.json \
  --from-file=./ui_templates/locales/en \
  --dry-run=client -o yaml | kubectl apply -f -
```

Then add to the container:

```yaml
          volumeMounts:
            - name: ui-templates
              mountPath: /app/data/ui_templates
              readOnly: true
          env:
            - name: UI_TEMPLATES_DIR
              value: /app/data/ui_templates
      volumes:
        - name: ui-templates
          configMap:
            name: lightrag-ui-templates
```

A ConfigMap flattens directories, so either keep the bundle flat and write the
manifest paths to match, or use several ConfigMaps mounted at subpaths. Whatever
the layout, the manifest's relative paths must resolve inside the mount.

### 7.5 Applying changes

There is **no hot reload**. The whole bundle is validated once at startup and
activated as an immutable in-memory snapshot; request handling never touches
the disk again. After editing the bundle, restart the server — and with
`lightrag-gunicorn` or any multi-worker setup, restart **all** workers, or some
of them will keep serving the previous revision.

No cache purge is needed: the content response is sent `Cache-Control:
no-store`, and logo URLs embed the file's content hash, so changed bytes
produce a new URL.

---

## 8. Verifying the deployment

**Startup log.** One of these lines always appears:

```
INFO: UI customization: no bundle configured (UI_TEMPLATES_DIR unset)
INFO: UI customization bundle active: bundle_revision=<sha256> locales=['en', 'zh']
```

`bundle_revision` is a hash over every referenced file. If it does not change
after you edited a file, the server is not reading the directory you think it
is — or it was not actually restarted.

**The endpoint.** It is public (the welcome page is shown before login), so a
plain `curl` works:

```bash
curl -s 'http://localhost:9621/ui/customization?locale=zh' | jq
```

```json
{
  "customized": true,
  "requested_locale": "zh",
  "locale": "zh",
  "fallback_used": false,
  "direction": "ltr",
  "brand": {
    "title": "My Graph KB",
    "description": "Simple and Fast Graph Based RAG System",
    "logo_url": "/ui/customization/assets/9f2a…/brand-logo",
    "logo_alt": "示例公司"
  },
  "welcome": { "format": "markdown", "content": "## 欢迎…" },
  "query_empty": { "format": "markdown", "content": "…" },
  "login": { "format": "markdown", "content": "…" },
  "agreements": { "format": "markdown", "content": "…" },
  "consent_required": true
}
```

Useful checks:

- `"customized": false` → no bundle is active (`UI_TEMPLATES_DIR` unset or
  empty). The frontend is showing LightRAG's built-in branding.
- `"fallback_used": true` → the requested locale is not declared; `locale`
  tells you where it landed.
- `"consent_required"` → whether the checkbox will appear for this locale.
- `logo_url: null` → this locale resolves to *no* logo (an explicit `null`
  somewhere), not to the LightRAG logo.

**In the browser.** Visit `/workspace` for the welcome page, `/workspace/#/login`
or `/webui/#/login` for the login page, and switch the interface language from
the settings menu to check each locale.

---

## 9. Troubleshooting

If `UI_TEMPLATES_DIR` is set and anything in the bundle is invalid, the server
**refuses to start** with a message beginning `UI_TEMPLATES_DIR bundle
invalid:`. That is deliberate: silently falling back to LightRAG content would
leave you believing customer branding is live when it is not.

| Message (abridged) | Cause / fix |
|---|---|
| `directory '…' does not exist or is not a directory` | Wrong path, or the container mount is missing. Check it from inside the container: `docker compose exec lightrag ls /app/data/ui_templates`. |
| `manifest.json is missing` | The bundle root must contain `manifest.json` directly, not one level down. A common cause is copying the parent directory. |
| `manifest.json is not valid JSON` | A trailing comma or a comment. JSON allows neither. |
| `unknown field(s) [...]` | A typo in a field name; the schema is closed on purpose. |
| `missing required field(s) [...]` | Add the field. Note `brand.logo` is required — use an explicit `null` for "no logo". |
| `unsupported schema_version` | Must be exactly `1`. |
| `locales: key 'zh_TW' uses the underscore form` | Write `zh-TW`. |
| `locales: key 'zh-tw' must be written in its normalized form 'zh-TW'` | Fix the casing. |
| `default_locale '…' is not a declared locale` | `default_locale` must appear in `locales`. |
| `fallbacks.xx: target 'yy' is not a declared locale` | Fallback targets must be declared; only sources may be uncovered. |
| `…: '…' does not exist or is not a file` | A manifest path points at nothing. Paths are relative to the bundle root and case-sensitive. |
| `…: absolute paths are not allowed` / `path traversal is not allowed` / `escapes the bundle directory` | Keep every referenced file inside the bundle, including symlink targets. |
| `…: file exceeds the … byte limit` | 64 KiB per template, 2 MiB per logo. |
| `…: file is not valid UTF-8` | Re-save the Markdown as UTF-8. |
| `logo '…': content is not PNG, JPEG, WebP or SVG` | The bytes do not match any accepted format — e.g. a `.svg` that is really HTML, or an SVG whose root element is missing or prefixed. |
| `locales.xx.login: template file is empty` | `login` / `agreements` are declared-or-absent switches; a blank declared file is rejected. `welcome` / `query_empty` may be blank. |
| `locales.xx.logo_alt must be a non-empty string` | Give each locale real alt text. |

Symptoms that are **not** startup failures:

| Symptom | Cause |
|---|---|
| Server starts, but the page still shows LightRAG branding | `UI_TEMPLATES_DIR` is unset or empty — check the log line and `/ui/customization`. In Docker, remember a compose `environment:` entry overrides `.env`. |
| Edits do not appear | No hot reload. Restart the server (all workers). |
| Consent checkbox missing | The resolved locale declares only one of `login` / `agreements`, or auth is disabled (`AUTH_ACCOUNTS` unset), or the visitor resolved to a different locale than you expected — check `locale` and `consent_required` in the endpoint response. |
| Content is in your language, buttons are not | The locale is outside the WebUI's interface languages — see [§5.3](#53-interface-languages-vs-bundle-locales) and the startup warning. |
| Logo does not show | The locale (or `brand`) resolves to `null`, or the browser failed to load the asset URL — fetch `logo_url` directly and check the status. |

---

## 10. Content authoring rules

- **Markdown with GFM** (tables, strikethrough, task lists). Links and images
  work normally; links open in a new tab.
- **Raw HTML is dropped.** This is a format boundary, not distrust: the
  customization tier simply does not open an HTML path. Express layout with
  Markdown.
- **Direction is not yours to set.** No `dir` attribute, no CSS — see
  [§5.2](#52-text-direction-rtl).
- **Keep it short.** The welcome text sits above the fold on a phone; the query
  empty state is a single centred paragraph under the logo.

---

## 11. Security notes

- The bundle is **trusted deployment content** — the same tier as `.env` and
  your compose files. Validation exists to catch configuration mistakes, not
  to defend against the bundle's own author.
- Its content is served **without authentication**, because the welcome and
  login pages precede sign-in. Never put secrets, internal hostnames or
  private paths in it.
- Only files the manifest references are ever read or served. The endpoint
  cannot be used to read arbitrary server files, and paths escaping the bundle
  root are rejected at load time.
- Logo responses carry `X-Content-Type-Options: nosniff` and a restrictive
  `Content-Security-Policy`, so an SVG served here stays inert even if opened
  as a top-level document.

---

## 12. See also

- [`docs/ui_templates_example/`](./ui_templates_example/) — a complete,
  copyable bundle (`en`, `zh`, `zh-TW`).
- [LightRAG-API-Server.md](./LightRAG-API-Server.md) — the full server guide,
  including `AUTH_ACCOUNTS` and `TOKEN_SECRET`.
- [DockerDeployment.md](./DockerDeployment.md) — Docker and Compose deployment.
- [InteractiveSetup.md](./InteractiveSetup.md) — the `make env-*` wizard and
  `docker-compose.final.yml`.
