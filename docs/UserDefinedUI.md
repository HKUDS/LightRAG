# User-Defined UI Content (`UI_TEMPLATES_DIR`)

LightRAG can serve **your own** welcome page, login-page text, user agreement, query empty state, copyright line and brand logo — in several languages — without rebuilding the WebUI. You write a small directory of Markdown files plus a `manifest.json`, point the `UI_TEMPLATES_DIR` environment variable at it, and restart the server.

This document is the complete guide: what can be customized, the exact bundle format, how to deploy it from source / Docker / Kubernetes, how to verify it, and what every startup error means.

A ready-to-copy bundle lives in [`docs/ui_templates_example/`](./ui_templates_example/).

---

## 1. What you can customize

| Surface | Where the visitor sees it | Bundle field |
|---|---|---|
| **Welcome page** | `/workspace` entry, before sign-in (`/workspace/#/welcome`) | `welcome` (required) |
| **Query empty state** | `/workspace` query page, while the conversation is empty (and again after *Clear*) | `query_empty` (required) |
| **Login page text** | Above the username/password form, on **both** entries (`/webui/#/login` and `/workspace/#/login`) | `login` (optional) |
| **User agreement + consent checkbox** | A checkbox on the login page, whose link opens the document in a dialog; the WebUI's Login button stays disabled until it is ticked — a WebUI prompt, not server-side enforcement ([§6](#6-the-login-consent-gate)) | `agreements` (optional, pairs with `login`) |
| **Consent checkbox link text** | The words the checkbox links — *"I agree to …"* | `consent_documents` (optional; falls back to the WebUI's generic *"Privacy Policy Agreement"*) |
| **Brand logo** | Welcome page, query empty state and login page | `brand.logo`, per-locale `logo`, `logo_alt` |
| **Copyright line** | Foot of the welcome and login pages — **outside** the card, at the bottom edge of the page | `brand.copyright`, per-locale `copyright` (optional) |

What you **cannot** set from the bundle:

- The browser tab title and the login card's heading and subtitle. The manifest cannot set them: `WEBUI_TITLE` supplies both the browser tab title and login heading (falling back to `LightRAG`), while the subtitle remains the localized `login.description` in `LoginPage.tsx`.
- The in-app header shown **after sign-in** (`SiteHeader` on `/webui`, the workspace header on `/workspace`). That one is set by the `WEBUI_TITLE` / `WEBUI_DESCRIPTION` environment variables, and the manifest deliberately cannot override them.
- The buttons, menus and settings around your content — see [§5.3 Interface languages](#53-interface-languages-vs-bundle-locales).
- Text direction. It is derived from the locale (see [§5.2](#52-text-direction-rtl)), never taken from bundle markup.

**All-or-nothing per locale.** A visitor either sees your bundle's representation of a locale *entirely*, or LightRAG's built-in branding *entirely*. Fields are never mixed one by one, so you cannot supply only a logo and inherit LightRAG's welcome text.

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

The bundled compose files already mount the bundle directory read-only **and**
point `UI_TEMPLATES_DIR` at it:

```yaml
    volumes:
      - ./data/ui_templates:/app/data/ui_templates:ro
    environment:
      UI_TEMPLATES_DIR: "/app/data/ui_templates"
```

So writing the bundle is the whole procedure — there is no compose file to edit:

```bash
mkdir -p ./data/ui_templates
cp -r docs/ui_templates_example/* ./data/ui_templates/
docker compose up -d --force-recreate lightrag
```

Until you do, the directory holds no `manifest.json`, the server serves its
built-in branding and says so once at startup. Nothing about the default
deployment changes.

See [§7 Deployment](#7-deployment) for the full picture, including the wizard-generated `docker-compose.final.yml` and Kubernetes.

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

The directory names above are a convention, not a rule: every file is located through `manifest.json`, and a file the manifest does not reference is never read and never served. Paths in the manifest are relative to the bundle root; absolute paths, `..` segments and symlinks pointing outside the bundle are all rejected at startup.

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
    "logo": "assets/logo.svg",
    "copyright": "© 2025 Example Corp. All rights reserved."
  },
  "locales": {
    "en": {
      "welcome": "locales/en/welcome.md",
      "query_empty": "locales/en/query_empty.md",
      "login": "locales/en/login.md",
      "agreements": "locales/en/agreements.md",
      "consent_documents": "Privacy Policy and Model Service Agreement",
      "logo_alt": "Example Corp"
    },
    "zh": {
      "welcome": "locales/zh/welcome.md",
      "query_empty": "locales/zh/query_empty.md",
      "login": "locales/zh/login.md",
      "agreements": "locales/zh/agreements.md",
      "consent_documents": "《用户隐私协议》和《模型服务协议》",
      "logo_alt": "示例公司"
    }
  }
}
```

> JSON has no comments — the `jsonc` above is for reading only. Keep your real manifest as plain JSON.

### 4.1 Top-level fields

| Field | Required | Type | Notes |
|---|---|---|---|
| `schema_version` | yes | number | Must be `1`. |
| `default_locale` | yes | string | Must be one of the declared `locales` keys. Used when a visitor's locale matches nothing. |
| `brand` | yes | object | Two keys: `logo` (required) and `copyright` (optional). |
| `brand.logo` | yes | string \| `null` | Path to the default logo, or an explicit `null` for "this deployment shows no logo". **Omitting the key fails startup** — a missing logo must never silently fall back to the LightRAG logo underneath your texts. |
| `brand.copyright` | no | string \| `null` | The copyright line for every locale, as **plain text written in the manifest** (not a Markdown file). Omitted, `null`, empty or whitespace-only all mean the same thing: **no line is shown**. There is no LightRAG default to inherit — see [§4.5](#45-the-copyright-line). |
| `locales` | yes | object | Non-empty map of locale → entry. |
| `fallbacks` | no | object \| `null` | Maps *uncovered* locales onto declared ones. See [§5.1](#51-locale-resolution). |

Unknown top-level fields are an error, so a typo (`defaultLocale`) is reported at startup rather than silently ignored.

### 4.2 Locale entries

| Field | Required | Type | Notes |
|---|---|---|---|
| `welcome` | yes | string | Path to the welcome-page Markdown. |
| `query_empty` | yes | string | Path to the query-empty-state Markdown. |
| `logo_alt` | yes | string | Non-empty alt text for the logo, in this locale's language. |
| `logo` | no | string \| `null` | Overrides `brand.logo` for this locale (`null` = no logo here). When the key is **absent**, the locale inherits `brand.logo`. |
| `login` | no | string \| `null` | Path to the login-page blurb. |
| `agreements` | no | string \| `null` | Path to the single user-agreement document. Absent or `null` means this locale has no agreement document: the gate is off and the login page shows no checkbox (see [§6](#6-the-login-consent-gate)). **There is no discovery by conventional filename** — an `agreements.md` that happens to sit in the bundle is never read unless the manifest references it (see [§3](#3-bundle-layout)). |
| `consent_documents` | no | string \| `null` | **Inline text, not a path**: how the consent checkbox names the document it links to, in this locale's language. Declared-but-blank fails startup. When absent or `null`, the WebUI's own translation is used. |
| `copyright` | no | string \| `null` | Overrides `brand.copyright` for this locale (`null` = no line here). When the key is **absent**, the locale inherits `brand.copyright`. |

`login` and `agreements` together switch on the login consent gate — see [§6](#6-the-login-consent-gate). `consent_documents` only *labels* that gate; it never turns it on.

### 4.3 Locale keys

Keys are BCP 47 tags in **hyphen** form, written exactly in their normalized shape:

- lowercase language (`zh`, `en`, `ar`);
- Titlecase 4-letter script (`Hant`, `Arab`);
- UPPERCASE 2-letter region (`TW`, `CN`);
- `zh_TW` (underscore) is rejected — write `zh-TW`;
- `zh-tw` is rejected too: the key must already be normalized (`zh-TW`).

What the server applies is a **shape and case check**, not full BCP 47 validation. It is worth being precise about, because the two differ in both directions:

| Rule | Accepted | Rejected |
|---|---|---|
| Primary subtag is 2–8 letters | `en`, `zh`, `art-lojban` | `x-acme`, `i-klingon` — a single-letter first subtag |
| Every later subtag is 1–8 alphanumerics | `zh-Hant-TW`, `de-CH-1901`, `sl-rozaj-biske` | `abcdefghi` |
| The whole tag is ≤ 35 characters | | anything longer |

Script, region, variant and extension subtags all work — `zh-Hant-TW`, `ar-aao-Latn`, `en-US-u-VA-posix`.

The exclusion is precisely **a first subtag of one letter**, which is what rules out private-use tags (`x-acme`) and the *irregular* grandfathered ones (`i-klingon`). It is not "grandfathered tags" as a class: the regular grandfathered tags begin with a normal language subtag and are accepted — `art-lojban`, `en-GB-oed` and `zh-min-nan` all pass.

In the other direction, the check is looser than BCP 47: `en-u` is accepted even though a well-formed tag must follow an extension singleton with at least one subtag. Nothing depends on that being caught — an unknown locale simply falls back — so treat the table as the contract and BCP 47 as the convention the table approximates.

Note that normalization is positional and applies to *every* subtag, including an extension's own values: `en-US-u-va-posix` normalizes to `en-US-u-VA-posix`, and since the key must already be normalized, that is the form to write.

Read [§5.3](#53-interface-languages-vs-bundle-locales) before declaring a locale outside the WebUI's own languages.

### 4.4 Limits and formats

| Rule | Value |
|---|---|
| Markdown template size | ≤ 64 KiB per file |
| `manifest.json` size | ≤ 64 KiB |
| Logo size | ≤ 2 MiB per file |
| Template encoding | UTF-8 (invalid UTF-8 fails startup) |
| Logo formats | PNG, JPEG, WebP, SVG — **detected from the file's bytes, not its extension** |

For SVG, the file must actually open an `<svg>` root element (an XML declaration, comments, a DOCTYPE or processing instructions may precede it). A namespace-prefixed root (`<s:svg>`) or a UTF-16/32-encoded file is not recognized — no SVG tool emits those.

### 4.5 The copyright line

The copyright line is **your deployment's own legal statement**, so LightRAG never writes one for you: without a bundle — or with a bundle that says nothing about copyright — the pages show no copyright line at all, and LightRAG's own notice is never printed on your pages.

Where it appears: at the bottom edge of the **welcome page** and the **login page**, *outside* the card, in small muted type. It is pre-login content, like everything else in the bundle; the in-app pages after sign-in do not show it.

How to declare it:

```jsonc
{
  "brand": {
    "logo": "assets/logo.svg",
    "copyright": "© 2025 Example Corp. All rights reserved."   // every locale
  },
  "locales": {
    "zh": {
      "copyright": "© 2025 示例公司 版权所有"                    // this locale instead
    },
    "en": {
      "copyright": null                                        // no line for en
    }
  }
}
```

The rules, in short:

- **Plain text, not Markdown, and not a file.** It is one short line, written directly in `manifest.json`; there is no template file to point at, and no Markdown is rendered — a footer is not a place for headings, images or links.
- **Same three-way shape as `logo`:** key absent → inherit `brand.copyright`; a string → use it; `null` → no line for this locale.
- **Empty means absent.** `""` or whitespace is treated exactly like omitting the field: no line. Unlike a blank `login` / `agreements` file, it does **not** fail startup — blankness here only turns something off, so there is nothing to be silently mis-shown.
- **Surrounding whitespace is stripped**, so indentation in the manifest never leaks into the page.
- It follows the **resolved** locale: a visitor who falls back from `ko` to `zh` sees the `zh` line (see [§5.1](#51-locale-resolution)).

---

## 5. Languages

### 5.1 Locale resolution

The WebUI asks for one locale: the interface language it resolved for this visitor (explicit setting in the UI > browser language > `en`). The server then resolves it in a **single hop**:

1. **Exact match** against a declared locale → used.
2. Otherwise the first *declared* target listed under `fallbacks.<requested>`.
3. Otherwise `default_locale`.

Every `fallbacks` target must be a declared locale, which is what makes resolution single-hop and cycles impossible. The *source* side may be any locale — pointing uncovered languages somewhere sensible is the entire purpose of the map:

```json
"fallbacks": {
  "ko": ["en"],
  "ja": ["en"],
  "de": ["en"]
}
```

Without an entry, an uncovered locale simply lands on `default_locale`, so `fallbacks` only matters when different uncovered languages should land in *different* places (e.g. `zh-HK` → `zh-TW`, everything else → `en`).

### 5.2 Text direction (RTL)

Direction is derived from the resolved locale against a CLDR-derived registry and sent to the browser; the bundle cannot set it. An explicit script subtag wins over the language, which is the escape hatch when the default is wrong:

- `ar`, `he`, `fa`, `ur`, `ps`, `ckb`, `dv`… → right-to-left;
- `ku` → left-to-right, but `ku-Arab` → right-to-left;
- `az-Arab`, `pa-Arab`, `ha-Arab` likewise.

### 5.3 Interface languages vs bundle locales

The bundle's set of languages and the WebUI's are **independent**. The WebUI ships interface translations (buttons, settings, login labels, the consent checkbox wording) for:

`en`, `zh`, `zh-TW`, `fr`, `ar`, `ru`, `ja`, `de`, `uk`, `ko`, `vi`

A bundle may declare a locale outside that list — say `nl`. Its content will render correctly, direction included, but the controls around it stay in the visitor's resolved interface language, because no Dutch interface translation exists to switch to. Startup logs a warning naming such locales:

```
WARNING: UI customization: the WebUI ships no interface translation for ['nl'] …
```

Declare a locale from the supported list whenever you want the whole page in one language.

---

## 6. The login consent gate

When a locale declares **both** `login` and `agreements`, the login page for that locale shows:

- your `login` Markdown above the form, and
- a checkbox reading (English UI) *"I agree to the …"*, whose single link — named by `consent_documents`, or by the WebUI's own translation when you declare none — opens your `agreements` document in a dialog.

The **Login** button stays disabled until the box is ticked, and pressing Enter in the form is refused the same way.

> **What this gate is, precisely — read before relying on it.** It is a **WebUI control, not server-side enforcement.** The server computes `consent_required` and the WebUI obeys it, but `POST /login` takes only the standard credential fields: it neither requires nor records acceptance, and stores nothing about which revision of the document a user agreed to. A client posting credentials straight to `/login` — curl, a script, the Ollama-compatible API, another frontend — receives a token without ever seeing the checkbox.
>
> So treat it as an **informed-consent prompt for people using the WebUI**, not as an access control, and do not treat it as evidence that a particular user accepted a particular revision. If your deployment needs enforceable, auditable acceptance, it has to be built server-side; this feature does not provide it.

### 6.1 One document, not two

The checkbox carries exactly **one** link, so everything the visitor has to agree to goes into that single `agreements.md`, separated by headings:

```markdown
# Privacy Policy and Model Service Agreement

## Privacy Policy

…

## Model Service Agreement

…
```

> **A merged document must name itself.** The WebUI's fallback link text is the generic *"Privacy Policy Agreement"*. The moment your file covers anything beyond a privacy policy — a model service agreement, terms of service, an acceptable-use policy — that fallback understates what the visitor is ticking, so set `consent_documents` to the real name.
>
> This includes bundles written before the field existed: they keep loading unchanged, but their checkbox now reads *"Privacy Policy Agreement"*. **If your `agreements.md` merges several documents, add `consent_documents` when you upgrade.**

#### Naming the link

`consent_documents` is the text the checkbox turns into the link — set it to whatever your deployment actually calls the document:

```jsonc
"consent_documents": "Example Corp Terms of Service"
```

The sentence around it (*"I agree to …"*) still comes from the WebUI's own translations, so it reads naturally in each interface language; only the document's name is yours. Leave the field out and that translation names the link too — the generic *"Privacy Policy Agreement"* (`《隐私政策协议》` in Chinese, and so on), which fits a file that really is just a privacy policy and understates every file that is not (see the note above).

#### What the dialog shows

The dialog renders `agreements.md` **as written** — no title is printed above it. So give the file its own heading; that heading is the document's title on screen:

```markdown
# Privacy Policy and Model Service Agreement

## Privacy Policy

…
```

Nothing reads the document — the dialog *renders* it. Its name for screen readers is the checkbox's own link text (`consent_documents`, or the WebUI's fallback), which is also what the visitor just ticked. **Keeping that name consistent with the file's heading, and with what the file actually contains, is yours to maintain.**

Headings, paragraphs, lists, tables, block quotes, code blocks, horizontal rules and links all render with the standard document typography. Raw HTML inside the Markdown is dropped, as in every bundle template — see [§10](#10-content-authoring-rules).

### 6.2 Rules to know

- **Both or neither.** Declaring only `login` gives a branded login page with no gate; declaring only `agreements` gives a document nothing links to. Neither turns the gate on — a half-configuration is treated as a half-configuration, not as consent.
- **Per locale.** A visitor resolving to a locale that declares neither field sees no checkbox. Declare the pair for every locale that must be gated, or route the uncovered ones there with `fallbacks`.
- **A declared-but-empty file fails startup.** The gate must never point at a blank document. The same holds for a blank `consent_documents`.
- **The link text is optional, the documents are not.** `consent_documents` on its own never switches the gate on, and a locale that declares the pair without it still gets a working checkbox — labelled by the WebUI translation.
- **The tick is not remembered.** It lives only as long as the login page and is bound to the exact document text on screen, so it is asked for on every visit. Switching the interface language mid-page replaces the document and clears the tick, unless the new locale's text is byte-for-byte identical.
- **WebUI only, not the API.** As above: `POST /login` has no consent field, so the gate constrains the WebUI's login form and nothing else.
- **Credentialed sign-in only.** A deployment with authentication disabled (`AUTH_ACCOUNTS` unset) admits visitors as guests without the gate. That is deliberate rather than a gap: with no authentication there is no identified user to bind an agreement to, and auth-disabled is a development/demo posture. **If the agreement must be accepted, configure `AUTH_ACCOUNTS`** (with `TOKEN_SECRET`).
- **An unreachable endpoint fails open only on the FIRST load.** If the very first customization request fails, no snapshot exists, the frontend falls back to its own default content, and the gate stays off rather than locking everyone out of the deployment.
- **A failed LANGUAGE SWITCH keeps the last good verdict instead.** Once a snapshot has loaded, a failing request for another locale leaves that snapshot on screen; when the retries are exhausted the gate goes back to obeying it. So if the previously loaded locale required consent, the checkbox stays — the visitor is held to the agreement they were last shown, not released by a network error. This is the safer of the two behaviours and is deliberate: only the case where *nothing* has ever loaded opens the gate.

---

## 7. Deployment

### 7.1 Where the bundle lives

| Deployment | Bundle directory | `UI_TEMPLATES_DIR` |
|---|---|---|
| From source | `lightrag_webui/ui_templates/` (git-ignored) | `./lightrag_webui/ui_templates` |
| Docker / Compose | host `./data/ui_templates/` → container `/app/data/ui_templates` | `/app/data/ui_templates` |
| Kubernetes | ConfigMap or PVC mounted at `/app/data/ui_templates` | `/app/data/ui_templates` |

Relative paths are resolved against the server's working directory, so an absolute path is the safer choice whenever you are not sure where the process starts.

### 7.2 Docker Compose

`docker-compose.yml` and `docker-compose-full.yml` ship both halves already:

```yaml
services:
  lightrag:
    volumes:
      - ./data/ui_templates:/app/data/ui_templates:ro
    environment:
      UI_TEMPLATES_DIR: "/app/data/ui_templates"
```

**Both are inert until you write a bundle.** A configured directory that holds no `manifest.json` is an unpopulated mount point, not a broken bundle: the server logs a warning naming the directory and serves its built-in branding. So the default deployment starts normally, and the whole activation procedure is dropping a bundle into `./data/ui_templates` and restarting — no compose edit, ever.

The leniency stops at the manifest's edge. Once `manifest.json` exists, the bundle is validated in full and any problem in it refuses startup ([§9](#9-troubleshooting)) — a half-copied bundle never quietly serves LightRAG content.

Three practical notes:

- **Create the directory before the first `up`.** If Docker creates it for you it will be owned by `root`, and you will need `sudo` to copy files into it.
- **A mount that lands in the wrong place now shows up as branding that did not change**, not as a startup failure — that is the price of booting out of the box. The startup warning and `"customized": false` on `/ui/customization` are how you tell that state from "I have not written the bundle yet"; both name what the server actually saw.
- `:ro` is intentional — the server only ever reads the bundle.
- **Podman**: `docker-compose.podman.yml` keeps the mount *and* `UI_TEMPLATES_DIR` commented out, because Podman is stricter than Docker about a bind mount whose host source is missing — an unconditional mount would turn the feature into a startup prerequisite. There, `mkdir -p ./data/ui_templates` first, then uncomment both.

**The compose entry outranks `.env` on purpose.** A compose `environment:` entry wins over the same key in the mounted `.env`, and `UI_TEMPLATES_DIR` uses that deliberately — exactly like `WORKING_DIR`, `INPUT_DIR` and `PROMPT_DIR`. The value is a *container* path, so keeping it out of `.env` is what lets one `.env`, holding host paths such as `./lightrag_webui/ui_templates`, serve a source run and this deployment at the same time. Setting `UI_TEMPLATES_DIR` in `.env` therefore affects the source run only; the container uses the compose value.

To point the container at a different bundle, edit the compose entry (the wizard preserves it — see [§7.3](#73-the-wizard-generated-compose-file)) or change the mount's host side.

### 7.3 The wizard-generated compose file

`make env-base` / `make env-storage` / `make env-server` generate `docker-compose.final.yml`. The generator now adds the same read-only mount if it is not already present, so regenerating an existing file picks it up:

```bash
make env-server        # or any other make env-* target
grep ui_templates docker-compose.final.yml
```

The wizard also seeds `UI_TEMPLATES_DIR: "/app/data/ui_templates"` into the `lightrag` service's `environment:` block, so a wizard-generated deployment behaves exactly like the shipped compose files: inert until `./data/ui_templates` holds a bundle.

**Seeded, not managed — the wizard never changes a value you set.** `WORKING_DIR` / `INPUT_DIR` / `PROMPT_DIR` are rewritten on every run, so a hand-edited value there does not survive. `UI_TEMPLATES_DIR` is written only when the compose file does not already declare the key:

- A value you edited by hand in `docker-compose.final.yml` survives every regeneration — including `UI_TEMPLATES_DIR: ""`, which is how a deployment turns the feature off, and `- UI_TEMPLATES_DIR` in a list-style block.
- Serving a bundle from another host directory needs no environment edit at all: change the mount's *host* side (`./my-branding:/app/data/ui_templates:ro`).
- The seeded value still overrides `.env`, which is what keeps a host-path `UI_TEMPLATES_DIR` in `.env` usable for source runs ([§7.2](#72-docker-compose)).

User-added bind mounts and other user-added environment keys are preserved across regenerations as before.

### 7.4 Kubernetes

The bundled Helm chart has no dedicated value for this yet, so mount the bundle by patching the deployment.

**A ConfigMap has no directories.** Its keys are flat, and every key becomes a file directly under the mount — `--from-file=<dir>` packages that directory's files under their bare basenames and skips subdirectories entirely. So a bundle delivered as a ConfigMap must be flat, with a manifest whose paths are exactly those keys. Copying the nested example layout as-is fails startup with `'locales/en/welcome.md' does not exist or is not a file`.

Write a flat manifest for this deployment:

```json
{
  "schema_version": 1,
  "default_locale": "en",
  "brand": { "logo": "logo.svg" },
  "locales": {
    "en": {
      "welcome": "welcome.en.md",
      "query_empty": "query_empty.en.md",
      "login": "login.en.md",
      "agreements": "agreements.en.md",
      "logo_alt": "Example Corp"
    }
  }
}
```

and name every referenced file explicitly, the logo included — the `key=path` form does the flattening, so your source tree can stay nested. A file the manifest references but the ConfigMap omits fails startup:

```bash
kubectl create configmap lightrag-ui-templates \
  --from-file=manifest.json=./k8s/ui-manifest.json \
  --from-file=welcome.en.md=./ui_templates/locales/en/welcome.md \
  --from-file=query_empty.en.md=./ui_templates/locales/en/query_empty.md \
  --from-file=login.en.md=./ui_templates/locales/en/login.md \
  --from-file=agreements.en.md=./ui_templates/locales/en/agreements.md \
  --from-file=logo.svg=./ui_templates/assets/logo.svg \
  --dry-run=client -o yaml | kubectl apply -f -
```

`kubectl` places a non-UTF-8 logo (PNG, JPEG, WebP) into the ConfigMap's `binaryData` automatically. Note that a ConfigMap is capped at ~1 MiB in total, well below this feature's 2 MiB per-logo limit: a bundle with a large logo needs a PVC instead, which also lets you keep the nested layout.

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

Keep `mountPath` equal to `UI_TEMPLATES_DIR`. The projected volume's internal `..data` symlinks stay inside the mount, so the bundle's containment checks are satisfied — a flat ConfigMap mount loads exactly like a directory on disk. Updating the ConfigMap does **not** reload the bundle; restart the pods (§7.5).

### 7.5 Applying changes

There is **no hot reload**. The whole bundle is validated once at startup and activated as an immutable in-memory snapshot; request handling never touches the disk again. After editing the bundle, restart the server — and with `lightrag-gunicorn` or any multi-worker setup, restart **all** workers, or some of them will keep serving the previous revision.

No cache purge is needed: the content response is sent `Cache-Control: no-store`, and logo URLs embed the file's content hash, so changed bytes produce a new URL.

---

## 8. Verifying the deployment

**Startup log.** One of these lines always appears:

```
INFO:    UI customization: no bundle configured (UI_TEMPLATES_DIR unset)
WARNING: UI customization: UI_TEMPLATES_DIR=/app/data/ui_templates holds no manifest.json — serving the built-in LightRAG branding. …
INFO:    UI customization bundle active: bundle_revision=<sha256> locales=['en', 'zh']
```

The middle line is the Docker default state: the variable is set by the shipped compose files, and the mounted directory is still empty. It is a warning rather than an info line because the same state is what a mount pointing at the wrong host directory produces — it names the directory the server actually read so you can tell the two apart.

`bundle_revision` is a hash over every referenced file. If it does not change after you edited a file, the server is not reading the directory you think it is — or it was not actually restarted.

**The endpoint.** It is public (the welcome page is shown before login), so a plain `curl` works:

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
    "logo_alt": "示例公司",
    "copyright": "© 2025 示例公司 版权所有"
  },
  "welcome": { "format": "markdown", "content": "## 欢迎…" },
  "query_empty": { "format": "markdown", "content": "…" },
  "login": { "format": "markdown", "content": "…" },
  "agreements": { "format": "markdown", "content": "…" },
  "consent_documents": "《用户隐私协议》和《模型服务协议》",
  "consent_required": true
}
```

Useful checks:

- `"customized": false` → no bundle is active (`UI_TEMPLATES_DIR` unset, or pointed at a directory with no `manifest.json`). The frontend is showing LightRAG's built-in branding.
- `"fallback_used": true` → the requested locale is not declared; `locale` tells you where it landed.
- `"consent_required"` → whether the checkbox will appear for this locale.
- `"consent_documents": null` → this locale declares no link text; the WebUI names the link from its own translation.
- `logo_url: null` → this locale resolves to *no* logo (an explicit `null` somewhere), not to the LightRAG logo.
- `brand.copyright: null` → this locale shows no copyright line. On a `"customized": false` response the key is absent entirely, which means the same thing: nothing to render.

**In the browser.** Visit `/workspace` for the welcome page, `/workspace/#/login` or `/webui/#/login` for the login page, and switch the interface language from the settings menu to check each locale.

---

## 9. Troubleshooting

Once the configured directory contains a `manifest.json`, anything invalid in the bundle makes the server **refuse to start** with a message beginning `UI_TEMPLATES_DIR bundle invalid:`. That is deliberate: silently falling back to LightRAG content would leave you believing customer branding is live when it is not.

A configured directory *without* `manifest.json` is the one exception — the unpopulated mount every default Docker deployment starts with. It is not a startup failure; see the second table below.

| Message (abridged) | Cause / fix |
|---|---|
| `directory '…' does not exist or is not a directory` | Wrong path, or the container mount is missing. Check it from inside the container: `docker compose exec lightrag ls /app/data/ui_templates`. |
| `manifest.json is not valid JSON` | A trailing comma or a comment. JSON allows neither. |
| `unknown field(s) [...]` | A typo in a field name; the schema is closed on purpose. |
| `missing required field(s) [...]` | Add the field. Note `brand.logo` is required — use an explicit `null` for "no logo". |
| `unsupported schema_version` | Must be exactly `1`. |
| `locales: key 'zh_TW' uses the underscore form` | Write `zh-TW`. |
| `locales: key 'x-acme' has an invalid language subtag` | The primary subtag must be 2–8 letters, so a one-letter first subtag is rejected — private-use (`x-…`) and irregular grandfathered (`i-…`) tags. Regular grandfathered tags such as `art-lojban` are fine — see [§4.3](#43-locale-keys). |
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
| `locales.xx.consent_documents must be a non-empty string or null` | It is inline text, not a path — a blank label would leave the checkbox naming nothing. Remove the key to fall back to the WebUI translation. |
| `brand.copyright must be a string or null` / `locales.xx.copyright must be a string or null` | The copyright line is plain text written in the manifest — a string, or `null` for "no line". Not a path, not a number, not a list. |

Symptoms that are **not** startup failures:

| Symptom | Cause |
|---|---|
| Server starts, but the page still shows LightRAG branding | `UI_TEMPLATES_DIR` is unset — check the log line and `/ui/customization`. In Docker, remember a compose `environment:` entry overrides `.env`. |
| Startup logs `holds no manifest.json` and the branding is unchanged | The configured directory exists but has no bundle in it. Either you have not written one yet, or the server is reading a different directory than you populated — the warning names the path it read. `manifest.json` must sit directly in the bundle root, not one level down (a common cause is copying the parent directory). Check it from inside the container: `docker compose exec lightrag ls /app/data/ui_templates`. |
| Edits do not appear | No hot reload. Restart the server (all workers). |
| Checkbox says "Privacy Policy Agreement", but the document covers more | That is the WebUI fallback for a locale declaring no `consent_documents`. Name the document yourself ([§6.1](#61-one-document-not-two)) — bundles written before the field existed hit this on upgrade. |
| Consent dialog has no title | `agreements.md` starts with no heading. The dialog prints the file as written — add a `# Title` line to it ([§6.1](#61-one-document-not-two)). |
| An `agreements.md` sits in the directory, but no checkbox ever appears | That locale's manifest entry does not reference it. `manifest.json` is the only index — there is no discovery by filename, so declare `agreements` (and `login`) explicitly ([§4.2](#42-locale-entries)). |
| Consent checkbox missing | The resolved locale declares only one of `login` / `agreements`, or auth is disabled (`AUTH_ACCOUNTS` unset), or the visitor resolved to a different locale than you expected — check `locale` and `consent_required` in the endpoint response. |
| Content is in your language, buttons are not | The locale is outside the WebUI's interface languages — see [§5.3](#53-interface-languages-vs-bundle-locales) and the startup warning. |
| Logo does not show | The locale (or `brand`) resolves to `null`, or the browser failed to load the asset URL — fetch `logo_url` directly and check the status. |
| Copyright line does not show | Neither `brand.copyright` nor the resolved locale's `copyright` declares text (or one of them is `null` / empty). Check `brand.copyright` in the endpoint response for the locale you are looking at. There is no LightRAG default: no declaration means no line ([§4.5](#45-the-copyright-line)). |

---

## 10. Content authoring rules

- **Markdown with GFM** (tables, strikethrough, task lists). Links and images work normally; links open in a new tab.
- **Raw HTML is dropped.** This is a format boundary, not distrust: the customization tier simply does not open an HTML path. Express layout with Markdown.
- **Direction is not yours to set.** No `dir` attribute, no CSS — see [§5.2](#52-text-direction-rtl).
- **Keep it short.** The welcome text sits above the fold on a phone; the query empty state is a single centred paragraph under the logo. The agreement document is the exception — it renders as a scrollable document in its own dialog, and its own headings structure it ([§6.1](#61-one-document-not-two)).

---

## 11. Security notes

- The bundle is **trusted deployment content** — the same tier as `.env` and your compose files. Validation exists to catch configuration mistakes, not to defend against the bundle's own author.
- Its content is served **without authentication**, because the welcome and login pages precede sign-in. Never put secrets, internal hostnames or private paths in it.
- Only files the manifest references are ever read or served. The endpoint cannot be used to read arbitrary server files, and paths escaping the bundle root are rejected at load time.
- Logo responses carry `X-Content-Type-Options: nosniff` and a restrictive `Content-Security-Policy`, so an SVG served here stays inert even if opened as a top-level document.

---

## 12. See also

- [`docs/ui_templates_example/`](./ui_templates_example/) — a complete, copyable bundle (`en`, `zh`, `zh-TW`).
- [LightRAG-API-Server.md](./LightRAG-API-Server.md) — the full server guide, including `AUTH_ACCOUNTS` and `TOKEN_SECRET`.
- [DockerDeployment.md](./DockerDeployment.md) — Docker and Compose deployment.
- [InteractiveSetup.md](./InteractiveSetup.md) — the `make env-*` wizard and `docker-compose.final.yml`.
