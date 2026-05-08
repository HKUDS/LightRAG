/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_PROXY: string
  readonly VITE_API_ENDPOINTS: string
  readonly VITE_BACKEND_URL: string

  /**
   * Browser-visible URL prefix used as `axios.baseURL`, in `fetch()` template
   * strings, and as the iframe `src` for the API docs page.
   *
   * Must equal the backend `LIGHTRAG_API_PREFIX` for the same site — i.e.
   * the prefix the reverse proxy strips before forwarding to FastAPI.
   * Empty / `/` → no prefix (single-instance deployment).
   *
   * Multi-site example (site01 routed at `https://host/site01/...`):
   *     VITE_API_PREFIX=/site01
   *
   * Statically replaced into the bundle at `bun run build` time, so each
   * site needs its own WebUI build until runtime config injection lands.
   * See `lightrag_webui/.env.example` and the project root `env.example`.
   *
   * Always read through {@link normalizeApiPrefix}; do not concatenate the
   * raw value, otherwise `/` produces protocol-relative `//path` and a
   * trailing slash produces `//`.
   */
  readonly VITE_API_PREFIX?: string

  /**
   * Browser-visible URL prefix where the WebUI itself is served. Used as
   * Vite's `base` option (asset paths in `index.html`) and by `<a>` links.
   *
   * Must equal `LIGHTRAG_API_PREFIX + LIGHTRAG_WEBUI_PATH + "/"` — i.e. the
   * FULL path the user's browser sees, including any reverse-proxy prefix
   * in front of the in-app mount path.
   *
   * Multi-site example (site01 WebUI at `https://host/site01/webui/`):
   *     VITE_WEBUI_PREFIX=/site01/webui/
   *
   * Statically replaced into the bundle at `bun run build` time, so each
   * site needs its own WebUI build until runtime config injection lands.
   * See `lightrag_webui/.env.example` and the project root `env.example`.
   *
   * Always read through {@link normalizeWebuiPrefix}; the helper guarantees
   * a leading `/` and exactly one trailing `/` as Vite requires.
   */
  readonly VITE_WEBUI_PREFIX?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
