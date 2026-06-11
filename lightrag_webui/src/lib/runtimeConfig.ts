/**
 * Runtime path prefix configuration.
 *
 * The browser-visible URL prefixes (API base, WebUI mount path) used to be
 * baked into the bundle from `import.meta.env.VITE_*_PREFIX` at build time,
 * forcing one build per reverse-proxy mount point. They are now resolved at
 * REQUEST time: the FastAPI server replaces a `<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->`
 * comment in `index.html` with a `<script>window.__LIGHTRAG_CONFIG__ = ...</script>`
 * snippet built from `LIGHTRAG_API_PREFIX` / `LIGHTRAG_WEBUI_PATH`. This module
 * is the single read point for that injected value.
 *
 * Dev parity: `vite.config.ts` performs the same injection via a
 * `transformIndexHtml` plugin using `VITE_DEV_API_PREFIX` /
 * `VITE_DEV_WEBUI_PREFIX`, so dev and prod use the same lookup.
 *
 * Always read through {@link normalizeApiPrefix} / {@link normalizeWebuiPrefix}
 * downstream (see `constants.ts`); this module returns the raw injected value.
 */

declare global {
  interface Window {
    __LIGHTRAG_CONFIG__?: {
      apiPrefix?: string
      webuiPrefix?: string
    }
  }
}

const config =
  (typeof window !== 'undefined' && window.__LIGHTRAG_CONFIG__) || {}

/** Browser-visible API prefix; empty string means same-origin / no prefix. */
export function getRuntimeApiPrefix(): string | undefined {
  return config.apiPrefix
}

/** Browser-visible WebUI mount path including the trailing slash. */
export function getRuntimeWebuiPrefix(): string | undefined {
  return config.webuiPrefix
}
