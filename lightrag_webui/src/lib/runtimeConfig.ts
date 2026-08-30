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
 * Always read through {@link normalizeApiPrefix} downstream (see
 * `constants.ts`); this module returns the raw injected value.
 */

declare global {
  interface Window {
    __LIGHTRAG_CONFIG__?: {
      apiPrefix?: string
      /**
       * Still injected by the server and pinned as part of the published
       * `{apiPrefix, webuiPrefix}` shape (docs/MultiSiteDeployment.md), so
       * the declaration must keep describing what actually arrives — but
       * nothing reads it any more. Both former consumers (SiteHeader, the
       * initializing header in App) now link to their own entry root with
       * the document-relative `entryHomeHref`, which is what makes one
       * build serve two entries without either linking into the other.
       */
      webuiPrefix?: string
      /**
       * `WEBUI_TITLE`, or absent/null when the deployment sets none. The
       * server's snippet already assigned it to `document.title` before this
       * bundle loaded; it is published here so the SPA can fall back to it
       * while the auth store has no title yet (see `documentTitle`).
       */
      webuiTitle?: string | null
    }
  }
}

const config =
  (typeof window !== 'undefined' && window.__LIGHTRAG_CONFIG__) || {}

/** Browser-visible API prefix; empty string means same-origin / no prefix. */
export function getRuntimeApiPrefix(): string | undefined {
  return config.apiPrefix
}

/**
 * Deployment title (`WEBUI_TITLE`), or undefined/null when none is set.
 *
 * Read from the live global rather than the snapshot above: this value is a
 * display string consulted on demand, so nothing is gained by freezing it at
 * module-init time, and reading it live keeps the lookup correct whichever
 * order the snippet and the bundle happen to run in.
 */
export function getRuntimeWebuiTitle(): string | null | undefined {
  if (typeof window === 'undefined') return undefined
  return window.__LIGHTRAG_CONFIG__?.webuiTitle
}
