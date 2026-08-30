/**
 * Path prefix normalization utilities.
 *
 * Used by both the React source (where Vite statically replaces
 * `import.meta.env.VITE_*`) and `vite.config.ts` (where the config is loaded
 * by Node and only `loadEnv()` is reliable for reading user env vars).
 *
 * Keep this module dependency-free so it can be imported from `vite.config.ts`
 * without dragging in `@/...` path aliases or React/UI types.
 */

/**
 * Normalize an API path prefix used as `axios.baseURL` and as a string-concat
 * prefix in `fetch(`${prefix}/path`)` and iframe `src`.
 *
 * - Empty / undefined / "/" → "" (avoids producing `//path` which fetch
 *   interprets as a protocol-relative URL pointing at host "path").
 * - Always returns either "" or a string with a leading "/" and no trailing "/".
 */
export function normalizeApiPrefix(value: string | undefined | null): string {
  if (!value) return ''
  const trimmed = value.trim()
  if (!trimmed || trimmed === '/') return ''
  const withLeading = trimmed.startsWith('/') ? trimmed : '/' + trimmed
  return withLeading.replace(/\/+$/, '')
}

/**
 * Normalize a WebUI mount path used as Vite's `base` option and as `<a href>`.
 *
 * Vite requires `base` to end with "/", and serving via `<a href={prefix}>`
 * also avoids a 307 redirect when the value already ends with "/".
 *
 * - Empty / undefined / "/" → fallback (default `/webui`) with trailing "/".
 * - Always returns a value with a leading "/" and exactly one trailing "/".
 */
export function normalizeWebuiPrefix(
  value: string | undefined | null,
  fallback = '/webui'
): string {
  const trimmed = (value ?? '').trim()
  const candidate =
    !trimmed || trimmed === '/'
      ? fallback
      : trimmed.startsWith('/')
        ? trimmed
        : '/' + trimmed
  return candidate.replace(/\/+$/, '') + '/'
}


/**
 * `href` for "go to THIS entry's own root", given the document's pathname.
 *
 * Production serves each entry as a directory index (`/webui/` → index.html,
 * `/workspace/` → workspace.html), where a plain `./` is exactly right. The
 * Vite dev server has no such mounts: an entry is a plain file, so the
 * workspace entry lives at `/workspace.html` and `./` there resolves to `/`
 * — which serves the ADMIN index.html. Clicking the workspace brand in
 * development would silently switch products.
 *
 * Anchoring on the document's own filename when the URL names one keeps the
 * link inside the entry under both layouts (and under the explicit
 * same-entry filenames the production mounts still allow).
 */
export function entryHomeHref(pathname: string | undefined | null): string {
  const path = pathname ?? ''
  const file = path.slice(path.lastIndexOf('/') + 1)
  return file.toLowerCase().endsWith('.html') ? file : './'
}
