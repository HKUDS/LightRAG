/**
 * Browser tab title (`document.title`).
 *
 * The entry HTML ships a static `<title>LightRAG</title>`, which a deployment
 * that set `WEBUI_TITLE` overrides. Two sources carry that value, and they
 * differ in AGE, which is what the two entry points below are about:
 *
 * - INJECTED — the server's runtime-config snippet (see `runtimeConfig`)
 *   assigns `document.title` inline in `<head>`, before the bundle loads, so
 *   the tab is right from the first paint. It is a snapshot of the moment the
 *   page was served and never changes afterwards. Under `bun run dev` there is
 *   no injection at all.
 * - REPORTED — the auth store's `webuiTitle`, refreshed from `/health` and the
 *   login response, so restarting the server with a different `WEBUI_TITLE`
 *   reaches an already-open tab on the next poll (same tier as the header
 *   title it matches).
 *
 * A reported value is always the fresher of the two and REPLACES the injected
 * one — including a reported `null`, which says the deployment now has no
 * title and must revert the tab to the default. Falling back to the injected
 * value there would pin the tab to a name the server no longer uses until the
 * next reload, while the header beside it already showed the default.
 *
 * The injected value is therefore only consulted BEFORE the server has
 * reported anything in this page's lifetime — `applyInitialDocumentTitle`,
 * called once at store construction. Every later change to the store's
 * `webuiTitle` comes from `login()` or `setCustomTitle()`, i.e. straight from
 * a server response, and goes through `applyServerDocumentTitle`.
 *
 * The injected field is THREE-STATE, and the difference decides whether the
 * localStorage cache may be used at all:
 *
 * - a string — this deployment's title;
 * - `null` — the server published the field and the deployment has NO title.
 *   The backend always writes the key (`WEBUI_TITLE` or `None`), so this is a
 *   statement, not an absence, and it overrides the cache;
 * - `undefined` — the field was never injected: the dev server publishes only
 *   the prefixes, and a page may be served by a build older than the field.
 *   Only here does the cache get a say.
 */
import { getRuntimeWebuiTitle } from '@/lib/runtimeConfig'

/** Shown when the deployment sets no `WEBUI_TITLE`. Matches the entry HTML. */
export const DEFAULT_DOCUMENT_TITLE = 'LightRAG'

const clean = (value: string | null | undefined): string => (value ?? '').trim()

const setTitle = (title: string): void => {
  if (typeof document === 'undefined') return
  if (document.title !== title) {
    document.title = title
  }
}

/**
 * The title before any server response has landed in this page's lifetime.
 *
 * `cachedTitle` is the store's start value, restored from localStorage — a
 * title this browser saw on some EARLIER visit, so it may name a title the
 * deployment has since changed or cleared.
 *
 * Whenever the page carries an injected field it decides alone, because it
 * came from the response that served this very page and therefore cannot be
 * out of date — including when it is `null`, which says the deployment has no
 * title and must leave the static `LightRAG` in the tab. Falling through to
 * the cache there would rename the tab to a title the deployment dropped, and
 * it would stay wrong until a `/health` or login response landed — or forever,
 * if none does (an unreachable backend, or a user sitting on the login page).
 */
export function resolveInitialDocumentTitle(cachedTitle?: string | null): string {
  const injected = getRuntimeWebuiTitle()
  if (injected !== undefined) {
    return clean(injected) || DEFAULT_DOCUMENT_TITLE
  }

  // Nothing was injected (dev server, or a build older than the field): the
  // cache is the only evidence this page has.
  return clean(cachedTitle) || DEFAULT_DOCUMENT_TITLE
}

/**
 * The title for a value the server just reported. `null`/blank means the
 * deployment has no `WEBUI_TITLE`, and reverts the tab to the default — the
 * injected value is NOT consulted, see the module comment.
 */
export function resolveServerDocumentTitle(reportedTitle?: string | null): string {
  return clean(reportedTitle) || DEFAULT_DOCUMENT_TITLE
}

/** Apply {@link resolveInitialDocumentTitle}, if there is a document. */
export function applyInitialDocumentTitle(cachedTitle?: string | null): void {
  setTitle(resolveInitialDocumentTitle(cachedTitle))
}

/** Apply {@link resolveServerDocumentTitle}, if there is a document. */
export function applyServerDocumentTitle(reportedTitle?: string | null): void {
  setTitle(resolveServerDocumentTitle(reportedTitle))
}
