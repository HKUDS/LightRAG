/**
 * Browser tab title (`document.title`).
 *
 * The entry HTML ships a static `<title>LightRAG</title>`, which a deployment
 * that set `WEBUI_TITLE` overrides. Two paths keep the tab correct, and both
 * are needed:
 *
 * 1. FIRST PAINT — the server's runtime-config snippet (see `runtimeConfig`)
 *    assigns `document.title` inline in `<head>`, before the bundle loads.
 *    Under `bun run dev` there is no such injection, so this path is absent.
 * 2. LIVE VALUE — the auth store's `webuiTitle`, refreshed from `/health` and
 *    the login response, so restarting the server with a new `WEBUI_TITLE`
 *    reaches an already-open tab on the next poll (same tier as the header
 *    title it matches).
 *
 * The store wins whenever it holds a title, because it is the fresher of the
 * two; the injected value backs it up while the store is still empty (a first
 * visit, before any authenticated request has returned a title) so the tab
 * never falls back from the deployment's name to the generic default.
 */
import { getRuntimeWebuiTitle } from '@/lib/runtimeConfig'

/** Shown when the deployment sets no `WEBUI_TITLE`. Matches the entry HTML. */
export const DEFAULT_DOCUMENT_TITLE = 'LightRAG'

/** The tab title for a given store value. Blank/whitespace counts as unset. */
export function resolveDocumentTitle(webuiTitle?: string | null): string {
  const fromStore = (webuiTitle ?? '').trim()
  if (fromStore) return fromStore

  const injected = (getRuntimeWebuiTitle() ?? '').trim()
  if (injected) return injected

  return DEFAULT_DOCUMENT_TITLE
}

/** Write {@link resolveDocumentTitle} to the document, if there is one. */
export function applyDocumentTitle(webuiTitle?: string | null): void {
  if (typeof document === 'undefined') return

  const next = resolveDocumentTitle(webuiTitle)
  if (document.title !== next) {
    document.title = next
  }
}
