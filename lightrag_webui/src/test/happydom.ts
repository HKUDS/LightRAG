/**
 * Register a DOM into the global scope for `bun test`.
 *
 * This runs as the FIRST `bunfig.toml` preload, before any other preload or
 * test file is evaluated. The ordering is load-bearing: React DOM and
 * Testing Library capture `document` / `window` when their modules are
 * first evaluated, so a DOM registered after them would leave those
 * libraries bound to nothing.
 *
 * Registration is global and process-wide — Bun evaluates every test file in
 * one process — so a DOM exists for the whole run, including the files that
 * predate it. Tests that need to assert DOM-LESS behaviour must therefore
 * remove the globals themselves for the duration of the assertion rather
 * than relying on their absence (see `stores/documentTitleSync.test.ts`).
 */
import { GlobalRegistrator } from '@happy-dom/global-registrator'

GlobalRegistrator.register({
  // Both are read by application code (entryHomeHref, healthCheck, the
  // settings store's persisted origin). A blank about:blank URL would make
  // `window.location.pathname` an empty string and mask path-prefix bugs.
  url: 'http://localhost:5173/webui/',
  width: 1280,
  height: 800
})
