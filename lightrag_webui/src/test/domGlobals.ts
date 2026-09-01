/**
 * Scoped removal of the DOM globals.
 *
 * Several tests exercise code paths that must survive having no `window` or
 * `document` — the workspace entry's SSR-safe guards, the injected-title
 * resolvers. Before the DOM preload existed those tests could simply
 * `delete globalThis.window`, because nothing put it back and nothing else
 * needed it.
 *
 * That is no longer true: `src/test/happydom.ts` installs one DOM for the
 * whole process (Bun evaluates every test file in a single process), so a
 * bare `delete` strips the DOM from every file that runs LATER in the same
 * run — and the failure surfaces in an unrelated component test rather than
 * in the file that caused it. These helpers capture the real descriptors at
 * import time and put them back.
 */
const REAL_WINDOW = Object.getOwnPropertyDescriptor(globalThis, 'window')
const REAL_DOCUMENT = Object.getOwnPropertyDescriptor(globalThis, 'document')

const restore = (key: 'window' | 'document', descriptor?: PropertyDescriptor): void => {
  if (descriptor) {
    Object.defineProperty(globalThis, key, descriptor)
  } else {
    // No DOM was preloaded (e.g. a file run through plain `bun` rather than
    // `bun test`): absent is the correct state to return to.
    delete (globalThis as Record<string, unknown>)[key]
  }
}

/** Put the preloaded `window` and `document` back, whatever a test did to them. */
export const restoreDomGlobals = (): void => {
  restore('window', REAL_WINDOW)
  restore('document', REAL_DOCUMENT)
}

type DomGlobalKey = 'window' | 'document'

/**
 * Remove the named DOM globals for the duration of `body`, then restore.
 *
 * `keys` defaults to both. Narrow it when the guard under test is about ONE
 * of them — `applyServerDocumentTitle` reads an injected `window` config and
 * writes `document.title`, so "no document" is a different path from "no
 * DOM at all", and removing both would silently stop exercising it.
 */
export const withoutDomGlobals = <T>(
  body: () => T,
  keys: readonly DomGlobalKey[] = ['window', 'document']
): T => {
  for (const key of keys) delete (globalThis as Record<string, unknown>)[key]
  try {
    return body()
  } finally {
    restoreDomGlobals()
  }
}

/**
 * Fail fast, and legibly, when `bun test` ran without the DOM preload.
 *
 * Bun resolves `bunfig.toml` — and the preload paths inside it — relative to
 * the CURRENT WORKING DIRECTORY. Running `bun test lightrag_webui/src/...`
 * from the repository root therefore loads no preload at all, silently: pure
 * logic tests still pass, and only the component tests fail, deep inside
 * Testing Library, with
 *
 *     TypeError: undefined is not an object (evaluating 'document[isPrepared]')
 *
 * which names neither the cause nor the fix. `bun test --config <path>` does
 * not help either, because the preload paths inside the file are still
 * resolved against the CWD.
 *
 * Called at import time by `src/test/render.tsx`, so any test that renders a
 * component reports this instead — before `userEvent.setup()` or `render()`
 * gets the chance to fail obscurely.
 */
export const assertDomAvailable = (): void => {
  if (typeof document !== 'undefined') return

  throw new Error(
    'No DOM: bun test started without the bunfig.toml preload. Run it from ' +
      'the lightrag_webui/ directory (`cd lightrag_webui && bun test ...`) — ' +
      'bun resolves bunfig.toml and its preload paths relative to the ' +
      'working directory, so running from the repository root loads no DOM.'
  )
}
