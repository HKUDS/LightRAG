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
