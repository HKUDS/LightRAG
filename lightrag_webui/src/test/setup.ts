/**
 * Testing Library wiring for `bun test`.
 *
 * Loaded as the SECOND `bunfig.toml` preload — the DOM is already installed
 * by `happydom.ts` at this point, which is what lets React DOM be imported
 * here at all.
 */
import { afterEach, expect } from 'bun:test'
import * as matchers from '@testing-library/jest-dom/matchers'
import { cleanup } from '@testing-library/react'

// jest-dom's assertions (toBeInTheDocument, toHaveAccessibleName, ...) are
// plain matcher objects; Bun's expect takes them the same way Jest's does.
expect.extend(matchers as Parameters<typeof expect.extend>[0])

// Unmount whatever a test rendered. Without this every render stays in
// `document.body` for the rest of the process, and the NEXT test's
// `getByRole` finds two matching nodes and throws — a failure that points
// at the innocent test rather than the one that leaked.
afterEach(() => {
  cleanup()
})
