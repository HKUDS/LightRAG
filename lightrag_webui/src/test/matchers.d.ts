/**
 * Register jest-dom's matchers with `bun:test`'s type surface.
 *
 * `expect.extend` (in `setup.ts`) adds them at RUNTIME; TypeScript still
 * types `expect(...)` from Bun's own `Matchers` interface, which knows
 * nothing about `toBeInTheDocument` & co. jest-dom ships types for Jest and
 * Vitest only, so the declaration merge has to be done here.
 */
import type { TestingLibraryMatchers } from '@testing-library/jest-dom/matchers'

declare module 'bun:test' {
  interface Matchers<T = unknown>
    extends TestingLibraryMatchers<(actual: unknown) => void, T> {}
  interface AsymmetricMatchers extends TestingLibraryMatchers<unknown, unknown> {}
}
