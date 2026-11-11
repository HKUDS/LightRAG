/**
 * Side-effect module that executes the settings-storage split migration at
 * module evaluation time.
 *
 * MUST be the FIRST static import of every entry file (`main.tsx`,
 * `workspace-main.tsx`): ESM evaluates same-level imports depth-first in
 * source order, so being first guarantees the migration completes before any
 * store module is evaluated (and therefore before any persist store reads or
 * writes localStorage). Its transitive import graph must stay free of
 * stores, `App`, the API client and the navigation service — guaranteed by
 * the pure-module contract of `legacySettingsChain.ts`.
 */

import { runSettingsStorageSplitMigration } from './splitSettingsStorage'

runSettingsStorageSplitMigration()
