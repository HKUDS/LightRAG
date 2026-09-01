import { readdirSync } from 'fs'

/**
 * Join a directory and an entry name with an explicit `/`.
 *
 * Exported so the separator rule can be pinned as a PURE function. A real
 * scan's separators are chosen by the host OS, so an assertion made against
 * one is green on Linux whether the rule holds or not — which is exactly how
 * a `path.join` walk survived here until Windows was considered.
 */
export const childPath = (dir: string, name: string): string => `${dir}/${name}`

/**
 * Walk `dir` and return every file `accept` keeps.
 *
 * Descends with an explicit `/` rather than `path.join`, which is the whole
 * point of this module existing: `join` emits the PLATFORM separator, so on
 * Windows every caller comparing a scanned path against a forward-slash
 * literal silently stops matching. That is not a cosmetic difference — the
 * three audits that used to carry their own copy of this walk all FAIL on
 * Windows, because an exclusion like `endsWith('/lib/loginIdentity.ts')`
 * never fires and the excluded file is then reported as a violation.
 *
 * What this guarantees is precisely the TAIL: every segment the walk appends
 * below `dir` is `/`-separated on every platform. The root is whatever the
 * caller passed — typically `path.join(import.meta.dir, …)`, so on Windows it
 * still holds backslashes — and it is deliberately left alone, because a
 * backslash is a legal filename character on Linux and rewriting one there
 * would point the scan at a directory that does not exist. Normalizing it
 * safely would need a `process.platform` branch that only one operating
 * system ever executes, which is the thing this module exists to avoid.
 *
 * That tail is all any caller needs: audits compare via `relativePath`, or
 * with a `/`-anchored suffix strictly below the root. `readdirSync` and
 * `readFileSync` accept forward slashes on Windows, so a mixed-separator
 * absolute path still reads.
 */
export const sourceFiles = (dir: string, accept: (path: string) => boolean): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = childPath(dir, entry.name)
    if (entry.isDirectory()) return sourceFiles(path, accept)
    return entry.isFile() && accept(path) ? [path] : []
  })

/**
 * `file` as a `/`-separated path relative to `dir`.
 *
 * `dir` itself is whatever the caller's `import.meta.dir` gave them and may
 * still hold native separators; only its LENGTH is used, so what comes back
 * is the tail `sourceFiles` built — separator-stable on every platform, which
 * is what makes it safe to compare against a literal or assert with `toEqual`.
 */
export const relativePath = (dir: string, file: string): string => file.slice(dir.length + 1)
