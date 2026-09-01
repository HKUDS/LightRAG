import { readdirSync } from 'fs'

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
 * Building the separator instead of asking the platform for it removes the
 * difference at the source, with no `process.platform` branch that only one
 * operating system would ever execute. `readdirSync` and `readFileSync`
 * accept forward slashes on Windows, so nothing downstream needs to care.
 */
export const sourceFiles = (dir: string, accept: (path: string) => boolean): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = `${dir}/${entry.name}`
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
