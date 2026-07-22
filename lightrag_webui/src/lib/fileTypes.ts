import type { DropzoneProps } from 'react-dropzone'

/**
 * Response of GET /documents/supported_file_types.
 *
 * `supported_extensions` is the allowlist for bare (unhinted) filenames,
 * derived server-side from the parser registry + LIGHTRAG_PARSER routing.
 * `engines` maps every usable parser engine to the dotted suffixes it can
 * parse, enabling local pre-validation of `[engine]`-hinted filenames.
 */
export interface SupportedFileTypes {
  supported_extensions: string[]
  engines: Record<string, string[]>
}

/** Lifecycle of the supported-file-types fetch inside the upload dialog. */
export type FileTypesState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; data: SupportedFileTypes }
  | { status: 'fallback' }

export interface UploaderInputs {
  disabled: boolean
  acceptedExtensions?: string[]
  engineCapabilities?: Record<string, string[]>
}

/**
 * Filename parser hint, e.g. `img.[mineru].png`. Must stay byte-identical to
 * the backend's `_PARSER_HINT_RE` (lightrag/parser/routing.py) — any drift
 * makes the local pre-check disagree with the server's verdict.
 */
export const PARSER_HINT_RE = /\.\[([^\]]*)\](\.[^.]+)$/

/**
 * Dotted lowercase extension, aligned with Python's `Path.suffix` (the
 * backend's suffix source): `README`, `.bashrc` and `file.` all have none.
 */
export function extractFileExtension(filename: string): string {
  const i = filename.lastIndexOf('.')
  return i > 0 && i < filename.length - 1 ? filename.slice(i).toLowerCase() : ''
}

/** Flatten a react-dropzone accept map into sorted unique dotted extensions. */
export function flattenAcceptExtensions(accept: NonNullable<DropzoneProps['accept']>): string[] {
  const out = new Set<string>()
  for (const extensions of Object.values(accept)) {
    for (const ext of extensions) {
      out.add(ext.toLowerCase())
    }
  }
  return [...out].sort()
}

/**
 * Engine-name candidate from a hint's bracket content. Mirrors the backend's
 * `normalize_parser_engine`: cut at `(`, take the segment before the first
 * `-`, trim, lowercase. Validity is decided by the capability matrix keys,
 * never by a hardcoded engine set (third-party engines register dynamically).
 */
export function normalizeParserEngine(token: string): string {
  let text = token.trim()
  const paren = text.indexOf('(')
  if (paren !== -1) {
    text = text.slice(0, paren)
  }
  return text.split('-', 1)[0].trim().toLowerCase()
}

/**
 * Engine + suffix pre-check, the single verdict shared by the dropzone
 * validator and the onDrop re-filter.
 *
 * Bare filenames are checked against the allowlist. Hinted filenames are
 * checked against the engine capability matrix; with no matrix available
 * (old backend / fetch failed) they pass through for the server to judge.
 * Option/parameter syntax inside the hint is NOT validated here — a
 * malformed hint the matrix cannot see still gets a 400 from the backend's
 * full validation.
 */
export function isAcceptedFilename(
  filename: string,
  allowlist: Set<string>,
  engineMatrix: Record<string, string[]> | null
): boolean {
  const ext = extractFileExtension(filename)
  const m = PARSER_HINT_RE.exec(filename)
  if (!m) {
    return allowlist.has(ext)
  }
  if (engineMatrix === null) {
    return true
  }
  const inner = m[1].trim()
  if (inner === '') {
    return false
  }
  if (inner.startsWith('-')) {
    // Options-only hint carries no engine; routing follows the bare suffix.
    return allowlist.has(ext)
  }
  const engine = normalizeParserEngine(inner)
  if (Object.prototype.hasOwnProperty.call(engineMatrix, engine)) {
    return engineMatrix[engine].includes(ext)
  }
  return false
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

/**
 * Validate an untrusted response into SupportedFileTypes, or null when the
 * shape is invalid. An empty `supported_extensions` array is a legitimate
 * authoritative answer (no bare suffix is routable), not a failure.
 */
export function normalizeSupportedFileTypes(data: unknown): SupportedFileTypes | null {
  if (typeof data !== 'object' || data === null) {
    return null
  }
  const { supported_extensions, engines } = data as Record<string, unknown>
  if (!isStringArray(supported_extensions)) {
    return null
  }
  if (typeof engines !== 'object' || engines === null || Array.isArray(engines)) {
    return null
  }
  for (const suffixes of Object.values(engines)) {
    if (!isStringArray(suffixes)) {
      return null
    }
  }
  return { supported_extensions, engines: engines as Record<string, string[]> }
}

/** Human-readable list for the i18n `fileTypes` interpolation: ".md" -> "MD". */
export function formatFileTypesLabel(exts: string[]): string {
  return exts.map((ext) => ext.replace(/^\./, '').toUpperCase()).join(', ')
}

/**
 * Map the fetch state to FileUploader inputs. Uploading stays disabled until
 * the fetch settles so a hinted file can never start uploading before the
 * capability matrix is known; `fallback` enables the static allowlist and
 * defers hinted files to the server.
 */
export function deriveUploaderInputs(state: FileTypesState): UploaderInputs {
  switch (state.status) {
    case 'ready':
      return {
        disabled: false,
        acceptedExtensions: state.data.supported_extensions,
        engineCapabilities: state.data.engines
      }
    case 'fallback':
      return { disabled: false }
    default:
      return { disabled: true }
  }
}
