/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { supportedFileTypes } from './constants'
import {
  deriveUploaderInputs,
  extractFileExtension,
  flattenAcceptExtensions,
  formatFileTypesLabel,
  isAcceptedFilename,
  normalizeParserEngine,
  normalizeSupportedFileTypes,
  type FileTypesState
} from './fileTypes'

const allowlist = new Set(['.md', '.txt', '.pdf', '.jpg'])
const matrix = {
  mineru: ['.png', '.jpg', '.pdf'],
  legacy: ['.md', '.txt', '.pdf'],
  native: ['.docx', '.md', '.textpack']
}

describe('extractFileExtension', () => {
  test('aligns with Python Path.suffix: no-dot, dotfile and trailing-dot names have none', () => {
    expect(extractFileExtension('README')).toBe('')
    expect(extractFileExtension('.bashrc')).toBe('')
    expect(extractFileExtension('file.')).toBe('')
  })

  test('returns the dotted lowercase last suffix', () => {
    expect(extractFileExtension('file.txt')).toBe('.txt')
    expect(extractFileExtension('file.PDF')).toBe('.pdf')
    expect(extractFileExtension('archive.tar.gz')).toBe('.gz')
    expect(extractFileExtension('img.[mineru].png')).toBe('.png')
  })
})

describe('normalizeParserEngine', () => {
  test('mirrors the backend: cut at "(", take segment before first "-", lowercase', () => {
    expect(normalizeParserEngine('mineru')).toBe('mineru')
    expect(normalizeParserEngine('MinerU')).toBe('mineru')
    expect(normalizeParserEngine('mineru-iteP(drop_rf)')).toBe('mineru')
    expect(normalizeParserEngine('mineru(page_range=1-3)')).toBe('mineru')
    expect(normalizeParserEngine('legacy-R(chunk_ts=800)')).toBe('legacy')
  })
})

describe('isAcceptedFilename — bare filenames', () => {
  test('accepts by allowlist extension, never by MIME', () => {
    expect(isAcceptedFilename('a.md', allowlist, matrix)).toBe(true)
    expect(isAcceptedFilename('a.rst', allowlist, matrix)).toBe(false)
  })

  test('uppercase suffix is normalized', () => {
    expect(isAcceptedFilename('file.PDF', allowlist, matrix)).toBe(true)
  })

  test('extensionless names are rejected', () => {
    expect(isAcceptedFilename('README', allowlist, matrix)).toBe(false)
  })

  test('empty allowlist rejects every bare filename', () => {
    expect(isAcceptedFilename('a.md', new Set<string>(), matrix)).toBe(false)
  })
})

describe('isAcceptedFilename — hinted filenames', () => {
  test('engine present in matrix: verdict follows its suffix capabilities', () => {
    expect(isAcceptedFilename('img.[mineru].png', allowlist, matrix)).toBe(true)
    expect(isAcceptedFilename('img.[mineru].txt', allowlist, matrix)).toBe(false)
  })

  test('engine missing from matrix (endpoint not configured) is rejected', () => {
    expect(isAcceptedFilename('img.[mineru].png', allowlist, { legacy: ['.md'] })).toBe(false)
  })

  test('matrix keys are authoritative — dynamic third-party engines pass', () => {
    expect(isAcceptedFilename('doc.[fooengine].foo', allowlist, { fooengine: ['.foo'] })).toBe(true)
  })

  test('case and parameter blocks are normalized before the matrix lookup', () => {
    expect(isAcceptedFilename('img.[MinerU].png', allowlist, matrix)).toBe(true)
    expect(isAcceptedFilename('img.[mineru-iteP(drop_rf)].png', allowlist, matrix)).toBe(true)
    expect(isAcceptedFilename('r.[legacy-R(chunk_ts=800)].pdf', allowlist, matrix)).toBe(true)
  })

  test('unknown engine and empty hint are rejected (backend would 400)', () => {
    expect(isAcceptedFilename('img.[nonexistent].png', allowlist, matrix)).toBe(false)
    expect(isAcceptedFilename('img.[].png', allowlist, matrix)).toBe(false)
  })

  test('options-only hint routes like a bare filename', () => {
    expect(isAcceptedFilename('doc.[-P].pdf', allowlist, matrix)).toBe(true)
    expect(isAcceptedFilename('doc.[-P].rst', allowlist, matrix)).toBe(false)
  })

  test('unclosed bracket is not a hint — the name validates as bare', () => {
    expect(isAcceptedFilename('img.[mineru.png', allowlist, matrix)).toBe(false)
    expect(isAcceptedFilename('img.[mineru.jpg', allowlist, matrix)).toBe(true)
  })

  test('hint still validates against an empty allowlist via the matrix', () => {
    expect(isAcceptedFilename('img.[mineru].png', new Set<string>(), matrix)).toBe(true)
  })
})

describe('isAcceptedFilename — degraded mode (no matrix)', () => {
  test('hinted filenames pass through for the server to judge', () => {
    expect(isAcceptedFilename('img.[mineru].png', allowlist, null)).toBe(true)
    expect(isAcceptedFilename('img.[nonexistent].png', allowlist, null)).toBe(true)
  })

  test('bare filenames still follow the allowlist', () => {
    expect(isAcceptedFilename('a.rst', allowlist, null)).toBe(false)
    expect(isAcceptedFilename('a.md', allowlist, null)).toBe(true)
  })
})

describe('normalizeSupportedFileTypes', () => {
  test('valid payload passes through, including an empty allowlist', () => {
    const valid = { supported_extensions: ['.md'], engines: { legacy: ['.md'] } }
    expect(normalizeSupportedFileTypes(valid)).toEqual(valid)
    const empty = { supported_extensions: [], engines: {} }
    expect(normalizeSupportedFileTypes(empty)).toEqual(empty)
  })

  test('invalid shapes collapse to null', () => {
    expect(normalizeSupportedFileTypes(undefined)).toBeNull()
    expect(normalizeSupportedFileTypes('x')).toBeNull()
    expect(normalizeSupportedFileTypes({})).toBeNull()
    expect(normalizeSupportedFileTypes({ supported_extensions: [null], engines: {} })).toBeNull()
    expect(normalizeSupportedFileTypes({ supported_extensions: ['.md'] })).toBeNull()
    expect(
      normalizeSupportedFileTypes({ supported_extensions: ['.md'], engines: { legacy: [1] } })
    ).toBeNull()
    expect(
      normalizeSupportedFileTypes({ supported_extensions: ['.md'], engines: ['.md'] })
    ).toBeNull()
  })
})

describe('flattenAcceptExtensions', () => {
  test('flattens the static fallback map into its full extension list', () => {
    const flat = flattenAcceptExtensions(supportedFileTypes)
    const expected = [...new Set(Object.values(supportedFileTypes).flat())].sort()
    expect(flat).toEqual(expected)
    expect(flat).toContain('.md')
    expect(flat).toContain('.pdf')
  })
})

describe('formatFileTypesLabel', () => {
  test('renders dotted extensions as an uppercase comma list', () => {
    expect(formatFileTypesLabel(['.jpg', '.md'])).toBe('JPG, MD')
    expect(formatFileTypesLabel([])).toBe('')
  })
})

describe('deriveUploaderInputs', () => {
  test('idle and loading keep the uploader disabled', () => {
    expect(deriveUploaderInputs({ status: 'idle' })).toEqual({ disabled: true })
    expect(deriveUploaderInputs({ status: 'loading' })).toEqual({ disabled: true })
  })

  test('ready enables the uploader with backend data', () => {
    const state: FileTypesState = {
      status: 'ready',
      data: { supported_extensions: ['.md'], engines: { legacy: ['.md'] } }
    }
    expect(deriveUploaderInputs(state)).toEqual({
      disabled: false,
      acceptedExtensions: ['.md'],
      engineCapabilities: { legacy: ['.md'] }
    })
  })

  test('fallback enables the uploader without backend data (static allowlist, hints deferred)', () => {
    expect(deriveUploaderInputs({ status: 'fallback' })).toEqual({ disabled: false })
  })
})
