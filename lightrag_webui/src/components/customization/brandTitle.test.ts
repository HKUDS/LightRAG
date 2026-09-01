import { afterEach, describe, expect, test } from 'bun:test'
import { resolveBrandTitle } from './brandTitle'
import { restoreDomGlobals } from '@/test/domGlobals'

const setRuntimeTitle = (webuiTitle: string | null | undefined): void => {
  Object.defineProperty(globalThis, 'window', {
    value: { __LIGHTRAG_CONFIG__: { webuiTitle } },
    configurable: true
  })
}

afterEach(() => {
  // The preloaded DOM is the state to return to; it is installed for the
  // whole process, so leaving this stub in place would follow the run into
  // every later test file.
  restoreDomGlobals()
})

describe('resolveBrandTitle', () => {
  test('prefers the successful customization snapshot', () => {
    setRuntimeTitle('Injected title')
    expect(resolveBrandTitle('Bundle title', 'Auth title')).toBe('Bundle title')
  })

  test('preserves the fresh auth-status title when customization fails', () => {
    setRuntimeTitle('Injected title')
    expect(resolveBrandTitle(null, 'Acme Safety')).toBe('Acme Safety')
  })

  test('falls back to the injected deployment title when both API sources fail', () => {
    setRuntimeTitle('Injected title')
    expect(resolveBrandTitle(null, null)).toBe('Injected title')
  })

  test('ignores blank sources and ultimately uses the product default', () => {
    setRuntimeTitle('   ')
    expect(resolveBrandTitle(' ', '\t')).toBe('LightRAG')
  })
})
