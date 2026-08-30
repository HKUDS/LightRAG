import { afterEach, describe, expect, test } from 'bun:test'
import { resolveBrandTitle } from './brandTitle'

type GlobalWithWindow = typeof globalThis & {
  window?: { __LIGHTRAG_CONFIG__?: { webuiTitle?: string | null } }
}

const globalWithWindow = globalThis as GlobalWithWindow
const previousWindow = Object.getOwnPropertyDescriptor(globalThis, 'window')

const setRuntimeTitle = (webuiTitle: string | null | undefined): void => {
  Object.defineProperty(globalThis, 'window', {
    value: { __LIGHTRAG_CONFIG__: { webuiTitle } },
    configurable: true
  })
}

afterEach(() => {
  if (previousWindow) {
    Object.defineProperty(globalThis, 'window', previousWindow)
  } else {
    delete globalWithWindow.window
  }
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
