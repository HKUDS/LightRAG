import { describe, expect, test } from 'bun:test'
import { assertDomAvailable, restoreDomGlobals, withoutDomGlobals } from './domGlobals'

describe('assertDomAvailable', () => {
  test('says nothing when the preload installed a DOM', () => {
    expect(() => assertDomAvailable()).not.toThrow()
  })

  test('names the cause and the fix when there is no DOM', () => {
    withoutDomGlobals(() => {
      // The whole point of the guard is that the message is actionable: the
      // failure it replaces (`document[isPrepared]`, thrown from inside
      // user-event) names neither the working directory nor the preload.
      expect(() => assertDomAvailable()).toThrow(/lightrag_webui/)
      expect(() => assertDomAvailable()).toThrow(/bunfig\.toml/)
    })
  })
})

describe('withoutDomGlobals', () => {
  test('removes both globals by default and puts them back', () => {
    withoutDomGlobals(() => {
      expect(typeof globalThis.window).toBe('undefined')
      expect(typeof globalThis.document).toBe('undefined')
    })

    expect(typeof globalThis.window).not.toBe('undefined')
    expect(typeof globalThis.document).not.toBe('undefined')
  })

  test('removes only the named globals', () => {
    withoutDomGlobals(() => {
      expect(typeof globalThis.document).toBe('undefined')
      // `window` survives, which is what lets a test exercise "resolved a
      // title from the injected config, but has nowhere to write it".
      expect(typeof globalThis.window).not.toBe('undefined')
    }, ['document'])

    expect(typeof globalThis.document).not.toBe('undefined')
  })

  test('restores the globals even when the body throws', () => {
    expect(() =>
      withoutDomGlobals(() => {
        throw new Error('boom')
      })
    ).toThrow('boom')

    expect(typeof globalThis.document).not.toBe('undefined')
  })

  test('returns whatever the body returned', () => {
    expect(withoutDomGlobals(() => 42)).toBe(42)
  })
})

describe('restoreDomGlobals', () => {
  test('reinstates a stubbed global', () => {
    const real = globalThis.window
    Object.defineProperty(globalThis, 'window', {
      value: { __LIGHTRAG_CONFIG__: { webuiTitle: 'stub' } },
      configurable: true
    })
    expect(globalThis.window).not.toBe(real)

    restoreDomGlobals()

    expect(globalThis.window).toBe(real)
  })
})
