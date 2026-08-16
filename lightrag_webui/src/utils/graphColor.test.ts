/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'

import { resolveNodeColor } from '@/utils/graphColor'

// entity_type comes from extracted document text (or a caller of
// ainsert_custom_kg), so it is untrusted. Two hazards must never crash the
// Legend, which does `key.toLowerCase()` on every type-color map key:
//   1a. A reserved *string* name (`__proto__`, `constructor`) used to index the
//       plain-object synonym table and returned an INHERITED non-string value
//       that was then stored as a Map key.
//   1b. A genuinely non-string value (object/array/number/boolean) hit
//       `nodeType.toLowerCase()` before any table lookup.
// resolveNodeColor must return a string color and a Map whose keys are all
// strings for every input.

// The exact expression the Legend renders for each map key (Legend.tsx).
const legendExpr = (key: unknown) =>
  String(key).toLowerCase().replace(/\s+/g, '')

const assertSafe = (nodeType: unknown) => {
  const { color, map } = resolveNodeColor(nodeType, undefined)
  expect(typeof color).toBe('string')
  expect(color.length).toBeGreaterThan(0)
  for (const key of map.keys()) {
    // Regression: reserved names used to land here as object/function keys.
    expect(typeof key).toBe('string')
    // Regression: the Legend must not throw while rendering the key.
    expect(() => legendExpr(key)).not.toThrow()
  }
}

describe('resolveNodeColor — reserved string entity types (1a)', () => {
  // The full Object.prototype member-name matrix.
  const reservedNames = [
    '__proto__',
    'constructor',
    'prototype',
    'toString',
    'valueOf',
    'hasOwnProperty',
    'isPrototypeOf'
  ]

  for (const name of reservedNames) {
    test(`does not produce a non-string map key for "${name}"`, () => {
      assertSafe(name)
    })
  }

  test('legitimate types still resolve to their predefined color', () => {
    expect(resolveNodeColor('person', undefined).color).toBe('#4169E1')
    expect(resolveNodeColor('organization', undefined).color).toBe('#00cc00')
    // Synonyms still map through.
    expect(resolveNodeColor('company', undefined).color).toBe('#00cc00')
  })

  test('an unknown but ordinary type gets an extended color, not a crash', () => {
    const { color, map } = resolveNodeColor('quokka', undefined)
    expect(typeof color).toBe('string')
    expect(map.has('quokka')).toBe(true)
  })
})

describe('resolveNodeColor — non-string entity types (1b)', () => {
  const nonStrings: Array<[string, unknown]> = [
    ['number', 42],
    ['object', {}],
    ['array', []],
    ['null', null],
    ['boolean', true],
    ['undefined', undefined]
  ]

  for (const [label, value] of nonStrings) {
    test(`does not throw and returns a string color for a ${label} type`, () => {
      expect(() => resolveNodeColor(value, undefined)).not.toThrow()
      assertSafe(value)
    })
  }
})
