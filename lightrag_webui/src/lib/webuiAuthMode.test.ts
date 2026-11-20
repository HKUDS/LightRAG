import { expect, test } from 'bun:test'
import { webuiAuthMode } from './webuiAuthMode'

test.each([
  [false, false, 'guest'], [false, true, 'api-key-only'],
  [true, false, 'account'], [true, true, 'account'],
  [false, undefined, 'unknown'], [true, undefined, 'account']
] as const)('accounts=%s key=%s selects %s', (accounts, key, mode) => {
  expect(webuiAuthMode({ auth_configured: accounts, api_key_configured: key })).toBe(mode)
})
