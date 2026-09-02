/// <reference types="bun" />
import { afterEach, describe, expect, test } from 'bun:test'
import { readdirSync, readFileSync } from 'fs'
import { join } from 'path'
import { cleanup, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import en from '@/locales/en.json'
import QuerySettings from './QuerySettings'

/**
 * The parameters edited in this panel are per-browser local state that the
 * query entry (/workspace) reads from the same storage, so the panel has to
 * say whose queries they affect. The notice must stay VISIBLE — the panel
 * already carries an `sr-only` card description, and folding the notice into
 * it would silently drop the point for everyone using a screen.
 */
describe('QuerySettings scope notice', () => {
  afterEach(() => {
    cleanup()
  })

  test('renders the session-scope notice as visible text', () => {
    renderWithProviders(<QuerySettings />)

    const notice = screen.getByText(en.retrievePanel.querySettings.parametersScopeNotice)
    expect(notice.className.includes('sr-only')).toBe(false)
    expect(notice.textContent).toContain('/workspace')
  })

  test('every interface language ships the notice', () => {
    const localesDir = join(import.meta.dir, '..', '..', 'locales')
    const files = readdirSync(localesDir).filter((name) => name.endsWith('.json'))
    expect(files.length).toBeGreaterThan(1)

    const missing = files.filter((name) => {
      const bundle = JSON.parse(readFileSync(join(localesDir, name), 'utf-8'))
      const notice = bundle?.retrievePanel?.querySettings?.parametersScopeNotice
      return typeof notice !== 'string' || notice.trim().length === 0
    })
    expect(missing).toEqual([])
  })
})
