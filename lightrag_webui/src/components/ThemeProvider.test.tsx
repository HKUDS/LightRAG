import { afterEach, describe, expect, test } from 'bun:test'
import { act, cleanup } from '@testing-library/react'

import ThemeProvider from '@/components/ThemeProvider'
import { renderWithProviders, testI18n } from '@/test/render'

/**
 * WCAG 3.1.1: the HTML shell ships a static lang="en", so without a runtime
 * sync every non-English locale is mislabeled for screen readers. The
 * provider must write documentElement.lang on mount and on every i18n
 * language change, mapping the underscore locale ids onto BCP-47.
 */

const renderProvider = () =>
  renderWithProviders(
    <ThemeProvider>
      <div>content</div>
    </ThemeProvider>
  )

afterEach(async () => {
  // Unmount FIRST (file-local afterEach runs before the preload's cleanup):
  // the lang listener must be gone before the shared instance is reset.
  cleanup()
  // testI18n and the DOM are process-wide singletons: leave English behind
  // so later files' asserted strings and lang attribute stay deterministic.
  await testI18n.changeLanguage('en')
  document.documentElement.lang = 'en'
})

describe('document language sync', () => {
  test('lang follows the active language, with BCP-47 tags', async () => {
    renderProvider()
    expect(document.documentElement.lang).toBe('en')

    // zh_TW is the one locale id that is not already a valid BCP-47 tag.
    await act(async () => {
      await testI18n.changeLanguage('zh_TW')
    })
    expect(document.documentElement.lang).toBe('zh-TW')

    await act(async () => {
      await testI18n.changeLanguage('fr')
    })
    expect(document.documentElement.lang).toBe('fr')
  })

  test('initial sync covers a non-English start, and unmount stops the sync', async () => {
    await testI18n.changeLanguage('de')
    renderProvider()
    expect(document.documentElement.lang).toBe('de')

    cleanup()
    await testI18n.changeLanguage('ja')
    // A leaked listener would have written 'ja' here.
    expect(document.documentElement.lang).toBe('de')
  })
})