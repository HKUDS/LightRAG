/// <reference types="bun" />
import { afterEach, describe, expect, test } from 'bun:test'
import { readdirSync, readFileSync } from 'fs'
import { join } from 'path'
import { act, cleanup, fireEvent, screen, waitFor } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import en from '@/locales/en.json'
import { useQuerySettingsStore, defaultQuerySettings } from '@/stores/querySettings'
import QuerySettings from './QuerySettings'

/**
 * The parameters edited in this panel are per-browser local state that the
 * query entry (/workspace) reads from the same storage, so the panel has to
 * say whose queries they affect. The one-line notice must stay VISIBLE — the
 * panel already carries an `sr-only` card description, and folding the notice
 * into it would silently drop the point for everyone using a screen — and the
 * full explanation must stay reachable from the help control beside it,
 * including by TAP: a device with no hover has nothing else to open it with.
 */
const strings = en.retrievePanel.querySettings

describe('QuerySettings scope notice', () => {
  afterEach(() => {
    cleanup()
  })

  test('renders the scope notice as visible text beside a touch-sized help control', () => {
    renderWithProviders(<QuerySettings />)

    const notice = screen.getByText(strings.parametersScopeNotice)
    expect(notice.className.includes('sr-only')).toBe(false)
    // Kept to a single line whatever the translation's length: the panel is
    // 280px wide and the controls below it must not be pushed down.
    expect(notice.className.includes('truncate')).toBe(true)

    // The icon is 12px; on a touch-only device the button IS the tap path, so
    // the padding that grows the hit box to 24x24 is load-bearing. Asserted as
    // a class because happy-dom computes no layout.
    const help = screen.getByRole('button', { name: strings.parametersScopeHelpLabel })
    expect(help.className.split(/\s+/)).toContain('p-1.5')
  })

  /**
   * A tap, as the DOM delivers it: pointerdown then click. Both halves matter —
   * Radix dismisses an open tooltip from the pointerdown, so a toggle that
   * reads the state at CLICK time can never close it again. `userEvent.click`
   * is deliberately not used: it also moves the pointer onto the trigger, and
   * the hover-open timer that starts there races the assertions.
   */
  const tap = (element: Element): void => {
    fireEvent.pointerDown(element)
    fireEvent.click(element)
  }

  test('tapping the help control reveals the explanation, and tapping again hides it', async () => {
    renderWithProviders(<QuerySettings />)

    // Closed to begin with: Radix only mounts the content while open.
    expect(screen.queryAllByText(strings.parametersScopeTooltip)).toHaveLength(0)

    const help = screen.getByRole('button', { name: strings.parametersScopeHelpLabel })
    tap(help)
    await act(async () => {})
    await waitFor(() => {
      expect(screen.getAllByText(strings.parametersScopeTooltip).length).toBeGreaterThan(0)
    })

    tap(help)
    await act(async () => {})

    await waitFor(() => {
      expect(screen.queryAllByText(strings.parametersScopeTooltip)).toHaveLength(0)
    })
  })

  test('the explanation is dismissed by interacting elsewhere', async () => {
    renderWithProviders(<QuerySettings />)

    tap(screen.getByRole('button', { name: strings.parametersScopeHelpLabel }))
    await act(async () => {})
    await waitFor(() => {
      expect(screen.getAllByText(strings.parametersScopeTooltip).length).toBeGreaterThan(0)
    })

    tap(document.body)
    await act(async () => {})

    await waitFor(() => {
      expect(screen.queryAllByText(strings.parametersScopeTooltip)).toHaveLength(0)
    })
  })

  test('every interface language ships the notice, its explanation and the help label', () => {
    const localesDir = join(import.meta.dir, '..', '..', 'locales')
    const files = readdirSync(localesDir).filter((name) => name.endsWith('.json'))
    expect(files.length).toBeGreaterThan(1)

    const keys = [
      'parametersScopeNotice',
      'parametersScopeTooltip',
      'parametersScopeHelpLabel'
    ] as const

    const missing = files.flatMap((name) => {
      const bundle = JSON.parse(readFileSync(join(localesDir, name), 'utf-8'))
      const panel = bundle?.retrievePanel?.querySettings
      return keys
        .filter((key) => typeof panel?.[key] !== 'string' || panel[key].trim().length === 0)
        .map((key) => `${name}:${key}`)
    })
    expect(missing).toEqual([])
  })
})

/**
 * The mode trigger's only visible keyboard focus indicator is the Select
 * primitive's focus ring. Zeroing it out of the trigger's own className
 * leaves a keyboard user with no way to see which control is focused.
 */
describe('QuerySettings mode selector focus', () => {
  afterEach(() => {
    cleanup()
    useQuerySettingsStore.setState({ querySettings: { ...defaultQuerySettings } })
  })

  test('the mode trigger keeps the primitive focus ring, with no zeroing classes', () => {
    // A narrower-coverage mode: the trigger's className is then a cn() merge
    // of the base classes and the red warning classes — the path where a
    // focus-zeroing class could hide.
    useQuerySettingsStore.setState({
      querySettings: { ...defaultQuerySettings, mode: 'naive' }
    })
    renderWithProviders(<QuerySettings />)

    const trigger = screen.getByRole('combobox', { name: strings.queryMode })
    const classes = (trigger.getAttribute('class') ?? '').split(/\s+/)

    // The warning branch really was exercised.
    expect(classes).toContain('border-red-400')

    // The primitive's ring survives the merge...
    expect(classes).toContain('focus:ring-2')
    expect(classes).toContain('focus:ring-ring')
    expect(classes).toContain('focus:ring-offset-2')
    // ...and nothing zeroes it out.
    expect(
      classes.filter((token) =>
        ['focus:ring-0', 'focus:ring-offset-0', 'focus:outline-0'].includes(token)
      )
    ).toEqual([])

    // The rest of the trigger's styling is untouched by the fix. In the
    // warning branch the red hover wins the twMerge over hover:bg-primary/5.
    expect(classes).toContain('hover:bg-red-100')
    expect(classes).toContain('[&>span]:line-clamp-1')
  })
})
