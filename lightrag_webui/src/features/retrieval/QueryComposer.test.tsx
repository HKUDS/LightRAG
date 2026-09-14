/// <reference types="bun" />
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { cleanup, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import QueryComposer from './QueryComposer'

/**
 * The composer owns the validation notice: `onSend` returns a message to show
 * or null to accept. What matters is what the user is left with afterwards —
 * whether the message is announced, whether it survives editing, and whether
 * the draft is still there to correct — so it is exercised, not read.
 */

const REJECTED = 'Only supports the following query modes: naive, local'
const NOTICE_ID = 'query-input-error'

/** Anything starting with `/` is rejected, like a bad mode prefix. */
const onSend = (input: string): string | null => {
  sent.push(input)
  return input.startsWith('/') ? REJECTED : null
}

let sent: string[]

const renderComposer = () =>
  renderWithProviders(
    <QueryComposer
      isLoading={false}
      stopDisabled={false}
      onSend={onSend}
      onStop={() => {}}
      onClear={() => {}}
    />
  )

const input = (): HTMLElement => within(screen.getByRole('search')).getByRole('textbox')

const notice = (): HTMLElement => screen.getByRole('alert')

beforeEach(() => {
  sent = []
})

afterEach(() => {
  cleanup()
})

describe('query composer validation notice', () => {
  test('a rejected send is announced as an alert, above the composer', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')

    // `role="alert"` is what makes a screen reader speak this without the user
    // going looking for it.
    expect(notice().textContent).toContain(REJECTED)

    // Below the composer it would sit off-screen on a short viewport, under
    // the mobile keyboard, or behind the send button.
    const classes = (notice().getAttribute('class') ?? '').split(/\s+/)
    expect(classes).toContain('absolute')
    expect(classes).toContain('bottom-full')
    expect(classes).not.toContain('top-full')
  })

  test('a rejected send keeps the draft so it can be corrected', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')

    expect((input() as HTMLInputElement).value).toBe('/nope hello')
  })

  test('the notice survives editing', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')
    await user.type(input(), ' more')

    // It names what to change; wiping it on the first keystroke would leave
    // the user correcting blind.
    expect(screen.queryAllByRole('alert')).toHaveLength(1)
    expect(notice().textContent).toContain(REJECTED)
  })

  test('the input is wired to the notice for assistive tech', async () => {
    const user = userEvent.setup()
    renderComposer()

    expect(input().getAttribute('aria-invalid')).toBe(null)
    expect(input().getAttribute('aria-describedby')).toBe(null)

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')

    expect(input().getAttribute('aria-invalid')).toBe('true')
    expect(input().getAttribute('aria-describedby')).toBe(NOTICE_ID)
    expect(notice().getAttribute('id')).toBe(NOTICE_ID)
  })

  test('the multi-line variant carries the same wiring', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')
    expect(input().tagName).toBe('INPUT')

    // Shift+Enter switches the single-line Input for a Textarea. The swap is
    // exactly where the wiring gets lost: the second element is a different
    // node and inherits none of the first one's attributes.
    await user.keyboard('{Shift>}{Enter}{/Shift}')

    expect(input().tagName).toBe('TEXTAREA')
    expect(input().getAttribute('aria-invalid')).toBe('true')
    expect(input().getAttribute('aria-describedby')).toBe(NOTICE_ID)
  })

  test('an accepted send clears the notice and the draft', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')
    expect(screen.queryAllByRole('alert')).toHaveLength(1)

    await user.clear(input())
    await user.type(input(), 'a real question')
    await user.keyboard('{Enter}')

    expect(sent).toEqual(['/nope hello', 'a real question'])
    expect(screen.queryAllByRole('alert')).toHaveLength(0)
    expect((input() as HTMLInputElement).value).toBe('')
    expect(input().getAttribute('aria-invalid')).toBe(null)
  })

  test('the notice can be dismissed without editing', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')

    await user.click(within(notice()).getByRole('button', { name: 'Dismiss error' }))

    expect(screen.queryAllByRole('alert')).toHaveLength(0)
    // Dismissing is not sending: the draft is still the user's to fix.
    expect((input() as HTMLInputElement).value).toBe('/nope hello')
    expect(sent).toEqual(['/nope hello'])
  })
})
