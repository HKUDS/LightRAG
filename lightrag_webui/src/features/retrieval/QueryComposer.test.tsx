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

describe('query composer multiline editing', () => {
  // Browsers run setTimeout(0) before the next animation frame; happy-dom runs
  // them the other way round. Without this stub the caret-restoring rAF in
  // QueryComposer fires before the handlers' setTimeout, so a test would only
  // pin the inputRef.current half of the fix. Scope the stub to this describe
  // block: bun runs every test file in one process, so a leaked global would
  // change the timing of unrelated tests.
  const realRaf = globalThis.requestAnimationFrame

  beforeEach(() => {
    globalThis.requestAnimationFrame = (cb: FrameRequestCallback) =>
      setTimeout(() => cb(performance.now()), 16) as unknown as number
  })

  afterEach(() => {
    globalThis.requestAnimationFrame = realRaf
  })

  // The stubbed frame is a 16ms timer. Let it run before asserting or typing,
  // otherwise it can move the caret back in the middle of the next keystroke.
  const settleFrame = () => new Promise((resolve) => setTimeout(resolve, 24))

  test('A: Shift+Enter keeps typing after the inserted newline', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'alpha beta')
    await user.keyboard('{Home}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    await settleFrame()

    expect(input().tagName).toBe('TEXTAREA')
    expect((input() as HTMLTextAreaElement).selectionStart).toBe(6)
    await user.keyboard('NEW')

    expect((input() as HTMLTextAreaElement).value).toBe('alpha\nNEW beta')
    expect(sent).toEqual([])
  })

  test('B: pasting a newline keeps typing at the end of the pasted text', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'alpha beta')
    await user.keyboard('{Home}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}')
    await user.paste('x\ny')
    await settleFrame()

    expect(input().tagName).toBe('TEXTAREA')
    expect((input() as HTMLTextAreaElement).selectionStart).toBe(8)
    await user.keyboard('Z')

    expect((input() as HTMLTextAreaElement).value).toBe('alphax\nyZ beta')
    expect(sent).toEqual([])
  })

  test('C: forward Delete at position 0 keeps typing at the start', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'alpha')
    await settleFrame()
    await user.keyboard('{Home}{Delete}')
    await settleFrame()
    await user.keyboard('X')

    // A 0 caret is a real position: `selectionStart || length` would send the
    // next keystroke to the end of the draft instead.
    expect((input() as HTMLInputElement).value).toBe('Xlpha')
    expect(sent).toEqual([])
  })

  test('F: Shift+Enter at position 0 keeps typing after the newline', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'alpha')
    await user.keyboard('{Home}')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    await settleFrame()

    expect(input().tagName).toBe('TEXTAREA')
    expect((input() as HTMLTextAreaElement).selectionStart).toBe(1)
    await user.keyboard('X')

    expect((input() as HTMLTextAreaElement).value).toBe('\nXalpha')
    expect(sent).toEqual([])
  })

  test('D: Backspace removing the only newline keeps the caret on the new input', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'alpha')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    await settleFrame()
    await user.type(input(), ' beta')
    await settleFrame()

    const textarea = input() as HTMLTextAreaElement
    expect(textarea.value).toBe('alpha\n beta')
    textarea.setSelectionRange(6, 6)
    await user.keyboard('{Backspace}')
    await settleFrame()

    expect(input().tagName).toBe('INPUT')
    expect((input() as HTMLInputElement).selectionStart).toBe(5)
    await user.keyboard('X')

    expect((input() as HTMLInputElement).value).toBe('alphaX beta')
    expect(sent).toEqual([])
  })

  // The cases above let the stubbed frame run before the next keystroke. A
  // fast typist does not wait for it: the keystrokes below arrive within the
  // same frame as the switch (user-event only yields a zero-delay timer between
  // keys), so the new element must already be focused with the caret in place.
  test('G: typing right after Shift+Enter in the middle lands after the newline', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'alpha beta')
    await settleFrame()
    await user.keyboard('{Home}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}')
    await user.keyboard('{Shift>}{Enter}{/Shift}NEW')

    expect((input() as HTMLTextAreaElement).value).toBe('alpha\nNEW beta')
    expect(sent).toEqual([])
  })

  test('H: typing right after Shift+Enter at the end lands after the newline', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'alpha')
    await settleFrame()
    await user.keyboard('{Shift>}{Enter}{/Shift}NEW')

    expect((input() as HTMLTextAreaElement).value).toBe('alpha\nNEW')
    expect(sent).toEqual([])
  })

  test('I: typing right after a multi-line paste lands after the pasted text', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'alpha beta')
    await settleFrame()
    await user.keyboard('{Home}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}')
    await user.paste('x\ny')
    await user.keyboard('Z')

    expect((input() as HTMLTextAreaElement).value).toBe('alphax\nyZ beta')
    expect(sent).toEqual([])
  })

  test('J: pasting over an identical selection collapses the caret to its end', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'a')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    await settleFrame()
    await user.type(input(), 'b')
    await settleFrame()

    // The pasted text equals the selection, so the draft does not change and
    // React does not re-render: the caret must still move past the paste, and
    // nothing may be left over to move it again on a later keystroke.
    const textarea = input() as HTMLTextAreaElement
    textarea.setSelectionRange(0, 3)
    await user.paste('a\nb')
    await settleFrame()
    await user.keyboard('Z')
    await settleFrame()
    await user.keyboard('Y')

    expect((input() as HTMLTextAreaElement).value).toBe('a\nbZY')
    expect(sent).toEqual([])
  })

  test('K: Shift+Enter and paste inside a textarea keep the caret without a switch', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'ab')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    await settleFrame()
    await user.type(input(), 'cd')
    await settleFrame()

    // Already a textarea: these edits re-render the same element, so the
    // caret restore must not depend on the element type changing.
    const textarea = input() as HTMLTextAreaElement
    textarea.setSelectionRange(1, 1)
    await user.keyboard('{Shift>}{Enter}{/Shift}X')
    await user.paste('1\n2')
    await user.keyboard('Y')

    expect(input() === textarea).toBe(true)
    expect(textarea.value).toBe('a\nX1\n2Yb\ncd')
    expect(sent).toEqual([])
  })
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
