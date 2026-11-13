import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import Textarea from '@/components/ui/Textarea'
import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { CircleAlertIcon, EraserIcon, SendIcon, SquareIcon, XIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

/**
 * Shared input layer: single/multi-line switching, draft handling, clear,
 * send/stop with the stop cooldown. Contains no QuerySettings and no admin
 * navigation. Request preparation lives in `onSend`, provided by the page
 * composition layer: it returns an error string to show above the composer,
 * or null to accept (which clears the draft).
 */
export interface QueryComposerProps {
  isLoading: boolean
  stopDisabled: boolean
  /** Returns an error message to display, or null when the input was accepted. */
  onSend: (input: string) => string | null
  onStop: () => void
  onClear: () => void
}

export default function QueryComposer({
  isLoading,
  stopDisabled,
  onSend,
  onStop,
  onClear
}: QueryComposerProps) {
  const { t } = useTranslation()
  const [inputValue, setInputValue] = useState('')
  const [inputError, setInputError] = useState('')
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null)
  // The caret position the next render must restore, or null when it has
  // nothing to restore. Set only together with a draft change, so the layout
  // effect below consumes it in the very render that change causes.
  const pendingSelectionRef = useRef<number | null>(null)

  const dismissInputError = useCallback(() => {
    setInputError('')
    requestAnimationFrame(() => inputRef.current?.focus())
  }, [])

  // Smart switching logic: use Input for single line, Textarea for multi-line
  const hasMultipleLines = inputValue.includes('\n')

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const nextValue = e.target.value
      // Deleting the only newline replaces the textarea with an input. The
      // browser has already put the caret at the deletion point on the old
      // element, but the new element starts with its own selection. Remember
      // it here, while the old element is still mounted, so the layout effect
      // can restore it before the next keystroke.
      if (hasMultipleLines && !nextValue.includes('\n')) {
        pendingSelectionRef.current = e.target.selectionStart ?? null
      }
      setInputValue(nextValue)
    },
    [hasMultipleLines]
  )

  // Replace the draft and put the caret at `caret` once React has rendered it.
  // The caret cannot be set here: the change may swap the input for a
  // textarea (or back), and the new element only exists after the render.
  const replaceDraft = useCallback(
    (target: HTMLInputElement | HTMLTextAreaElement, nextValue: string, caret: number) => {
      if (nextValue === inputValue) {
        // React skips the render for an unchanged value, so nothing would
        // consume a pending caret; the element stays, so set it directly.
        target.setSelectionRange(caret, caret)
        return
      }
      pendingSelectionRef.current = caret
      setInputValue(nextValue)
    },
    [inputValue]
  )

  // Unified height adjustment function for textarea
  const adjustTextareaHeight = useCallback((element: HTMLTextAreaElement) => {
    requestAnimationFrame(() => {
      element.style.height = 'auto'
      element.style.height = Math.min(element.scrollHeight, 120) + 'px'
    })
  }, [])

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      if (!inputValue.trim() || isLoading) return

      const error = onSend(inputValue)
      if (error) {
        setInputError(error)
        return
      }
      setInputError('')
      setInputValue('')

      // Reset input height to minimum after clearing input
      if (inputRef.current && 'style' in inputRef.current) {
        inputRef.current.style.height = '40px'
      }
    },
    [inputValue, isLoading, onSend]
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && e.shiftKey) {
        // Shift+Enter: Insert newline
        e.preventDefault()
        const target = e.target as HTMLInputElement | HTMLTextAreaElement
        const start = target.selectionStart || 0
        const end = target.selectionEnd || 0
        const newValue = inputValue.slice(0, start) + '\n' + inputValue.slice(end)
        // The textarea height follows from the height effect below.
        replaceDraft(target, newValue, start + 1)
      } else if (e.key === 'Enter' && !e.shiftKey) {
        // Enter: Submit form
        e.preventDefault()
        handleSubmit(e as any)
      }
    },
    [inputValue, handleSubmit, replaceDraft]
  )

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      // Get pasted text content
      const pastedText = e.clipboardData.getData('text')

      // Check if it contains newlines
      if (pastedText.includes('\n')) {
        e.preventDefault() // Prevent default paste behavior

        // Get current cursor position
        const target = e.target as HTMLInputElement | HTMLTextAreaElement
        const start = target.selectionStart || 0
        const end = target.selectionEnd || 0

        // Build new value; pasted into an input, it switches to a textarea
        const newValue = inputValue.slice(0, start) + pastedText + inputValue.slice(end)
        replaceDraft(target, newValue, start + pastedText.length)
      }
      // If no newlines, let default paste behavior continue
    },
    [inputValue, replaceDraft]
  )

  // Restore the caret synchronously after the render. When the draft change
  // swapped the input for a textarea or back, the focused element was removed
  // and the new one starts unfocused with its own selection; a timer or an
  // animation frame would fix that too late, since the next keystroke can
  // arrive first and land in the wrong place or nowhere.
  useLayoutEffect(() => {
    const pending = pendingSelectionRef.current
    if (pending === null || !inputRef.current) return

    pendingSelectionRef.current = null
    const element = inputRef.current
    element.focus()
    if (element.setSelectionRange) {
      element.setSelectionRange(pending, pending)
    }
  }, [hasMultipleLines, inputValue])

  // Effect to handle component switching and maintain focus
  useEffect(() => {
    if (inputRef.current) {
      // Keep focus on the composer after renders the layout effect above does
      // not handle, such as the switch back to an input when a send clears a
      // multi-line draft. Edits restore their caret there, not here.
      const currentElement = inputRef.current

      // Use requestAnimationFrame to ensure DOM update is complete
      requestAnimationFrame(() => {
        currentElement.focus()
        if (currentElement.setSelectionRange) {
          const cursorPosition = currentElement.selectionStart ?? inputValue.length
          currentElement.setSelectionRange(cursorPosition, cursorPosition)
        }
      })
    }
  }, [hasMultipleLines, inputValue.length]) // Include inputValue.length dependency

  // Effect to adjust textarea height when switching to multi-line mode
  useEffect(() => {
    if (hasMultipleLines && inputRef.current && inputRef.current.tagName === 'TEXTAREA') {
      adjustTextareaHeight(inputRef.current as HTMLTextAreaElement)
    }
  }, [hasMultipleLines, inputValue, adjustTextareaHeight])

  return (
    <form
      onSubmit={handleSubmit}
      className="relative flex shrink-0 items-center gap-2"
      autoComplete="on"
      method="post"
      action="#"
      role="search"
    >
      {/* Hidden submit button to ensure form meets HTML standards */}
      <input type="submit" style={{ display: 'none' }} tabIndex={-1} />
      {inputError && (
        <div
          id="query-input-error"
          role="alert"
          className="border-destructive/35 bg-background/95 text-destructive absolute inset-x-0 bottom-full z-20 mb-2 flex items-start gap-2 rounded-lg border px-3 py-2 text-sm shadow-lg backdrop-blur-sm"
        >
          <CircleAlertIcon className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <span className="min-w-0 flex-1 break-words">{inputError}</span>
          <button
            type="button"
            onClick={dismissInputError}
            className="hover:bg-destructive/10 focus-visible:ring-ring relative -my-1 -me-1 flex size-8 shrink-0 items-center justify-center rounded-md after:absolute after:-inset-1.5 focus-visible:ring-2 focus-visible:outline-none"
            aria-label={t('retrievePanel.retrieval.dismissError')}
          >
            <XIcon className="size-4" aria-hidden="true" />
          </button>
        </div>
      )}
      <Button
        type="button"
        variant="outline"
        onClick={onClear}
        disabled={isLoading}
        size="sm"
        className="min-h-11"
      >
        <EraserIcon />
        {t('retrievePanel.retrieval.clear')}
      </Button>
      <div className="relative flex-1">
        <label htmlFor="query-input" className="sr-only">
          {t('retrievePanel.retrieval.placeholder')}
        </label>
        {hasMultipleLines ? (
          <Textarea
            ref={inputRef as React.RefObject<HTMLTextAreaElement>}
            id="query-input"
            autoComplete="on"
            className="max-h-[120px] min-h-[44px] w-full overflow-y-auto"
            value={inputValue}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={t('retrievePanel.retrieval.placeholder')}
            disabled={isLoading}
            aria-invalid={inputError ? true : undefined}
            aria-describedby={inputError ? 'query-input-error' : undefined}
            rows={1}
            style={{
              resize: 'none',
              height: 'auto',
              minHeight: '44px',
              maxHeight: '120px'
            }}
            onInput={(e: React.FormEvent<HTMLTextAreaElement>) => {
              const target = e.target as HTMLTextAreaElement
              requestAnimationFrame(() => {
                target.style.height = 'auto'
                target.style.height = Math.min(target.scrollHeight, 120) + 'px'
              })
            }}
          />
        ) : (
          <Input
            ref={inputRef as React.RefObject<HTMLInputElement>}
            id="query-input"
            autoComplete="on"
            className="min-h-11 w-full"
            value={inputValue}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={t('retrievePanel.retrieval.placeholder')}
            disabled={isLoading}
            aria-invalid={inputError ? true : undefined}
            aria-describedby={inputError ? 'query-input-error' : undefined}
          />
        )}
      </div>
      {/* Send and Stop swap in the SAME stable position (avoids stray taps);
          the stop cooldown is handled by the session controller. */}
      {isLoading ? (
        <Button
          type="button"
          variant="destructive"
          onClick={onStop}
          disabled={stopDisabled}
          size="sm"
          className="min-h-11 min-w-11"
          aria-label={t('retrievePanel.retrieval.stop')}
        >
          <SquareIcon />
          {t('retrievePanel.retrieval.stop')}
        </Button>
      ) : (
        <Button
          type="submit"
          variant="default"
          size="sm"
          className="min-h-11 min-w-11"
          aria-label={t('retrievePanel.retrieval.send')}
        >
          <SendIcon />
          {t('retrievePanel.retrieval.send')}
        </Button>
      )}
    </form>
  )
}
