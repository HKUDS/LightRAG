import { useCallback, useEffect, useRef, useState } from 'react'
import Textarea from '@/components/ui/Textarea'
import Input from '@/components/ui/Input'
import Button from '@/components/ui/Button'
import { EraserIcon, SendIcon, SquareIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

/**
 * Shared input layer: single/multi-line switching, draft handling, clear,
 * send/stop with the stop cooldown. Contains no QuerySettings and no admin
 * navigation. Page differences (the admin `/mode` prefix) live in `onSend`,
 * provided by the page composition layer: it returns an error string to show
 * under the input, or null to accept (which clears the draft).
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

  // Smart switching logic: use Input for single line, Textarea for multi-line
  const hasMultipleLines = inputValue.includes('\n')

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setInputValue(e.target.value)
      if (inputError) setInputError('')
    },
    [inputError]
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
        setInputValue(newValue)

        // Set cursor position after the newline and adjust height if needed
        setTimeout(() => {
          if (target.setSelectionRange) {
            target.setSelectionRange(start + 1, start + 1)
          }

          // Manually trigger height adjustment for textarea after component switch
          if (inputRef.current && inputRef.current.tagName === 'TEXTAREA') {
            adjustTextareaHeight(inputRef.current as HTMLTextAreaElement)
          }
        }, 0)
      } else if (e.key === 'Enter' && !e.shiftKey) {
        // Enter: Submit form
        e.preventDefault()
        handleSubmit(e as any)
      }
    },
    [inputValue, handleSubmit, adjustTextareaHeight]
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

        // Build new value
        const newValue = inputValue.slice(0, start) + pastedText + inputValue.slice(end)

        // Update state (this will trigger component switch to Textarea)
        setInputValue(newValue)

        // Set cursor position to end of pasted content
        setTimeout(() => {
          if (inputRef.current && inputRef.current.setSelectionRange) {
            const newCursorPosition = start + pastedText.length
            inputRef.current.setSelectionRange(newCursorPosition, newCursorPosition)
          }
        }, 0)
      }
      // If no newlines, let default paste behavior continue
    },
    [inputValue]
  )

  // Effect to handle component switching and maintain focus
  useEffect(() => {
    if (inputRef.current) {
      // When component type changes, restore focus and cursor position
      const currentElement = inputRef.current
      const cursorPosition = currentElement.selectionStart || inputValue.length

      // Use requestAnimationFrame to ensure DOM update is complete
      requestAnimationFrame(() => {
        currentElement.focus()
        if (currentElement.setSelectionRange) {
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
      className="flex shrink-0 items-center gap-2"
      autoComplete="on"
      method="post"
      action="#"
      role="search"
    >
      {/* Hidden submit button to ensure form meets HTML standards */}
      <input type="submit" style={{ display: 'none' }} tabIndex={-1} />
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
      <div className="flex-1 relative">
        <label htmlFor="query-input" className="sr-only">
          {t('retrievePanel.retrieval.placeholder')}
        </label>
        {hasMultipleLines ? (
          <Textarea
            ref={inputRef as React.RefObject<HTMLTextAreaElement>}
            id="query-input"
            autoComplete="on"
            className="w-full min-h-[44px] max-h-[120px] overflow-y-auto"
            value={inputValue}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={t('retrievePanel.retrieval.placeholder')}
            disabled={isLoading}
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
            className="w-full min-h-11"
            value={inputValue}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={t('retrievePanel.retrieval.placeholder')}
            disabled={isLoading}
          />
        )}
        {/* Error message below input */}
        {inputError && (
          <div className="absolute left-0 top-full mt-1 text-xs text-red-500">
            {inputError}
          </div>
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
