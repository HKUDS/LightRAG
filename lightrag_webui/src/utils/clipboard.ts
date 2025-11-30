/**
 * Robust clipboard utility with multiple fallback strategies
 * Handles various browser environments and security contexts
 */

export interface CopyResult {
  success: boolean
  method: 'clipboard-api' | 'execCommand' | 'manual-select' | 'fallback'
  error?: string
}

/**
 * Copy text to clipboard with multiple fallback strategies
 * @param text - Text to copy to clipboard
 * @returns Promise<CopyResult> - Result object with success status and method used
 */
export async function copyToClipboard(text: string): Promise<CopyResult> {
  if (!text || text.trim() === '') {
    return {
      success: false,
      method: 'fallback',
      error: 'No text provided',
    }
  }

  // Strategy 1: Modern Clipboard API (preferred)
  if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
    try {
      await navigator.clipboard.writeText(text)
      return {
        success: true,
        method: 'clipboard-api',
      }
    } catch (error) {
      console.warn('Clipboard API failed:', error)
      // Continue to fallback methods
    }
  }

  // Strategy 2: Legacy execCommand (for older browsers)
  try {
    const result = await copyWithExecCommand(text)
    if (result.success) {
      return result
    }
  } catch (error) {
    console.warn('execCommand failed:', error)
    // Continue to fallback methods
  }

  // Strategy 3: Manual text selection (most compatible)
  try {
    const result = await copyWithManualSelection(text)
    if (result.success) {
      return result
    }
  } catch (error) {
    console.warn('Manual selection failed:', error)
  }

  // Strategy 4: Complete fallback - return error
  return {
    success: false,
    method: 'fallback',
    error: 'All copy methods failed. Please copy the text manually.',
  }
}

/**
 * Copy using legacy execCommand method
 */
async function copyWithExecCommand(text: string): Promise<CopyResult> {
  return new Promise((resolve) => {
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.style.position = 'fixed'
    textarea.style.left = '-9999px'
    textarea.style.top = '-9999px'
    textarea.style.opacity = '0'
    textarea.setAttribute('readonly', '')

    document.body.appendChild(textarea)

    try {
      textarea.select()
      textarea.setSelectionRange(0, text.length)

      const successful = document.execCommand('copy')

      if (successful) {
        resolve({
          success: true,
          method: 'execCommand',
        })
      } else {
        resolve({
          success: false,
          method: 'execCommand',
          error: 'execCommand returned false',
        })
      }
    } catch (error) {
      resolve({
        success: false,
        method: 'execCommand',
        error: error instanceof Error ? error.message : 'execCommand failed',
      })
    } finally {
      document.body.removeChild(textarea)
    }
  })
}

/**
 * Copy using manual text selection method
 */
async function copyWithManualSelection(text: string): Promise<CopyResult> {
  return new Promise((resolve) => {
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.style.position = 'absolute'
    textarea.style.left = '-9999px'
    textarea.style.top = '-9999px'
    textarea.style.opacity = '0'
    textarea.style.pointerEvents = 'none'
    textarea.setAttribute('readonly', '')
    textarea.setAttribute('tabindex', '-1')

    document.body.appendChild(textarea)

    try {
      // Focus and select the text
      textarea.focus()
      textarea.select()
      textarea.setSelectionRange(0, text.length)

      // Try to trigger copy event
      const copyEvent = new ClipboardEvent('copy', {
        clipboardData: new DataTransfer(),
      })

      if (copyEvent.clipboardData) {
        copyEvent.clipboardData.setData('text/plain', text)
        document.dispatchEvent(copyEvent)

        resolve({
          success: true,
          method: 'manual-select',
        })
      } else {
        // Fallback: keep text selected for manual copy
        resolve({
          success: false,
          method: 'manual-select',
          error: 'Manual selection prepared, but automatic copy failed',
        })
      }
    } catch (error) {
      resolve({
        success: false,
        method: 'manual-select',
        error: error instanceof Error ? error.message : 'Manual selection failed',
      })
    } finally {
      // Clean up after a short delay to allow copy operation
      setTimeout(() => {
        if (document.body.contains(textarea)) {
          document.body.removeChild(textarea)
        }
      }, 100)
    }
  })
}

/**
 * Check if clipboard functionality is available
 */
export function isClipboardSupported(): boolean {
  return !!(
    (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') ||
    typeof document !== 'undefined'
  )
}

/**
 * Get the best available clipboard method
 */
export function getBestClipboardMethod():
  | 'clipboard-api'
  | 'execCommand'
  | 'manual-select'
  | 'none' {
  if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
    return 'clipboard-api'
  }

  if (typeof document !== 'undefined') {
    return 'execCommand'
  }

  return 'none'
}
