import React from 'react'
import { describe, expect, test, vi } from 'vitest'
import { renderToString } from 'react-dom/server'

Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: vi.fn(() => null),
    setItem: vi.fn(),
    removeItem: vi.fn()
  },
  configurable: true
})

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key
  })
}))

vi.mock('sonner', () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn()
  }
}))

describe('CancelPipelineButton', () => {
  test('renders direct cancel button text and disabled state when pipeline is idle', async () => {
    const module = await import('./CancelPipelineButton')
    const CancelPipelineButton = module.default

    const html = renderToString(<CancelPipelineButton busy={false} />)

    expect(html).toContain('documentPanel.documentManager.cancelPipelineButton')
    expect(html).toContain('disabled')
  })
})
