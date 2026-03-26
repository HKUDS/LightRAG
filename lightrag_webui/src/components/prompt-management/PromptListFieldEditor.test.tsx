import React from 'react'
import { describe, expect, test, vi } from 'vitest'
import { renderToString } from 'react-dom/server'

import PromptListFieldEditor from './PromptListFieldEditor'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key
  })
}))

describe('PromptListFieldEditor', () => {
  test('renders roomy list editors with vertical resize support', () => {
    const html = renderToString(
      <PromptListFieldEditor
        value={['Example block']}
        onChange={() => {}}
        placeholder="Example"
        itemLabel="Example"
      />
    )

    expect(html).toContain('min-h-[160px]')
    expect(html).toContain('resize-y')
  })
})
