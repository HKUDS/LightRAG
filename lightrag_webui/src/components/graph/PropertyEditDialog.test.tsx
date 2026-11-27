/// <reference types="bun" />
import { afterEach, describe, expect, test } from 'bun:test'
import { cleanup, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import PropertyEditDialog from './PropertyEditDialog'

const noop = () => {}

const renderDialog = (propertyName: string, value: string) =>
  renderWithProviders(
    <PropertyEditDialog
      isOpen
      onClose={noop}
      onSave={noop}
      propertyName={propertyName}
      value={value}
      allowMerge={false}
      onValueChange={noop}
      onAllowMergeChange={noop}
    />
  )

/**
 * The textarea is the dialog's primary control. Without an accessible name a
 * screen reader announces only "edit text", leaving the user unsure which
 * property they are about to overwrite. Both branches (description and the
 * token-like entity_id) must carry the name; entity_id additionally must not
 * offer autocomplete or spellcheck on a name token.
 */
describe('PropertyEditDialog', () => {
  afterEach(() => {
    cleanup()
  })

  test('the description textarea is named after the property and carries the draft', () => {
    renderDialog('description', 'A description draft')

    const textarea = screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Edit Description' })
    expect(textarea.value).toBe('A description draft')
  })

  test('the entity_id textarea is named and disables autocomplete and spellcheck', () => {
    renderDialog('entity_id', 'SomeEntity')

    const textarea = screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Edit Name' })
    expect(textarea.value).toBe('SomeEntity')
    expect(textarea.getAttribute('spellcheck')).toBe('false')
    expect(textarea.getAttribute('autocomplete')).toBe('off')
  })
})
