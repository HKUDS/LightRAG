/// <reference types="bun" />
import { afterEach, describe, expect, mock, test } from 'bun:test'
import { cleanup, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import { EditIcon } from './PropertyRowComponents'

/**
 * EditIcon is the SOLE entry into property editing/merging. It must be a real
 * button: a clickable svg is invisible to the keyboard, so Tab can never reach
 * it and Enter/Space can never fire it. The accessible name comes from the
 * shared editProperty key with the TRANSLATED property name, so a screen
 * reader announces the same name the visible PropertyName label shows.
 */
describe('EditIcon', () => {
  afterEach(() => {
    cleanup()
  })

  test('is a named button in the tab order that click, Enter and Space all activate', async () => {
    const user = userEvent.setup()
    const onClick = mock(() => {})
    renderWithProviders(<EditIcon onClick={onClick} propertyName="entity_id" />)

    // entity_id is displayed as "Name" (propertyNames.entity_id), so the
    // button must announce "Edit Name", not the raw key.
    const button = screen.getByRole('button', { name: 'Edit Name' })
    expect(button.getAttribute('type')).toBe('button')
    // happy-dom computes no layout, so the visible focus ring is pinned as a class.
    expect(button.className.split(/\s+/)).toContain('focus-visible:ring-2')

    // Tab reaches it from an unfocused page (a bare svg would be skipped).
    await user.tab()
    expect(document.activeElement === button).toBe(true)

    await user.keyboard('{Enter}')
    expect(onClick).toHaveBeenCalledTimes(1)
    await user.keyboard(' ')
    expect(onClick).toHaveBeenCalledTimes(2)
    await user.click(button)
    expect(onClick).toHaveBeenCalledTimes(3)
  })

  test('a property without a translation keeps its raw name in the accessible name', () => {
    renderWithProviders(<EditIcon onClick={mock(() => {})} propertyName="custom_field" />)

    const button = screen.getByRole('button', { name: 'Edit custom_field' })
    expect(button.getAttribute('aria-label')).toBe('Edit custom_field')
  })
})
