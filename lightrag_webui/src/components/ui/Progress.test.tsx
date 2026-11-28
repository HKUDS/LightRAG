import { afterEach, describe, expect, test } from 'bun:test'
import { cleanup, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import Progress from './Progress'

afterEach(cleanup)

describe('Progress value semantics', () => {
  test.each([
    ['forwards value to the progressbar as aria-valuenow', 40, '40'],
    ['no value stays indeterminate', undefined, null]
  ] as const)('%s', (_name, value, valueNow) => {
    renderWithProviders(<Progress value={value} />)
    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe(valueNow)
  })
})
