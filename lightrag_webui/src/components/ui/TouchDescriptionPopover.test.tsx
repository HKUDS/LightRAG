import { describe, expect, test } from 'bun:test'
import { readFileSync, readdirSync } from 'fs'
import { join } from 'path'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import TouchDescriptionPopover from './TouchDescriptionPopover'

const localesDir = join(import.meta.dir, '..', '..', 'locales')
const localeFiles = readdirSync(localesDir)
  .filter((file) => file.endsWith('.json'))
  .sort()

const DESCRIPTION = 'Production cluster — read only'

describe('touch description popover', () => {
  test('renders nothing when the deployment declares no description', () => {
    const { container } = renderWithProviders(<TouchDescriptionPopover description={null} />)

    // Not "hidden": absent. An empty trigger would still take layout width
    // in the header next to the brand title.
    expect(container).toBeEmptyDOMElement()
  })

  test('reveals the description when the trigger is activated', async () => {
    const user = userEvent.setup()
    renderWithProviders(<TouchDescriptionPopover description={DESCRIPTION} />)

    // Out of the tree until the trigger is used. The trigger is found by its
    // purpose, not by the text it reveals: reading the whole deployment blurb
    // out on focus is what naming it after the description would do.
    expect(screen.queryAllByText(DESCRIPTION)).toHaveLength(0)
    await user.click(screen.getByRole('button', { name: 'Deployment information' }))

    // This is what the source-text assertion could never reach: that Radix's
    // `asChild` trigger actually wires our Button up, and that the content
    // lands in the accessibility tree rather than merely existing in the DOM.
    expect(await screen.findByText(DESCRIPTION)).toBeVisible()
  })

  test('provides a short accessible name in every locale', () => {
    expect(localeFiles.length).toBeGreaterThan(0)

    for (const localeFile of localeFiles) {
      const locale = JSON.parse(readFileSync(join(localesDir, localeFile), 'utf8'))
      expect(locale.header.deploymentInfo).toBeTruthy()
    }
  })
})
