import { afterEach, describe, expect, test } from 'bun:test'
import { cleanup, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import { Tabs } from '@/components/ui/Tabs'
import { useAuthStore } from '@/stores/state'
import SiteHeader from './SiteHeader'

const DESCRIPTION = 'Production cluster — read only'

// SiteHeader lives inside the app's <Tabs> root (App.tsx), and its
// TabsNavigation would throw outside one. Nothing else about Tabs matters
// here, so the value is fixed.
const renderHeader = () =>
  renderWithProviders(
    <Tabs value="documents">
      <SiteHeader />
    </Tabs>
  )

const setDeployment = (webuiTitle: string | null, webuiDescription: string | null): void => {
  useAuthStore.setState({ webuiTitle, webuiDescription })
}

afterEach(() => {
  // Unmount FIRST: a file-local `afterEach` runs BEFORE the preload's
  // Testing Library `cleanup()`, so resetting a store here would otherwise
  // update a still-mounted component. `cleanup` is idempotent, so the
  // preload's own call is still fine.
  cleanup()
  // The store is a module singleton shared by every test file in the run,
  // so it is reset rather than left dirty.
  setDeployment(null, null)
})

describe('admin header deployment identity', () => {
  test('a keyboard user reaches the brand link first and it describes the deployment', async () => {
    const user = userEvent.setup()
    setDeployment('Acme KB', DESCRIPTION)
    renderHeader()

    // This is the accessibility property the header exists to satisfy: the
    // tooltip hangs off the FOCUSABLE brand link, so the description is
    // reachable without a pointer. A non-focusable wrapper would render
    // identically and silently strand keyboard users.
    await user.tab()

    const link = screen.getByRole('link', { name: /LightRAG/ })
    // Compared as a boolean: a failing `toBe(element)` asks Bun to serialise
    // the DOM node, which costs ~10x the runtime and buries the reason.
    expect(document.activeElement === link).toBe(true)
    expect(await screen.findByText(DESCRIPTION)).toBeVisible()
    expect(link).toHaveAttribute('aria-describedby')
  })

  test('shows the product name and the deployment title inside that one link', () => {
    setDeployment('Acme KB', null)
    renderHeader()

    const link = screen.getByRole('link', { name: /LightRAG/ })
    // Both inside the link, so the whole brand block is one target — the
    // title is not a separate, unfocusable label beside it.
    expect(within(link).getByText('LightRAG')).toBeInTheDocument()
    expect(within(link).getByText('Acme KB')).toBeInTheDocument()
  })

  test('an uncustomized deployment shows the product name alone', () => {
    setDeployment(null, null)
    renderHeader()

    const link = screen.getByRole('link', { name: /LightRAG/ })
    expect(within(link).getByText('LightRAG')).toBeInTheDocument()
    // No empty separator or blank label left behind.
    expect(link).toHaveTextContent(/^LightRAG$/)
  })

  test('offers the touch fallback only when there is a description to show', () => {
    setDeployment('Acme KB', DESCRIPTION)
    const { unmount } = renderHeader()

    // Hover cannot reach a tooltip on a touch device; the popover button is
    // the same information behind an explicit control.
    expect(screen.getByRole('button', { name: 'Deployment information' })).toBeInTheDocument()
    unmount()

    setDeployment('Acme KB', null)
    renderHeader()
    expect(screen.queryAllByRole('button', { name: 'Deployment information' })).toHaveLength(0)
  })

  test('links back to this entry, never to a sibling entry root', () => {
    setDeployment(null, null)
    renderHeader()

    // Document-relative: under a reverse-proxy prefix or the dev server's
    // file-per-entry layout this still resolves to THIS entry (entryHomeHref).
    const link = screen.getByRole('link', { name: /LightRAG/ })
    expect(link.getAttribute('href')).toBe('./')
  })
})
