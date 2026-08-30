import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const source = readFileSync(join(import.meta.dir, 'WorkspaceApp.tsx'), 'utf8')

describe('workspace header tooltip accessibility', () => {
  test('uses the focusable brand link as the tooltip trigger', () => {
    const triggerStart = source.indexOf('<TooltipTrigger asChild>')
    const triggerEnd = source.indexOf('</TooltipTrigger>', triggerStart)
    const triggerSource = source.slice(triggerStart, triggerEnd)

    expect(triggerStart).toBeGreaterThan(-1)
    expect(triggerEnd).toBeGreaterThan(triggerStart)
    expect(triggerSource).toContain('<a href={entryHomeHref(window.location.pathname)}')
    expect(triggerSource).toContain('{webuiTitle || \'LightRAG\'}')
    expect(triggerSource).toContain('</a>')
    expect(source).toContain('<TouchDescriptionPopover description={webuiDescription} />')
  })
})
