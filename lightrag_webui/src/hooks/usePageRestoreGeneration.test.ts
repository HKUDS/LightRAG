import { describe, expect, test } from 'bun:test'
import { subscribeToPageRestore } from './usePageRestoreGeneration'

const pageShowEvent = (persisted: boolean): Event => {
  const event = new Event('pageshow')
  Object.defineProperty(event, 'persisted', { value: persisted })
  return event
}

describe('subscribeToPageRestore', () => {
  test('notifies only for BFCache restores', () => {
    const target = new EventTarget()
    let restores = 0
    const unsubscribe = subscribeToPageRestore(() => {
      restores += 1
    }, target)

    target.dispatchEvent(pageShowEvent(false))
    target.dispatchEvent(pageShowEvent(true))
    target.dispatchEvent(pageShowEvent(true))

    expect(restores).toBe(2)
    unsubscribe()
  })

  test('stops notifying after cleanup', () => {
    const target = new EventTarget()
    let restores = 0
    const unsubscribe = subscribeToPageRestore(() => {
      restores += 1
    }, target)

    unsubscribe()
    target.dispatchEvent(pageShowEvent(true))

    expect(restores).toBe(0)
  })
})

describe('BFCache consumers', () => {
  test('admin health and document polling restart without beforeunload poisoning', async () => {
    const appSource = await Bun.file(new URL('../App.tsx', import.meta.url)).text()
    const documentManagerSource = await Bun.file(
      new URL('../features/DocumentManager.tsx', import.meta.url)
    ).text()

    for (const source of [appSource, documentManagerSource]) {
      expect(source).toContain('const pageRestoreGeneration = usePageRestoreGeneration()')
      expect(source).not.toContain('addEventListener(\'beforeunload\'')
    }

    expect(appSource).toContain('[enableHealthCheck, apiKeyAlertOpen, pageRestoreGeneration]')
    expect(documentManagerSource).toContain(
      'pipelineActive, pageRestoreGeneration, startPollingInterval'
    )
    expect(documentManagerSource).toContain('pageRestoreGeneration,\n    fetchPaginatedDocuments')
  })

  test('the initial document state is distinct from a confirmed empty response', async () => {
    const source = await Bun.file(
      new URL('../features/DocumentManager.tsx', import.meta.url)
    ).text()

    expect(source).toContain('!hasLoadedDocuments && initialLoadError === null')
    expect(source).toContain('!hasLoadedDocuments && initialLoadError !== null')
    // A loaded-but-empty corpus is told apart from the pre-load state by the
    // pagination total, not by a separate grouped-documents state.
    expect(source).toContain('hasLoadedDocuments && pagination.total_count === 0')
  })
})
