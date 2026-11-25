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
  // DocumentManager's half is rendered in features/DocumentManager.test.tsx
  // ("DocumentManager BFCache restore"). App stays a source read: rendering it
  // pulls in the whole admin shell for one dependency list.
  test('admin health polling restarts without beforeunload poisoning', async () => {
    const appSource = (await Bun.file(new URL('../App.tsx', import.meta.url)).text()).replace(/\r\n/g, '\n')

    expect(appSource).toContain('const pageRestoreGeneration = usePageRestoreGeneration()')
    expect(appSource).not.toContain('addEventListener(\'beforeunload\'')
    expect(appSource).toContain('[enableHealthCheck, pageRestoreGeneration]')
  })
})
