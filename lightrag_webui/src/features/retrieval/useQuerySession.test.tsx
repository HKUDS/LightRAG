import { afterEach, beforeEach, describe, expect, spyOn, test } from 'bun:test'
import { act, cleanup, fireEvent, screen } from '@testing-library/react'
import * as api from '@/api/lightrag'
import type { QueryRequest } from '@/api/lightrag'
import { ChatMessage } from '@/components/retrieval/ChatMessage'
import { createRetrievalHistoryStore } from '@/stores/retrievalHistory'
import type { RetrievalHistoryStore } from '@/stores/retrievalHistory'
import type { QuerySettings } from '@/stores/querySettings'
import type { MessageWithError } from '@/types/retrieval'
import { renderWithProviders } from '@/test/render'
import { useQuerySession } from './useQuerySession'

const HISTORY_KEY = 'test-query-session-error-history'
const FAILURE = 'LLM service failed: timeout'
const ANSWER = 'A successful answer'
const settings: QuerySettings = { mode: 'bypass', stream: true, history_turns: 0 }

function QuerySession({ historyStore, stream = true }: {
  historyStore: RetrievalHistoryStore
  stream?: boolean
}) {
  const session = useQuerySession({
    historyStore,
    getQuerySettingsSnapshot: () => ({ ...settings, stream })
  })
  return (
    <>
      <button disabled={session.isLoading} onClick={() => void session.submitQuery('My question')}>
        Send
      </button>
      {session.messages.map((message) => (
        <div key={message.id} role={message.isError ? 'alert' : undefined}>
          <ChatMessage message={message} />
        </div>
      ))}
    </>
  )
}

let previousHistory: string | null
let fetchSpy: ReturnType<typeof spyOn<typeof globalThis, 'fetch'>>
let querySpy: ReturnType<typeof spyOn<typeof api, 'queryText'>> | undefined
const requests: QueryRequest[] = []
let streamLines: Record<string, string>[]

beforeEach(() => {
  previousHistory = localStorage.getItem(HISTORY_KEY)
  localStorage.removeItem(HISTORY_KEY)
  requests.length = 0
  streamLines = [{ response: ANSWER }]
  const fetchResponse = async (_input: RequestInfo | URL, init?: RequestInit) => {
    requests.push(JSON.parse(init?.body as string))
    const body = streamLines.map((line) => JSON.stringify(line)).join('\n') + '\n'
    return new Response(body, { headers: { 'Content-Type': 'application/x-ndjson' } })
  }
  fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(fetchResponse as typeof fetch)
})

afterEach(() => {
  cleanup()
  fetchSpy.mockRestore()
  querySpy?.mockRestore()
  querySpy = undefined
  if (previousHistory === null) localStorage.removeItem(HISTORY_KEY)
  else localStorage.setItem(HISTORY_KEY, previousHistory)
})

const send = async () => {
  // Keep the request completion in the same React batch as submission. The
  // history write must not rely on React having evaluated a state updater.
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  })
}

const persistedAssistant = (): MessageWithError => {
  const envelope = JSON.parse(localStorage.getItem(HISTORY_KEY)!)
  return envelope.state.history.find((message: MessageWithError) => message.role === 'assistant')
}

describe('query error persistence', () => {
  for (const failure of ['stream', 'partial stream', 'non-stream'] as const) {
    test(`${failure} failures stay classified after reload and are excluded from bypass history`, async () => {
      const isStreaming = failure !== 'non-stream'
      if (isStreaming) {
        streamLines = failure === 'partial stream'
          ? [{ response: 'Partial answer' }, { error: FAILURE }]
          : [{ error: FAILURE }]
      } else {
        querySpy = spyOn(api, 'queryText').mockRejectedValue(new Error(FAILURE))
      }
      const view = renderWithProviders(
        <QuerySession historyStore={createRetrievalHistoryStore(HISTORY_KEY)} stream={isStreaming} />
      )
      await send()
      expect(screen.getByRole('alert').textContent).toContain(FAILURE)
      expect(persistedAssistant().isError).toBe(true)

      view.unmount()
      querySpy?.mockRestore()
      querySpy = undefined
      const restoredStore = createRetrievalHistoryStore(HISTORY_KEY)
      renderWithProviders(<QuerySession historyStore={restoredStore} />)
      expect(screen.getByRole('alert').textContent).toContain(FAILURE)
      expect(restoredStore.getState().history[1].isError).toBe(true)

      streamLines = [{ response: ANSWER }]
      requests.length = 0
      await send()
      expect(requests).toHaveLength(1)
      expect(requests[0].conversation_history).toEqual([{ role: 'user', content: 'My question' }])
      expect(screen.queryAllByText(ANSWER)).toHaveLength(1)
    })
  }

  for (const stream of [true, false]) {
    test(`${stream ? 'stream' : 'non-stream'} success remains eligible for bypass history after reload`, async () => {
      if (!stream) {
        querySpy = spyOn(api, 'queryText').mockResolvedValue({ response: ANSWER })
      }
      const view = renderWithProviders(
        <QuerySession historyStore={createRetrievalHistoryStore(HISTORY_KEY)} stream={stream} />
      )
      await send()
      expect(screen.queryAllByRole('alert')).toHaveLength(0)
      expect(persistedAssistant().isError === true).toBe(false)
      expect(persistedAssistant().content).toBe(ANSWER)

      view.unmount()
      querySpy?.mockRestore()
      querySpy = undefined
      renderWithProviders(<QuerySession historyStore={createRetrievalHistoryStore(HISTORY_KEY)} />)
      expect(screen.queryAllByRole('alert')).toHaveLength(0)
      expect(screen.queryAllByText(ANSWER)).toHaveLength(1)

      requests.length = 0
      await send()
      expect(requests).toHaveLength(1)
      expect(requests[0].conversation_history).toEqual([
        { role: 'user', content: 'My question' },
        { role: 'assistant', content: ANSWER }
      ])
    })
  }
})
