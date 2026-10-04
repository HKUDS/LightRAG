import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import FileUploader from './FileUploader'

const createDescriptor = Object.getOwnPropertyDescriptor(URL, 'createObjectURL')
const revokeDescriptor = Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL')
let nextPreview = 0
const createObjectURL = mock(() => `blob:upload-${++nextPreview}`)
const revokeObjectURL = mock((url: string) => { void url })

beforeEach(() => {
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL })
})

afterEach(() => {
  cleanup()
  if (createDescriptor) Object.defineProperty(URL, 'createObjectURL', createDescriptor)
  else Reflect.deleteProperty(URL, 'createObjectURL')
  if (revokeDescriptor) Object.defineProperty(URL, 'revokeObjectURL', revokeDescriptor)
  else Reflect.deleteProperty(URL, 'revokeObjectURL')
  createObjectURL.mockClear()
  revokeObjectURL.mockClear()
  nextPreview = 0
})

const addFile = async (container: HTMLElement, name: string) => {
  const input = container.querySelector('input[type="file"]')!
  fireEvent.change(input, {
    target: { files: [new File(['content'], name, { type: 'text/plain' })] }
  })
  await screen.findByText(name)
}

describe('FileUploader preview resources', () => {
  test('releases files added after mounting when unmounted', async () => {
    const { container, unmount } = renderWithProviders(<FileUploader maxFileCount={3} />)
    await addFile(container, 'first.txt')
    await addFile(container, 'second.txt')
    expect(createObjectURL).toHaveBeenCalledTimes(2)
    expect(revokeObjectURL).toHaveBeenCalledTimes(0)

    unmount()
    expect(revokeObjectURL).toHaveBeenCalledTimes(2)
    expect(revokeObjectURL.mock.calls.map(([url]) => url)).toEqual(['blob:upload-1', 'blob:upload-2'])
  })

  test('releases removed files immediately and keeps remaining previews alive', async () => {
    const { container, unmount } = renderWithProviders(<FileUploader maxFileCount={3} />)
    await addFile(container, 'first.txt')
    await addFile(container, 'second.txt')
    fireEvent.click(screen.getAllByRole('button', { name: /remove file/i })[0])
    await waitFor(() => expect(screen.queryByText('first.txt')).toBeNull())
    expect(revokeObjectURL.mock.calls.map(([url]) => url)).toEqual(['blob:upload-1'])

    unmount()
    expect(revokeObjectURL.mock.calls.map(([url]) => url)).toEqual(['blob:upload-1', 'blob:upload-2'])
  })
})
