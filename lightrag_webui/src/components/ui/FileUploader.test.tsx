/**
 * The uploader's per-file progress bar is a plain styled div pair; without
 * progressbar semantics assistive tech cannot read upload progress at all.
 * Each file's bar must expose its own value on the 0-100 scale and a name
 * that distinguishes it from the other files' bars.
 */
import { afterEach, describe, expect, test } from 'bun:test'
import { cleanup, screen } from '@testing-library/react'

import FileUploader from './FileUploader'
import { renderWithProviders } from '@/test/render'

afterEach(() => {
  cleanup()
})

const makeFile = (name: string) => new File(['content'], name, { type: 'text/plain' })

describe('FileUploader progress semantics', () => {
  test('each uploading file exposes a named progressbar with its own value', () => {
    renderWithProviders(
      <FileUploader
        value={[makeFile('alpha.txt'), makeFile('beta.txt')]}
        progresses={{ 'alpha.txt': 25, 'beta.txt': 80 }}
        onValueChange={() => {}}
      />
    )

    const alphaBar = screen.getByRole('progressbar', { name: 'Upload progress for alpha.txt' })
    const betaBar = screen.getByRole('progressbar', { name: 'Upload progress for beta.txt' })
    expect(alphaBar.getAttribute('aria-valuenow')).toBe('25')
    expect(betaBar.getAttribute('aria-valuenow')).toBe('80')
    expect(alphaBar.getAttribute('aria-valuemin')).toBe('0')
    expect(alphaBar.getAttribute('aria-valuemax')).toBe('100')
  })

  test('a failed upload still exposes a full progressbar beside the error', () => {
    renderWithProviders(
      <FileUploader
        value={[makeFile('alpha.txt')]}
        fileErrors={{ 'alpha.txt': 'Upload failed' }}
        onValueChange={() => {}}
      />
    )

    const bar = screen.getByRole('progressbar', { name: 'Upload progress for alpha.txt' })
    expect(bar.getAttribute('aria-valuenow')).toBe('100')
    expect(screen.queryAllByText('Upload failed')).toHaveLength(1)
  })
})
