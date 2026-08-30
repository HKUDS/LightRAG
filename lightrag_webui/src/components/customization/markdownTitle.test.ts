import { describe, expect, test } from 'bun:test'
import { extractMarkdownTitle } from './markdownTitle'

describe('extractMarkdownTitle', () => {
  test('the first ATX heading is the title', () => {
    expect(extractMarkdownTitle('# User Agreement\n\nBody text.')).toBe(
      'User Agreement'
    )
  })

  test('any heading level counts — a document may open at h2', () => {
    expect(extractMarkdownTitle('## 用户协议\n\n正文')).toBe('用户协议')
  })

  test('leading content before the heading is skipped', () => {
    const doc = '> A note to the reader.\n\nSome text.\n\n# Terms of Service\n'
    expect(extractMarkdownTitle(doc)).toBe('Terms of Service')
  })

  test('a closing sequence is decoration, not part of the title', () => {
    expect(extractMarkdownTitle('## Terms ##')).toBe('Terms')
  })

  test('inline markup is stripped so the announced name is words', () => {
    expect(extractMarkdownTitle('# **Terms** of `Service`')).toBe('Terms of Service')
    expect(extractMarkdownTitle('# [Terms](https://example.com/tos)')).toBe('Terms')
    expect(extractMarkdownTitle('# ~~Old~~ *Terms*')).toBe('Old Terms')
    expect(extractMarkdownTitle('# Terms \\# 2')).toBe('Terms # 2')
  })

  test('setext headings count — bundle authors write what their editor emits', () => {
    expect(extractMarkdownTitle('User Agreement\n==============\n\nBody')).toBe(
      'User Agreement'
    )
    expect(extractMarkdownTitle('User Agreement\n---\n\nBody')).toBe('User Agreement')
  })

  test('a leading horizontal rule is not read as a setext heading', () => {
    // Nothing precedes the rule, so there is no heading text to take.
    expect(extractMarkdownTitle('---\n\nJust a paragraph.')).toBeNull()
  })

  test('a `#` inside a fenced code block is not the title', () => {
    const doc = '```bash\n# not a heading\n```\n\n# Real Title\n'
    expect(extractMarkdownTitle(doc)).toBe('Real Title')
  })

  test('a fence closed by a different character stays open', () => {
    // `~~~` cannot close a ``` fence, so everything after it is still code.
    expect(extractMarkdownTitle('```\n~~~\n# inside code\n')).toBeNull()
  })

  test('an empty heading is skipped rather than announced as a blank name', () => {
    expect(extractMarkdownTitle('#\n\n## Terms')).toBe('Terms')
  })

  test('a document with no heading has no title', () => {
    expect(extractMarkdownTitle('Just a paragraph of legal text.')).toBeNull()
    expect(extractMarkdownTitle('')).toBeNull()
    expect(extractMarkdownTitle(null)).toBeNull()
  })

  test('CRLF documents parse identically', () => {
    expect(extractMarkdownTitle('# Terms\r\n\r\nBody\r\n')).toBe('Terms')
  })
})
