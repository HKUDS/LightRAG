import { describe, expect, test } from 'bun:test'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'
import rehypeKatex from 'rehype-katex'

import { remarkFootnotes } from '@/utils/remarkFootnotes'
import { chatMarkdownSanitizeSchema } from '@/utils/markdownSanitizeSchema'

// Render Markdown through the SAME plugin pipeline ChatMessage uses for
// answer/thinking content — rehypeRaw → rehypeSanitize(schema) → rehypeKatex —
// minus the no-op rehypeReact compiler and the component style map, neither of
// which affects sanitization. This mirrors the GHSA-xpjq-3w4w-w5wr reporter's
// PoC harness, but with the sanitizer in place.
const render = (markdown: string): string =>
  renderToStaticMarkup(
    React.createElement(
      ReactMarkdown,
      {
        remarkPlugins: [remarkGfm, remarkFootnotes, remarkMath],
        rehypePlugins: [
          rehypeRaw,
          [rehypeSanitize, chatMarkdownSanitizeSchema],
          [rehypeKatex, { trust: false, throwOnError: false, strict: false }],
        ],
        skipHtml: false,
      } as React.ComponentProps<typeof ReactMarkdown>,
      markdown
    )
  )

describe('chat markdown sanitize — blocks stored XSS (GHSA-xpjq-3w4w-w5wr)', () => {
  test('strips <iframe srcdoc> (the reported PoC vector)', () => {
    const out = render(
      '<iframe srcdoc="<script>parent.document.title=1</script>"></iframe>'
    )
    expect(out).not.toContain('<iframe')
    expect(out).not.toContain('srcdoc')
  })

  test('strips <script> and <svg><script>', () => {
    expect(render('<script>alert(1)</script>')).not.toContain('<script')
    expect(render('<svg><script>alert(1)</script></svg>')).not.toContain('<script')
  })

  test('drops on* event-handler attributes', () => {
    const out = render('<img src="https://example.com/x.png" onerror="alert(1)">')
    expect(out).not.toContain('onerror')
  })

  test('neutralizes javascript: links', () => {
    const out = render('<a href="javascript:alert(1)">x</a>')
    expect(out).not.toContain('javascript:')
  })

  test('strips inline style attribute (defaultSchema policy)', () => {
    const out = render('<div style="position:fixed">x</div>')
    expect(out).not.toContain('style=')
  })
})

describe('chat markdown sanitize — preserves legitimate rendering', () => {
  test('keeps basic Markdown formatting', () => {
    const out = render('**bold** and *italic*\n\n- a\n- b')
    expect(out).toContain('<strong>')
    expect(out).toContain('<li>')
  })

  test('keeps <mark> and <u> (styled by chat CSS, added to schema)', () => {
    const out = render('<mark>hi</mark> <u>there</u>')
    expect(out).toContain('<mark>')
    expect(out).toContain('<u>')
  })

  test('renders inline and display KaTeX (math classes survive sanitize)', () => {
    const out = render('inline $a^2+b^2$ and\n\n$$\\int_0^1 x\\,dx$$')
    // rehype-katex only emits `.katex` markup if the math-inline/math-display
    // classes were preserved through sanitize.
    expect(out).toContain('katex')
  })

  test('keeps inline footnote refs from remarkFootnotes', () => {
    const out = render('claim[^1]')
    expect(out).toContain('#footnote-1')
    expect(out).toContain('footnote-ref')
  })

  test('GFM footnote ref/target ids stay consistent (no clobber double-prefix)', () => {
    // Full footnote (reference + definition) exercises remark-rehype's id
    // generation. Without clobberPrefix:'' hast-util-sanitize re-prefixes the
    // target `id` but not the ref `href`, desyncing them and breaking
    // navigation. Assert each in-page href fragment points at a real id, in
    // both directions (forward ref → definition, backref → reference).
    const out = render('A claim[^1].\n\n[^1]: The footnote body.')
    const fwd = out.match(/href="(#[^"]*fn-1)"/)
    expect(fwd).not.toBeNull()
    expect(out).toContain(`id="${fwd![1].slice(1)}"`)
    const back = out.match(/href="(#[^"]*fnref-1)"/)
    expect(back).not.toBeNull()
    expect(out).toContain(`id="${back![1].slice(1)}"`)
  })
})
