import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn } from '@/lib/utils'

/**
 * The `customization` Markdown tier — one of the two EXPLICITLY NAMED
 * rendering profiles (the other being the `chat` tier embodied by
 * `chatMarkdownSanitizeSchema` + rehypeRaw in ChatMessage). Do not "unify"
 * them: chat content needs raw-HTML parsing for footnotes/inline tags with an
 * allow-list; customization content has no such need, so this tier simply
 * does not open a raw HTML path at all (`skipHtml` — HTML is DROPPED, which
 * is a product format boundary, not distrust of the bundle author).
 *
 * Frontend DEFAULT welcome/empty-state content renders through this exact
 * component too, so bundle-vs-default deployments show identical rendering
 * behavior and the default path keeps the same protections.
 */

/**
 * How much vertical rhythm the block keeps.
 *
 * - `compact` (default): tightened paragraph/heading margins, for the short
 *   blurbs that sit INSIDE another layout — a welcome card, the login-page
 *   text, an empty state — where the prose scale would dwarf its container.
 * - `document`: the typography scale exactly as configured, for content that
 *   IS the page — the user-agreement dialog. A legal document has to read
 *   like the Markdown file it came from: its own heading hierarchy, list,
 *   table and quote spacing intact, not squeezed into a caption's rhythm.
 */
export type CustomizedMarkdownVariant = 'compact' | 'document'

const VARIANT_CLASSES: Record<CustomizedMarkdownVariant, string> = {
  compact: 'prose-p:my-2 prose-headings:my-3',
  document: ''
}

export default function CustomizedMarkdown({
  content,
  className,
  dir,
  variant = 'compact'
}: {
  content: string
  className?: string
  /** Writing direction for this block, from the server's resolved locale
   * (never from bundle-provided markup). Omit to inherit the page's. */
  dir?: 'ltr' | 'rtl'
  variant?: CustomizedMarkdownVariant
}) {
  return (
    <div
      dir={dir}
      className={cn(
        'prose dark:prose-invert max-w-none break-words',
        VARIANT_CLASSES[variant],
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ children, ...props }) => (
            // Tab-nabbing hygiene on new-window links (zero cost).
            <a {...props} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          )
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
