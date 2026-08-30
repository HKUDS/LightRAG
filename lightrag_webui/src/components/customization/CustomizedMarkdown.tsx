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
export default function CustomizedMarkdown({
  content,
  className
}: {
  content: string
  className?: string
}) {
  return (
    <div
      className={cn(
        'prose dark:prose-invert max-w-none break-words prose-p:my-2 prose-headings:my-3',
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
