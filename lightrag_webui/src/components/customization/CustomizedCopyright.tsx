import { cn } from '@/lib/utils'

/**
 * The deployment's copyright line, rendered at the FOOT OF THE PAGE — outside
 * the login / welcome card, not inside it.
 *
 * Renders nothing at all unless a customization bundle declares the text:
 * LightRAG has no default line, so an uncustomized deployment shows none, and
 * a customer's page never carries LightRAG's own notice.
 *
 * Empty and whitespace-only are the same state as undeclared, and that is
 * decided HERE as well as upstream (the bundle loader strips, and so does
 * `useCustomizedContent`) — a footer element holding one space still draws
 * padding at the foot of every page, so the last mile checks rather than
 * trusts.
 *
 * Plain text, never Markdown: this is one short line whose whole job is to
 * state a legal fact, and giving it a renderer able to pull in headings,
 * images or links would let a footer redraw the page it sits under.
 */
export default function CustomizedCopyright({
  copyright,
  direction,
  className
}: {
  copyright: string
  direction?: 'ltr' | 'rtl'
  className?: string
}) {
  const text = copyright.trim()
  if (!text) return null

  return (
    <footer
      dir={direction}
      className={cn(
        'text-muted-foreground w-full shrink-0 text-center text-xs',
        className
      )}
    >
      {text}
    </footer>
  )
}
