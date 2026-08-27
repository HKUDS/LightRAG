import { useState } from 'react'
import CustomizedMarkdown from '@/components/customization/CustomizedMarkdown'
import { useCustomizedContent } from '@/components/customization/useCustomizedContent'

/**
 * The workspace query page's empty state: customizable logo + welcome text,
 * vertically centered in the message area. Shown while the history is empty;
 * disappears after the first question, reappears after Clear.
 */
export default function WorkspaceEmptyState() {
  const content = useCustomizedContent()
  const [logoFailed, setLogoFailed] = useState(false)

  if (content.loading) {
    // Loading placeholder — never flash the default content before knowing
    // whether a bundle is active (§8.8).
    return (
      <div
        className="border-primary size-8 animate-spin rounded-full border-4 border-t-transparent"
        role="status"
        aria-label="Loading"
      />
    )
  }

  return (
    <div
      dir={content.direction}
      className="flex max-w-prose flex-col items-center gap-4 px-4 text-center"
    >
      {content.logoUrl && !logoFailed && (
        <img
          src={content.logoUrl}
          alt={content.logoAlt}
          // Aspect ratio preserved; 120px max edge on desktop, 88px on phones.
          className="max-h-[88px] max-w-[88px] object-contain md:max-h-[120px] md:max-w-[120px]"
          onError={() => setLogoFailed(true)}
        />
      )}
      <CustomizedMarkdown
        content={content.queryEmptyMarkdown}
        className="text-muted-foreground text-base"
      />
    </div>
  )
}
