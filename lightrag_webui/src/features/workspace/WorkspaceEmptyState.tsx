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
  // Latch the URL that failed, not a boolean: a transient failure must not
  // keep hiding the logo after a language switch supplies a different URL.
  const [failedLogoUrl, setFailedLogoUrl] = useState<string | null>(null)

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
      {content.logoUrl && failedLogoUrl !== content.logoUrl && (
        <img
          src={content.logoUrl}
          alt={content.logoAlt}
          // Aspect ratio preserved; 120px max edge on desktop, 88px on phones.
          className="max-h-[88px] max-w-[88px] object-contain md:max-h-[120px] md:max-w-[120px]"
          onError={() => setFailedLogoUrl(content.logoUrl)}
        />
      )}
      <CustomizedMarkdown
        content={content.queryEmptyMarkdown}
        className="text-muted-foreground text-base"
      />
    </div>
  )
}
