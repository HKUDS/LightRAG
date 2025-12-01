/**
 * CitationMarker Component
 *
 * Renders citation markers (e.g., [1]) as interactive hover cards
 * showing source metadata like document title, section, page, and excerpt.
 */

import type { CitationSource } from '@/api/lightrag'
import Badge from '@/components/ui/Badge'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/HoverCard'
import { FileTextIcon } from 'lucide-react'

interface CitationMarkerProps {
  /** The citation marker text, e.g., "[1]" or "[1,2]" */
  marker: string
  /** Reference IDs this marker cites */
  referenceIds: string[]
  /** Confidence score (0-1) */
  confidence: number
  /** Source metadata for hover card */
  sources: CitationSource[]
}

/**
 * Interactive citation marker with hover card showing source metadata
 */
export function CitationMarker({
  marker,
  referenceIds,
  confidence,
  sources,
}: CitationMarkerProps) {
  // Find sources matching our reference IDs
  const matchingSources = sources.filter((s) => referenceIds.includes(s.reference_id))

  // Confidence badge color based on score
  const confidenceColor =
    confidence >= 0.8
      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      : confidence >= 0.6
        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'

  return (
    <HoverCard openDelay={200} closeDelay={100}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center text-primary hover:text-primary/80 hover:underline cursor-pointer font-medium text-sm mx-0.5 focus:outline-none focus:ring-2 focus:ring-primary/20 rounded"
        >
          {marker}
        </button>
      </HoverCardTrigger>
      <HoverCardContent className="w-80" side="top" align="center">
        <div className="space-y-3">
          {matchingSources.map((source) => (
            <div key={source.reference_id} className="space-y-2">
              {/* Document title */}
              <div className="flex items-start gap-2">
                <FileTextIcon className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" />
                <h4 className="font-semibold text-sm leading-tight">
                  {source.document_title || 'Untitled Document'}
                </h4>
              </div>

              {/* Section title */}
              {source.section_title && (
                <p className="text-xs text-muted-foreground pl-6">
                  Section: {source.section_title}
                </p>
              )}

              {/* Page range */}
              {source.page_range && (
                <p className="text-xs text-muted-foreground pl-6">
                  Pages: {source.page_range}
                </p>
              )}

              {/* Excerpt */}
              {source.excerpt && (
                <blockquote className="pl-6 border-l-2 border-muted text-xs italic text-muted-foreground line-clamp-3">
                  "{source.excerpt}"
                </blockquote>
              )}

              {/* File path */}
              <p className="text-xs text-muted-foreground/70 pl-6 truncate" title={source.file_path}>
                {source.file_path}
              </p>
            </div>
          ))}

          {/* Confidence badge */}
          <div className="flex items-center justify-between pt-2 border-t">
            <span className="text-xs text-muted-foreground">Match confidence</span>
            <Badge variant="outline" className={confidenceColor}>
              {(confidence * 100).toFixed(0)}%
            </Badge>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}

/**
 * Parses text containing citation markers and returns React elements
 * with interactive CitationMarker components.
 *
 * @param text - Text that may contain [n] or [n,m] patterns
 * @param sources - Array of citation sources for hover card metadata
 * @param markers - Array of citation markers with position and confidence data
 * @returns Array of React elements (strings and CitationMarker components)
 */
export function renderTextWithCitations(
  text: string,
  sources: CitationSource[],
  markers: Array<{ marker: string; reference_ids: string[]; confidence: number }>
): React.ReactNode[] {
  // Match citation patterns like [1], [2], [1,2], etc.
  const citationPattern = /\[(\d+(?:,\d+)*)\]/g
  const parts: React.ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = citationPattern.exec(text)) !== null) {
    // Add text before the citation
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }

    // Parse reference IDs from the marker
    const markerText = match[0]
    const refIds = match[1].split(',').map((id) => id.trim())

    // Find matching marker data for confidence
    const markerData = markers.find((m) => m.marker === markerText)
    const confidence = markerData?.confidence ?? 0.5

    // Add the citation marker component
    parts.push(
      <CitationMarker
        key={`citation-${match.index}`}
        marker={markerText}
        referenceIds={refIds}
        confidence={confidence}
        sources={sources}
      />
    )

    lastIndex = match.index + match[0].length
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return parts
}
