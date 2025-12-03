import type { CitationsMetadata, Message } from '@/api/lightrag'
import useTheme from '@/hooks/useTheme'
import { cn } from '@/lib/utils'
import { type ReactNode, memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { remarkFootnotes } from '@/utils/remarkFootnotes'
import mermaid from 'mermaid'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import rehypeReact from 'rehype-react'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/cjs/styles/prism'

import { BrainIcon, ChevronDownIcon, LoaderIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { CitationMarker } from './CitationMarker'

// KaTeX configuration options interface
interface KaTeXOptions {
  errorColor?: string
  throwOnError?: boolean
  displayMode?: boolean
  strict?: boolean
  trust?: boolean
  errorCallback?: (error: string, latex: string) => void
}

export type MessageWithError = Message & {
  id: string // Unique identifier for stable React keys
  isError?: boolean
  isThinking?: boolean // Flag to indicate if the message is in a "thinking" state
  /**
   * Indicates if the mermaid diagram in this message has been rendered.
   * Used to persist the rendering state across updates and prevent flickering.
   */
  mermaidRendered?: boolean
  /**
   * Indicates if the LaTeX formulas in this message are complete and ready for rendering.
   * Used to prevent red error text during streaming of incomplete LaTeX formulas.
   */
  latexRendered?: boolean
}

/**
 * Helper component to render text with citation markers as interactive HoverCards.
 * Parses [n] and [n,m] patterns and replaces them with CitationMarker components.
 */
function TextWithCitations({
  children,
  citationsMetadata,
}: {
  children: ReactNode
  citationsMetadata?: CitationsMetadata
}) {
  // If no citation metadata or children is not a string, render as-is
  if (!citationsMetadata || typeof children !== 'string') {
    return <>{children}</>
  }

  const text = children
  // Match citation patterns like [1], [2], [1,2], etc.
  const citationPattern = /\[(\d+(?:,\d+)*)\]/g
  const parts: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  let keyIndex = 0

  while ((match = citationPattern.exec(text)) !== null) {
    // Add text before the citation
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }

    // Parse reference IDs from the marker
    const markerText = match[0]
    const refIds = match[1].split(',').map((id) => id.trim())

    // Find matching marker data for confidence
    const markerData = citationsMetadata.markers?.find((m) => m.marker === markerText)
    const confidence = markerData?.confidence ?? 0.5

    // Add the citation marker component
    parts.push(
      <CitationMarker
        key={`citation-${keyIndex++}`}
        marker={markerText}
        referenceIds={refIds}
        confidence={confidence}
        sources={citationsMetadata.sources || []}
      />
    )

    lastIndex = match.index + match[0].length
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  // If no citations found, return original text
  if (parts.length === 0) {
    return <>{children}</>
  }

  return <>{parts}</>
}

// Restore original component definition and export
export const ChatMessage = ({
  message,
  isTabActive = true,
}: {
  message: MessageWithError
  isTabActive?: boolean
}) => {
  const { t } = useTranslation()
  const { theme } = useTheme()
  const [katexPlugin, setKatexPlugin] = useState<((options?: KaTeXOptions) => any) | null>(null)
  const [isThinkingExpanded, setIsThinkingExpanded] = useState<boolean>(false)

  // Directly use props passed from the parent.
  const { thinkingContent, displayContent, thinkingTime, isThinking } = message

  // Reset expansion state when new thinking starts
  useEffect(() => {
    if (isThinking) {
      // When thinking starts, always reset to collapsed state
      setIsThinkingExpanded(false)
    }
  }, [isThinking, message.id])

  // The content to display is now non-ambiguous.
  const finalThinkingContent = thinkingContent
  // For user messages, displayContent will be undefined, so we fall back to content.
  // For assistant messages, we prefer displayContent but fallback to content for backward compatibility
  const finalDisplayContent =
    message.role === 'user'
      ? message.content
      : displayContent !== undefined
        ? displayContent
        : message.content || ''

  // Load KaTeX rehype plugin dynamically
  // Note: KaTeX extensions (mhchem, copy-tex) are imported statically in main.tsx
  useEffect(() => {
    const loadKaTeX = async () => {
      try {
        const { default: rehypeKatex } = await import('rehype-katex')
        setKatexPlugin(() => rehypeKatex)
      } catch (error) {
        console.error('Failed to load KaTeX plugin:', error)
        setKatexPlugin(null)
      }
    }

    loadKaTeX()
  }, [])

  // Get citationsMetadata from message for use in markdown components
  const citationsMetadata = message.citationsMetadata

  const mainMarkdownComponents = useMemo(
    () => ({
      code: (props: any) => {
        const { inline, className, children, ...restProps } = props
        const match = /language-(\w+)/.exec(className || '')
        const language = match ? match[1] : undefined

        // Handle math blocks ($$...$$) - provide better container and styling
        if (language === 'math' && !inline) {
          return (
            <div className="katex-display-wrapper my-4 overflow-x-auto">
              <div className="text-current">{children}</div>
            </div>
          )
        }

        // Handle inline math ($...$) - ensure proper inline display
        if (language === 'math' && inline) {
          return (
            <span className="katex-inline-wrapper">
              <span className="text-current">{children}</span>
            </span>
          )
        }

        // Handle all other code (inline and block)
        return (
          <CodeHighlight
            inline={inline}
            className={className}
            {...restProps}
            renderAsDiagram={message.mermaidRendered ?? false}
            messageRole={message.role}
          >
            {children}
          </CodeHighlight>
        )
      },
      // Custom text renderer that handles citation markers [n]
      // Transforms plain text [1], [2], [1,2] into interactive CitationMarker components
      text: ({ children }: { children?: ReactNode }) => (
        <TextWithCitations citationsMetadata={citationsMetadata}>{children}</TextWithCitations>
      ),
      p: ({ children }: { children?: ReactNode }) => <div className="my-2">{children}</div>,
      h1: ({ children }: { children?: ReactNode }) => (
        <h1 className="text-xl font-bold mt-4 mb-2">{children}</h1>
      ),
      h2: ({ children }: { children?: ReactNode }) => (
        <h2 className="text-lg font-bold mt-4 mb-2">{children}</h2>
      ),
      h3: ({ children }: { children?: ReactNode }) => (
        <h3 className="text-base font-bold mt-3 mb-2">{children}</h3>
      ),
      h4: ({ children }: { children?: ReactNode }) => (
        <h4 className="text-base font-semibold mt-3 mb-2">{children}</h4>
      ),
      ul: ({ children }: { children?: ReactNode }) => (
        <ul className="list-disc pl-5 my-2">{children}</ul>
      ),
      ol: ({ children }: { children?: ReactNode }) => (
        <ol className="list-decimal pl-5 my-2">{children}</ol>
      ),
      li: ({ children }: { children?: ReactNode }) => <li className="my-1">{children}</li>,
    }),
    [message.mermaidRendered, message.role, citationsMetadata]
  )

  const thinkingMarkdownComponents = useMemo(
    () => ({
      code: (props: any) => (
        <CodeHighlight
          {...props}
          renderAsDiagram={message.mermaidRendered ?? false}
          messageRole={message.role}
        />
      ),
    }),
    [message.mermaidRendered, message.role]
  )

  return (
    <div
      className={`${
        message.role === 'user'
          ? 'max-w-[80%] bg-primary text-primary-foreground'
          : message.isError
            ? 'w-[95%] bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400'
            : 'w-[95%] bg-muted'
      } rounded-lg px-4 py-2`}
    >
      {/* Thinking Pill - collapsible bubble UI */}
      {message.role === 'assistant' && (isThinking || thinkingTime !== null) && (
        <div className={cn('mb-3', !isTabActive && 'opacity-50')}>
          {/* Pill Header - always visible */}
          <button
            type="button"
            onClick={() => {
              if (finalThinkingContent && finalThinkingContent.trim() !== '') {
                setIsThinkingExpanded(!isThinkingExpanded)
              }
            }}
            className={cn(
              'inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all',
              'border shadow-sm select-none',
              isThinking
                ? 'bg-amber-50 border-amber-200 text-amber-700 dark:bg-amber-950/50 dark:border-amber-800 dark:text-amber-300'
                : 'bg-slate-100 border-slate-200 text-slate-600 dark:bg-slate-800 dark:border-slate-700 dark:text-slate-300',
              finalThinkingContent?.trim() && 'cursor-pointer hover:shadow-md'
            )}
          >
            {isThinking ? (
              <>
                {isTabActive && (
                  <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse" />
                )}
                <span>{t('retrievePanel.chatMessage.thinking')}</span>
              </>
            ) : (
              typeof thinkingTime === 'number' && (
                <>
                  <BrainIcon className="w-3.5 h-3.5" />
                  <span>{t('retrievePanel.chatMessage.thinkingTime', { time: thinkingTime })}</span>
                </>
              )
            )}
            {finalThinkingContent && finalThinkingContent.trim() !== '' && (
              <ChevronDownIcon
                className={cn(
                  'w-3.5 h-3.5 transition-transform',
                  isThinkingExpanded && 'rotate-180'
                )}
              />
            )}
          </button>

          {/* Expandable Content */}
          {isThinkingExpanded && finalThinkingContent && finalThinkingContent.trim() !== '' && (
            <div
              className={cn(
                'mt-2 ml-2 p-3 rounded-lg text-sm',
                'bg-slate-50 border border-slate-200 dark:bg-slate-900 dark:border-slate-700',
                'max-h-[400px] overflow-y-auto',
                'prose dark:prose-invert max-w-none break-words prose-p:my-1 prose-headings:my-2',
                '[&_sup]:text-[0.75em] [&_sup]:align-[0.1em] [&_sup]:leading-[0]',
                '[&_sub]:text-[0.75em] [&_sub]:align-[-0.2em] [&_sub]:leading-[0]',
                '[&_mark]:bg-yellow-200 [&_mark]:dark:bg-yellow-800',
                'text-foreground'
              )}
            >
              {isThinking && (
                <div className="mb-2 text-xs text-amber-600 dark:text-amber-400 italic">
                  {t('retrievePanel.chatMessage.thinkingInProgress', 'Thinking in progress...')}
                </div>
              )}
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkFootnotes, remarkMath]}
                rehypePlugins={[
                  rehypeRaw,
                  ...(katexPlugin && (message.latexRendered ?? true)
                    ? [
                        [
                          katexPlugin,
                          {
                            errorColor: theme === 'dark' ? '#ef4444' : '#dc2626',
                            throwOnError: false,
                            displayMode: false,
                            strict: false,
                            trust: true,
                            errorCallback: (error: string, latex: string) => {
                              if (process.env.NODE_ENV === 'development') {
                                console.warn('KaTeX error in thinking:', error, latex)
                              }
                            },
                          },
                        ] as any,
                      ]
                    : []),
                  rehypeReact,
                ]}
                skipHtml={false}
                components={thinkingMarkdownComponents}
              >
                {finalThinkingContent}
              </ReactMarkdown>
            </div>
          )}
        </div>
      )}
      {/* Main content display */}
      {finalDisplayContent && (
        <div className="relative">
          <ReactMarkdown
            className={`prose dark:prose-invert max-w-none text-sm break-words prose-headings:mt-4 prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-1 [&_.katex]:text-current [&_.katex-display]:my-4 [&_.katex-display]:max-w-full [&_.katex-display_>.base]:overflow-x-auto [&_sup]:text-[0.75em] [&_sup]:align-[0.1em] [&_sup]:leading-[0] [&_sub]:text-[0.75em] [&_sub]:align-[-0.2em] [&_sub]:leading-[0] [&_mark]:bg-yellow-200 [&_mark]:dark:bg-yellow-800 [&_u]:underline [&_del]:line-through [&_ins]:underline [&_ins]:decoration-green-500 [&_.footnotes]:mt-8 [&_.footnotes]:pt-4 [&_.footnotes]:border-t [&_.footnotes_ol]:text-sm [&_.footnotes_li]:my-1 ${
              message.role === 'user' ? 'text-primary-foreground' : 'text-foreground'
            } ${
              message.role === 'user'
                ? '[&_.footnotes]:border-primary-foreground/30 [&_a[href^="#fn"]]:text-primary-foreground [&_a[href^="#fn"]]:no-underline [&_a[href^="#fn"]]:hover:underline [&_a[href^="#fnref"]]:text-primary-foreground [&_a[href^="#fnref"]]:no-underline [&_a[href^="#fnref"]]:hover:underline'
                : '[&_.footnotes]:border-border [&_a[href^="#fn"]]:text-primary [&_a[href^="#fn"]]:no-underline [&_a[href^="#fn"]]:hover:underline [&_a[href^="#fnref"]]:text-primary [&_a[href^="#fnref"]]:no-underline [&_a[href^="#fnref"]]:hover:underline'
            }`}
            remarkPlugins={[remarkGfm, remarkFootnotes, remarkMath]}
            rehypePlugins={[
              rehypeRaw,
              ...(katexPlugin && (message.latexRendered ?? true)
                ? [
                    [
                      katexPlugin,
                      {
                        errorColor: theme === 'dark' ? '#ef4444' : '#dc2626',
                        throwOnError: false,
                        displayMode: false,
                        strict: false,
                        trust: true,
                        // Add silent error handling to avoid console noise
                        errorCallback: (error: string, latex: string) => {
                          // Only show detailed errors in development environment
                          if (process.env.NODE_ENV === 'development') {
                            console.warn(
                              'KaTeX rendering error in main content:',
                              error,
                              'for LaTeX:',
                              latex
                            )
                          }
                        },
                      },
                    ] as any,
                  ]
                : []),
              rehypeReact,
            ]}
            skipHtml={false}
            components={mainMarkdownComponents}
          >
            {finalDisplayContent}
          </ReactMarkdown>
        </div>
      )}
      {/* Loading indicator - only show in active tab */}
      {isTabActive &&
        (() => {
          // More comprehensive loading state check
          const hasVisibleContent = finalDisplayContent && finalDisplayContent.trim() !== ''
          const isLoadingState = !hasVisibleContent && !isThinking && !thinkingTime
          return isLoadingState && <LoaderIcon className="animate-spin duration-2000" />
        })()}
    </div>
  )
}

// Remove the incorrect memo export line

interface CodeHighlightProps {
  inline?: boolean
  className?: string
  children?: ReactNode
  renderAsDiagram?: boolean // Flag to indicate if rendering as diagram should be attempted
  messageRole?: 'user' | 'assistant' // Message role for context-aware styling
}

// Check if it is a large JSON
const isLargeJson = (language: string | undefined, content: string | undefined): boolean => {
  if (!content || language !== 'json') return false
  return content.length > 5000 // JSON larger than 5KB is considered large JSON
}

// Memoize the CodeHighlight component
const CodeHighlight = memo(
  ({
    inline,
    className,
    children,
    renderAsDiagram = false,
    messageRole,
    ...props
  }: CodeHighlightProps) => {
    const { theme } = useTheme()
    const [hasRendered, setHasRendered] = useState(false) // State to track successful render
    const match = className?.match(/language-(\w+)/)
    const language = match ? match[1] : undefined
    const mermaidRef = useRef<HTMLDivElement>(null)
    const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null) // Use ReturnType for better typing

    // Get the content string, check if it is a large JSON
    const contentStr = String(children || '').replace(/\n$/, '')
    const isLargeJsonBlock = isLargeJson(language, contentStr)

    // Handle Mermaid rendering with debounce
    useEffect(() => {
      // Effect should run when renderAsDiagram becomes true or hasRendered changes.
      // The actual rendering logic inside checks language and hasRendered state.
      if (renderAsDiagram && !hasRendered && language === 'mermaid' && mermaidRef.current) {
        const container = mermaidRef.current // Capture ref value

        // Clear previous timer if dependencies change before timeout (e.g., renderAsDiagram flips quickly)
        if (debounceTimerRef.current) {
          clearTimeout(debounceTimerRef.current)
        }

        debounceTimerRef.current = setTimeout(() => {
          if (!container) return // Container might have unmounted

          // Double check hasRendered state inside timeout, in case it changed rapidly
          if (hasRendered) return

          try {
            // Initialize mermaid config
            mermaid.initialize({
              startOnLoad: false,
              theme: theme === 'dark' ? 'dark' : 'default',
              securityLevel: 'loose',
              suppressErrorRendering: true,
            })

            // Show loading indicator
            container.innerHTML =
              '<div class="flex justify-center items-center p-4"><svg class="animate-spin h-5 w-5 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg></div>'

            // Preprocess mermaid content
            const rawContent = String(children).replace(/\n$/, '').trim()

            // Heuristic check for potentially complete graph definition
            const looksPotentiallyComplete =
              rawContent.length > 10 &&
              (rawContent.startsWith('graph') ||
                rawContent.startsWith('sequenceDiagram') ||
                rawContent.startsWith('classDiagram') ||
                rawContent.startsWith('stateDiagram') ||
                rawContent.startsWith('gantt') ||
                rawContent.startsWith('pie') ||
                rawContent.startsWith('flowchart') ||
                rawContent.startsWith('erDiagram'))

            if (!looksPotentiallyComplete) {
              console.log(
                'Mermaid content might be incomplete, skipping render attempt:',
                rawContent
              )
              // Optionally keep loading indicator or show a message
              // container.innerHTML = '<p class="text-sm text-muted-foreground">Waiting for complete diagram...</p>';
              return
            }

            const processedContent = rawContent
              .split('\n')
              .map((line) => {
                const trimmedLine = line.trim()
                if (trimmedLine.startsWith('subgraph')) {
                  const parts = trimmedLine.split(' ')
                  if (parts.length > 1) {
                    const title = parts.slice(1).join(' ').replace(/["']/g, '')
                    return `subgraph "${title}"`
                  }
                }
                return trimmedLine
              })
              .filter((line) => !line.trim().startsWith('linkStyle'))
              .join('\n')

            const mermaidId = `mermaid-${Date.now()}`
            mermaid
              .render(mermaidId, processedContent)
              .then(({ svg, bindFunctions }) => {
                // Check ref and hasRendered state again inside async callback
                if (mermaidRef.current === container && !hasRendered) {
                  container.innerHTML = svg
                  setHasRendered(true) // Mark as rendered successfully
                  if (bindFunctions) {
                    try {
                      bindFunctions(container)
                    } catch (bindError) {
                      console.error('Mermaid bindFunctions error:', bindError)
                      container.innerHTML +=
                        '<p class="text-orange-500 text-xs">Diagram interactions might be limited.</p>'
                    }
                  }
                } else if (mermaidRef.current !== container) {
                  console.log('Mermaid container changed before rendering completed.')
                }
              })
              .catch((error) => {
                console.error('Mermaid rendering promise error (debounced):', error)
                console.error('Failed content (debounced):', processedContent)
                if (mermaidRef.current === container) {
                  const errorMessage = error instanceof Error ? error.message : String(error)
                  const errorPre = document.createElement('pre')
                  errorPre.className = 'text-red-500 text-xs whitespace-pre-wrap break-words'
                  errorPre.textContent = `Mermaid diagram error: ${errorMessage}\n\nContent:\n${processedContent}`
                  container.innerHTML = ''
                  container.appendChild(errorPre)
                }
              })
          } catch (error) {
            console.error('Mermaid synchronous error (debounced):', error)
            console.error('Failed content (debounced):', String(children))
            if (mermaidRef.current === container) {
              const errorMessage = error instanceof Error ? error.message : String(error)
              const errorPre = document.createElement('pre')
              errorPre.className = 'text-red-500 text-xs whitespace-pre-wrap break-words'
              errorPre.textContent = `Mermaid diagram setup error: ${errorMessage}`
              container.innerHTML = ''
              container.appendChild(errorPre)
            }
          }
        }, 300) // Debounce delay
      }

      // Cleanup function to clear the timer on unmount or before re-running effect
      return () => {
        if (debounceTimerRef.current) {
          clearTimeout(debounceTimerRef.current)
        }
      }
      // Dependencies: renderAsDiagram ensures effect runs when diagram should be shown.
      // Dependencies include all values used inside the effect to satisfy exhaustive-deps.
      // The !hasRendered check prevents re-execution of render logic after success.
    }, [renderAsDiagram, hasRendered, language, children, theme]) // Add children and theme back

    // For large JSON, skip syntax highlighting completely and use a simple pre tag
    if (isLargeJsonBlock) {
      return (
        <pre className="whitespace-pre-wrap break-words bg-muted p-4 rounded-md overflow-x-auto text-sm font-mono">
          {contentStr}
        </pre>
      )
    }

    // Render based on language type
    // If it's a mermaid language block and rendering as diagram is not requested (e.g., incomplete stream), display as plain text
    if (language === 'mermaid' && !renderAsDiagram) {
      return (
        <SyntaxHighlighter
          style={theme === 'dark' ? oneDark : oneLight}
          PreTag="div"
          language="text" // Use text as language to avoid syntax highlighting errors
          {...props}
        >
          {contentStr}
        </SyntaxHighlighter>
      )
    }

    // If it's a mermaid language block and the message is complete, render as diagram
    if (language === 'mermaid') {
      // Container for Mermaid diagram
      return <div className="mermaid-diagram-container my-4 overflow-x-auto" ref={mermaidRef}></div>
    }

    // ReactMarkdown determines inline vs block based on markdown syntax
    // Inline code: `code` (no className with language)
    // Block code: ```language (has className like "language-js")
    // If there's no language className and no explicit inline prop, it's likely inline code
    const isInline = inline ?? !className?.startsWith('language-')

    // Generate dynamic inline code styles based on message role and theme
    const getInlineCodeStyles = () => {
      if (messageRole === 'user') {
        // User messages have dark background (bg-primary), need light inline code
        return theme === 'dark'
          ? 'bg-primary-foreground/20 text-primary-foreground border border-primary-foreground/30'
          : 'bg-primary-foreground/20 text-primary-foreground border border-primary-foreground/30'
      } else {
        // Assistant messages have light background (bg-muted), need contrasting inline code
        return theme === 'dark'
          ? 'bg-muted-foreground/20 text-muted-foreground border border-muted-foreground/30'
          : 'bg-slate-200 text-slate-800 border border-slate-300'
      }
    }

    // Handle non-Mermaid code blocks
    return !isInline ? (
      <SyntaxHighlighter
        style={theme === 'dark' ? oneDark : oneLight}
        PreTag="div"
        language={language}
        {...props}
      >
        {contentStr}
      </SyntaxHighlighter>
    ) : (
      // Handle inline code with context-aware styling
      <code
        className={cn(
          className,
          'mx-1 rounded-sm px-1 py-0.5 font-mono text-sm',
          getInlineCodeStyles()
        )}
        {...props}
      >
        {children}
      </code>
    )
  }
)

// Assign display name for React DevTools
CodeHighlight.displayName = 'CodeHighlight'
