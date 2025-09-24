import { ReactNode, useCallback, useEffect, useMemo, useRef, memo, useState } from 'react' // Import useMemo
import { Message } from '@/api/lightrag'
import useTheme from '@/hooks/useTheme'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeReact from 'rehype-react'
import remarkMath from 'remark-math'
import mermaid from 'mermaid'


import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'

import { LoaderIcon, CopyIcon, ChevronDownIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export type MessageWithError = Message & {
  id: string // Unique identifier for stable React keys
  isError?: boolean
  isThinking?: boolean // Flag to indicate if the message is in a "thinking" state
  /**
   * Indicates if the mermaid diagram in this message has been rendered.
   * Used to persist the rendering state across updates and prevent flickering.
   */
  mermaidRendered?: boolean
}

// Restore original component definition and export
export const ChatMessage = ({ message }: { message: MessageWithError }) => { // Remove isComplete prop
  const { t } = useTranslation()
  const { theme } = useTheme()
  const [katexPlugin, setKatexPlugin] = useState<any>(null)
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
  const finalDisplayContent = message.role === 'user'
    ? message.content
    : (displayContent !== undefined ? displayContent : (message.content || ''))

  // Load KaTeX dynamically
  useEffect(() => {
    const loadKaTeX = async () => {
      try {
        const [{ default: rehypeKatex }] = await Promise.all([
          import('rehype-katex'),
          import('katex/dist/katex.min.css')
        ])
        setKatexPlugin(() => rehypeKatex)
      } catch (error) {
        console.error('Failed to load KaTeX:', error)
      }
    }
    loadKaTeX()
  }, [])
  const handleCopyMarkdown = useCallback(async () => {
    if (message.content) {
      try {
        await navigator.clipboard.writeText(message.content)
      } catch (err) {
        console.error(t('chat.copyError'), err)
      }
    }
  }, [message, t]) // Added t to dependency array

  const mainMarkdownComponents = useMemo(() => ({
    code: (props: any) => (
      <CodeHighlight
        {...props}
        renderAsDiagram={message.mermaidRendered ?? false}
        messageRole={message.role}
      />
    ),
    p: ({ children }: { children?: ReactNode }) => <p className="my-2">{children}</p>,
    h1: ({ children }: { children?: ReactNode }) => <h1 className="text-xl font-bold mt-4 mb-2">{children}</h1>,
    h2: ({ children }: { children?: ReactNode }) => <h2 className="text-lg font-bold mt-4 mb-2">{children}</h2>,
    h3: ({ children }: { children?: ReactNode }) => <h3 className="text-base font-bold mt-3 mb-2">{children}</h3>,
    h4: ({ children }: { children?: ReactNode }) => <h4 className="text-base font-semibold mt-3 mb-2">{children}</h4>,
    ul: ({ children }: { children?: ReactNode }) => <ul className="list-disc pl-5 my-2">{children}</ul>,
    ol: ({ children }: { children?: ReactNode }) => <ol className="list-decimal pl-5 my-2">{children}</ol>,
    li: ({ children }: { children?: ReactNode }) => <li className="my-1">{children}</li>
  }), [message.mermaidRendered, message.role]);

  const thinkingMarkdownComponents = useMemo(() => ({
    code: (props: any) => (<CodeHighlight {...props} renderAsDiagram={message.mermaidRendered ?? false} messageRole={message.role} />)
  }), [message.mermaidRendered, message.role]);

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
      {/* Thinking process display - only for assistant messages */}
      {message.role === 'assistant' && (isThinking || thinkingTime !== null) && (
        <div className="mb-2">
          <div
            className="flex items-center text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors duration-200 text-sm cursor-pointer select-none"
            onClick={() => {
              // Allow expansion when there's thinking content, even during thinking process
              if (finalThinkingContent && finalThinkingContent.trim() !== '') {
                setIsThinkingExpanded(!isThinkingExpanded)
              }
            }}
          >
            {isThinking ? (
              <>
                <LoaderIcon className="mr-2 size-4 animate-spin" />
                <span>{t('retrievePanel.chatMessage.thinking')}</span>
              </>
            ) : (
              typeof thinkingTime === 'number' && <span>{t('retrievePanel.chatMessage.thinkingTime', { time: thinkingTime })}</span>
            )}
            {/* Show chevron when there's thinking content, even during thinking process */}
            {finalThinkingContent && finalThinkingContent.trim() !== '' && <ChevronDownIcon className={`ml-2 size-4 shrink-0 transition-transform ${isThinkingExpanded ? 'rotate-180' : ''}`} />}
          </div>
          {/* Show thinking content when expanded and content exists, even during thinking process */}
          {isThinkingExpanded && finalThinkingContent && finalThinkingContent.trim() !== '' && (
            <div className="mt-2 pl-4 border-l-2 border-primary/20 text-sm prose dark:prose-invert max-w-none break-words prose-p:my-1 prose-headings:my-2">
              {isThinking && (
                <div className="mb-2 text-xs text-gray-400 dark:text-gray-500 italic">
                  {t('retrievePanel.chatMessage.thinkingInProgress', 'Thinking in progress...')}
                </div>
              )}
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[
                  ...(katexPlugin ? [[katexPlugin, { errorColor: theme === 'dark' ? '#ef4444' : '#dc2626', throwOnError: false, displayMode: false }] as any] : []),
                  rehypeReact
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
            className="prose dark:prose-invert max-w-none text-sm break-words prose-headings:mt-4 prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-1 [&_.katex]:text-current [&_.katex-display]:my-4 [&_.katex-display]:overflow-x-auto"
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[
              ...(katexPlugin ? [[
                katexPlugin,
                {
                  errorColor: theme === 'dark' ? '#ef4444' : '#dc2626',
                  throwOnError: false,
                  displayMode: false
                }
              ] as any] : []),
              rehypeReact
            ]}
            skipHtml={false}
            components={mainMarkdownComponents}
          >
            {finalDisplayContent}
          </ReactMarkdown>
          {message.role === 'assistant' && finalDisplayContent && finalDisplayContent.length > 0 && (
            <Button
              onClick={handleCopyMarkdown}
              className="absolute right-0 bottom-0 size-6 rounded-md opacity-20 transition-opacity hover:opacity-100"
              tooltip={t('retrievePanel.chatMessage.copyTooltip')}
              variant="default"
              size="icon"
            >
              <CopyIcon className="size-4" /> {/* Explicit size */}
            </Button>
          )}
        </div>
      )}
      {(() => {
        // More comprehensive loading state check
        const hasVisibleContent = finalDisplayContent && finalDisplayContent.trim() !== '';
        const isLoadingState = !hasVisibleContent && !isThinking && !thinkingTime;
        return isLoadingState && <LoaderIcon className="animate-spin duration-2000" />;
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
  if (!content || language !== 'json') return false;
  return content.length > 5000; // JSON larger than 5KB is considered large JSON
};

// Memoize the CodeHighlight component
const CodeHighlight = memo(({ inline, className, children, renderAsDiagram = false, messageRole, ...props }: CodeHighlightProps) => {
  const { theme } = useTheme();
  const [hasRendered, setHasRendered] = useState(false); // State to track successful render
  const match = className?.match(/language-(\w+)/);
  const language = match ? match[1] : undefined;
  const mermaidRef = useRef<HTMLDivElement>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null); // Use ReturnType for better typing

  // Get the content string, check if it is a large JSON
  const contentStr = String(children || '').replace(/\n$/, '');
  const isLargeJsonBlock = isLargeJson(language, contentStr);

  // Handle Mermaid rendering with debounce
  useEffect(() => {
    // Effect should run when renderAsDiagram becomes true or hasRendered changes.
    // The actual rendering logic inside checks language and hasRendered state.
    if (renderAsDiagram && !hasRendered && language === 'mermaid' && mermaidRef.current) {
      const container = mermaidRef.current; // Capture ref value

      // Clear previous timer if dependencies change before timeout (e.g., renderAsDiagram flips quickly)
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }

      debounceTimerRef.current = setTimeout(() => {
        if (!container) return; // Container might have unmounted

        // Double check hasRendered state inside timeout, in case it changed rapidly
        if (hasRendered) return;

        try {
          // Initialize mermaid config
          mermaid.initialize({
            startOnLoad: false,
            theme: theme === 'dark' ? 'dark' : 'default',
            securityLevel: 'loose',
            suppressErrorRendering: true,
          });

          // Show loading indicator
          container.innerHTML = '<div class="flex justify-center items-center p-4"><svg class="animate-spin h-5 w-5 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg></div>';

          // Preprocess mermaid content
          const rawContent = String(children).replace(/\n$/, '').trim();

          // Heuristic check for potentially complete graph definition
          const looksPotentiallyComplete = rawContent.length > 10 && (
            rawContent.startsWith('graph') ||
            rawContent.startsWith('sequenceDiagram') ||
            rawContent.startsWith('classDiagram') ||
            rawContent.startsWith('stateDiagram') ||
            rawContent.startsWith('gantt') ||
            rawContent.startsWith('pie') ||
            rawContent.startsWith('flowchart') ||
            rawContent.startsWith('erDiagram')
          );

          if (!looksPotentiallyComplete) {
            console.log('Mermaid content might be incomplete, skipping render attempt:', rawContent);
            // Optionally keep loading indicator or show a message
            // container.innerHTML = '<p class="text-sm text-muted-foreground">Waiting for complete diagram...</p>';
            return;
          }

          const processedContent = rawContent
            .split('\n')
            .map(line => {
              const trimmedLine = line.trim();
              if (trimmedLine.startsWith('subgraph')) {
                const parts = trimmedLine.split(' ');
                if (parts.length > 1) {
                  const title = parts.slice(1).join(' ').replace(/["']/g, '');
                  return `subgraph "${title}"`;
                }
              }
              return trimmedLine;
            })
            .filter(line => !line.trim().startsWith('linkStyle'))
            .join('\n');

          const mermaidId = `mermaid-${Date.now()}`;
          mermaid.render(mermaidId, processedContent)
            .then(({ svg, bindFunctions }) => {
              // Check ref and hasRendered state again inside async callback
              if (mermaidRef.current === container && !hasRendered) {
                container.innerHTML = svg;
                setHasRendered(true); // Mark as rendered successfully
                if (bindFunctions) {
                  try {
                    bindFunctions(container);
                  } catch (bindError) {
                    console.error('Mermaid bindFunctions error:', bindError);
                    container.innerHTML += '<p class="text-orange-500 text-xs">Diagram interactions might be limited.</p>';
                  }
                }
              } else if (mermaidRef.current !== container) {
                console.log('Mermaid container changed before rendering completed.');
              }
            })
            .catch(error => {
              console.error('Mermaid rendering promise error (debounced):', error);
              console.error('Failed content (debounced):', processedContent);
              if (mermaidRef.current === container) {
                const errorMessage = error instanceof Error ? error.message : String(error);
                const errorPre = document.createElement('pre');
                errorPre.className = 'text-red-500 text-xs whitespace-pre-wrap break-words';
                errorPre.textContent = `Mermaid diagram error: ${errorMessage}\n\nContent:\n${processedContent}`;
                container.innerHTML = '';
                container.appendChild(errorPre);
              }
            });

        } catch (error) {
          console.error('Mermaid synchronous error (debounced):', error);
          console.error('Failed content (debounced):', String(children));
          if (mermaidRef.current === container) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            const errorPre = document.createElement('pre');
            errorPre.className = 'text-red-500 text-xs whitespace-pre-wrap break-words';
            errorPre.textContent = `Mermaid diagram setup error: ${errorMessage}`;
            container.innerHTML = '';
            container.appendChild(errorPre);
          }
        }
      }, 300); // Debounce delay
    }

    // Cleanup function to clear the timer on unmount or before re-running effect
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  // Dependencies: renderAsDiagram ensures effect runs when diagram should be shown.
  // Dependencies include all values used inside the effect to satisfy exhaustive-deps.
  // The !hasRendered check prevents re-execution of render logic after success.
  }, [renderAsDiagram, hasRendered, language, children, theme]); // Add children and theme back

  // For large JSON, skip syntax highlighting completely and use a simple pre tag
  if (isLargeJsonBlock) {
    return (
      <pre className="whitespace-pre-wrap break-words bg-muted p-4 rounded-md overflow-x-auto text-sm font-mono">
        {contentStr}
      </pre>
    );
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
    );
  }

  // If it's a mermaid language block and the message is complete, render as diagram
  if (language === 'mermaid') {
    // Container for Mermaid diagram
    return <div className="mermaid-diagram-container my-4 overflow-x-auto" ref={mermaidRef}></div>;
  }


  // ReactMarkdown determines inline vs block based on markdown syntax
  // Inline code: `code` (no className with language)
  // Block code: ```language (has className like "language-js")
  // If there's no language className and no explicit inline prop, it's likely inline code
  const isInline = inline ?? !className?.startsWith('language-');

  // Generate dynamic inline code styles based on message role and theme
  const getInlineCodeStyles = () => {
    if (messageRole === 'user') {
      // User messages have dark background (bg-primary), need light inline code
      return theme === 'dark'
        ? 'bg-primary-foreground/20 text-primary-foreground border border-primary-foreground/30'
        : 'bg-primary-foreground/20 text-primary-foreground border border-primary-foreground/30';
    } else {
      // Assistant messages have light background (bg-muted), need contrasting inline code
      return theme === 'dark'
        ? 'bg-muted-foreground/20 text-muted-foreground border border-muted-foreground/30'
        : 'bg-slate-200 text-slate-800 border border-slate-300';
    }
  };

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
  );
});

// Assign display name for React DevTools
CodeHighlight.displayName = 'CodeHighlight';
