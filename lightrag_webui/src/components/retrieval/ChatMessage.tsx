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

import type { Element } from 'hast'

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'

import { LoaderIcon, CopyIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export type MessageWithError = Message & {
  id: string // Unique identifier for stable React keys
  isError?: boolean
  /**
   * Indicates if the mermaid diagram in this message has been rendered.
   * Used to persist the rendering state across updates and prevent flickering.
   */
  mermaidRendered?: boolean
}

// Restore original component definition and export
export const ChatMessage = ({ message }: { message: MessageWithError }) => { // Remove isComplete prop
  const { t } = useTranslation()
  const handleCopyMarkdown = useCallback(async () => {
    if (message.content) {
      try {
        await navigator.clipboard.writeText(message.content)
      } catch (err) {
        console.error(t('chat.copyError'), err)
      }
    }
  }, [message, t]) // Added t to dependency array

  return (
    <div
      className={`${
        message.role === 'user'
          ? 'max-w-[80%] bg-primary text-primary-foreground'
          : message.isError
            ? 'w-[90%] bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400'
            : 'w-[90%] bg-muted'
      } rounded-lg px-4 py-2`}
    >
      <div className="relative">
        <ReactMarkdown
          className="prose dark:prose-invert max-w-none text-sm break-words prose-headings:mt-4 prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-1"
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeReact]}
          skipHtml={false}
          // Memoize the components object to prevent unnecessary re-renders of ReactMarkdown children
          components={useMemo(() => ({
            code: (props: any) => ( // Add type annotation if needed, e.g., props: CodeProps from 'react-markdown/lib/ast-to-react'
              <CodeHighlight
                {...props}
                renderAsDiagram={message.mermaidRendered ?? false}
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
          }), [message.mermaidRendered])} // Dependency ensures update if mermaid state changes
        >
          {message.content}
        </ReactMarkdown>
        {message.role === 'assistant' && message.content && message.content.length > 0 && ( // Added check for message.content existence
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
      {message.content === '' && <LoaderIcon className="animate-spin duration-2000" />} {/* Check for empty string specifically */}
    </div>
  )
}

// Remove the incorrect memo export line

interface CodeHighlightProps {
  inline?: boolean
  className?: string
  children?: ReactNode
  node?: Element // Keep node for inline check
  renderAsDiagram?: boolean // Flag to indicate if rendering as diagram should be attempted
}

// Helper function remains the same
const isInlineCode = (node?: Element): boolean => {
  if (!node || !node.children) return false;
  const textContent = node.children
    .filter((child) => child.type === 'text')
    .map((child) => (child as any).value)
    .join('');
  // Consider inline if it doesn't contain newline or is very short
  return !textContent.includes('\n') || textContent.length < 40;
};


// Memoize the CodeHighlight component
const CodeHighlight = memo(({ className, children, node, renderAsDiagram = false, ...props }: CodeHighlightProps) => {
  const { theme } = useTheme();
  const [hasRendered, setHasRendered] = useState(false); // State to track successful render
  const match = className?.match(/language-(\w+)/);
  const language = match ? match[1] : undefined;
  const inline = isInlineCode(node); // Use the helper function
  const mermaidRef = useRef<HTMLDivElement>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null); // Use ReturnType for better typing

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
  // children, language, theme trigger re-render if code/context changes.
  // Dependencies include all values used inside the effect to satisfy exhaustive-deps.
  // The !hasRendered check prevents re-execution of render logic after success.
  }, [renderAsDiagram, hasRendered, language, children, theme]); // Add children and theme back

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
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    );
  }

  // If it's a mermaid language block and the message is complete, render as diagram
  if (language === 'mermaid') {
    // Container for Mermaid diagram
    return <div className="mermaid-diagram-container my-4 overflow-x-auto" ref={mermaidRef}></div>;
  }

  // Handle non-Mermaid code blocks
  return !inline ? (
    <SyntaxHighlighter
      style={theme === 'dark' ? oneDark : oneLight}
      PreTag="div" // Use div for block code
      language={language}
      {...props}
    >
      {String(children).replace(/\n$/, '')}
    </SyntaxHighlighter>
  ) : (
    // Handle inline code
    <code
      className={cn(className, 'mx-1 rounded-sm bg-muted px-1 py-0.5 font-mono text-sm')} // 添加 font-mono 确保使用等宽字体
      {...props}
    >
      {children}
    </code>
  );
});

// Assign display name for React DevTools
CodeHighlight.displayName = 'CodeHighlight';
