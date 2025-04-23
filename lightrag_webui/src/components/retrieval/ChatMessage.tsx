import { ReactNode, useCallback, useEffect, useRef } from 'react'
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
  isError?: boolean
}

export const ChatMessage = ({ message }: { message: MessageWithError }) => {
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
          components={{
            code: CodeHighlight,
            p: ({ children }) => <p className="my-2">{children}</p>,
            h1: ({ children }) => <h1 className="text-xl font-bold mt-4 mb-2">{children}</h1>,
            h2: ({ children }) => <h2 className="text-lg font-bold mt-4 mb-2">{children}</h2>,
            h3: ({ children }) => <h3 className="text-base font-bold mt-3 mb-2">{children}</h3>,
            h4: ({ children }) => <h4 className="text-base font-semibold mt-3 mb-2">{children}</h4>,
            ul: ({ children }) => <ul className="list-disc pl-5 my-2">{children}</ul>,
            ol: ({ children }) => <ol className="list-decimal pl-5 my-2">{children}</ol>,
            li: ({ children }) => <li className="my-1">{children}</li>
          }}
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

interface CodeHighlightProps {
  inline?: boolean
  className?: string
  children?: ReactNode
  node?: Element // Keep node for inline check
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


const CodeHighlight = ({ className, children, node, ...props }: CodeHighlightProps) => {
  const { theme } = useTheme();
  const match = className?.match(/language-(\w+)/);
  const language = match ? match[1] : undefined;
  const inline = isInlineCode(node); // Use the helper function
  const mermaidRef = useRef<HTMLDivElement>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null); // Use ReturnType for better typing

  // Handle Mermaid rendering with debounce
  useEffect(() => {
    // Clear any existing timer when dependencies change
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    if (language === 'mermaid' && mermaidRef.current) {
      const container = mermaidRef.current; // Capture ref value for use inside timeout/callbacks

      // Set a new timer to render after a short delay
      debounceTimerRef.current = setTimeout(() => {
        // Ensure container still exists when timer fires
        if (!container) return;

        try {
          // Initialize mermaid config (safe to call multiple times)
          mermaid.initialize({
            startOnLoad: false,
            theme: theme === 'dark' ? 'dark' : 'default',
            securityLevel: 'loose',
          });

          // Show loading indicator while processing
          container.innerHTML = '<div class="flex justify-center items-center p-4"><svg class="animate-spin h-5 w-5 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg></div>';

          // Preprocess mermaid content
          const rawContent = String(children).replace(/\n$/, '').trim(); // Trim whitespace as well

          // Heuristic check for potentially complete graph definition
          // Looks for graph type declaration and some content beyond it.
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
            // Keep loading indicator or show a message
            // container.innerHTML = '<p class="text-sm text-muted-foreground">Waiting for complete diagram...</p>';
            return; // Don't attempt to render potentially incomplete content
          }

          const processedContent = rawContent
            .split('\n')
            .map(line => {
              const trimmedLine = line.trim();
              // Keep subgraph processing
              if (trimmedLine.startsWith('subgraph')) {
                const parts = trimmedLine.split(' ');
                if (parts.length > 1) {
                  const title = parts.slice(1).join(' ').replace(/["']/g, '');
                  return `subgraph "${title}"`;
                }
              }
              return trimmedLine;
            })
            .filter(line => !line.trim().startsWith('linkStyle')) // Keep filtering linkStyle
            .join('\n');

          const mermaidId = `mermaid-${Date.now()}`;
          mermaid.render(mermaidId, processedContent)
            .then(({ svg, bindFunctions }) => {
              // Check ref again inside async callback
              // Ensure the container is still the one we intended to update
              if (mermaidRef.current === container) {
                container.innerHTML = svg;
                if (bindFunctions) {
                  try { // Add try-catch around bindFunctions as it can also throw
                    bindFunctions(container);
                  } catch (bindError) {
                    console.error('Mermaid bindFunctions error:', bindError);
                    // Optionally display a message in the container
                    container.innerHTML += '<p class="text-orange-500 text-xs">Diagram interactions might be limited.</p>';
                  }
                }
              } else {
                console.log('Mermaid container changed before rendering completed.');
              }
            })
            .catch(error => {
              console.error('Mermaid rendering promise error (debounced):', error);
              console.error('Failed content (debounced):', processedContent);
              if (mermaidRef.current === container) {
                const errorMessage = error instanceof Error ? error.message : String(error);
                // Make error display more robust
                const errorPre = document.createElement('pre');
                errorPre.className = 'text-red-500 text-xs whitespace-pre-wrap break-words';
                errorPre.textContent = `Mermaid diagram error: ${errorMessage}\n\nContent:\n${processedContent}`;
                container.innerHTML = ''; // Clear previous content
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
            container.innerHTML = ''; // Clear previous content
            container.appendChild(errorPre);
          }
        }
      }, 300); // 300ms debounce delay
    }

    // Cleanup function to clear the timer
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [language, children, theme]); // Dependencies

  // Render based on language type
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
};
