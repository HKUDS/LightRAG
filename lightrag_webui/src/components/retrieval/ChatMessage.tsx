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
  // Remove extra spaces around bold text
  // message.content = message.content.replace(/\*\ {3}/g, '').replace(/\ {4}\*\*/g, '**').replace(/\*\ \[/g, '[')

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
            suppressErrorRendering: true,
            gitGraph: {
              useMaxWidth: true,
              mainBranchName: 'main'
            },
            flowchart: {
              useMaxWidth: true,
              htmlLabels: true
            }
          });

          // Show loading indicator while processing
          container.innerHTML = '<div class="flex justify-center items-center p-4"><svg class="animate-spin h-5 w-5 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg></div>';

          // Preprocess mermaid content
          const rawContent = String(children).replace(/\n$/, '').trim(); // Trim whitespace as well

          // Preprocess gitGraph content to ensure correct format
          let processedRawContent = rawContent;
          if (rawContent.includes('gitGraph') && !rawContent.startsWith('gitGraph')) {
          // Attempt to fix incorrect gitGraph format
          processedRawContent = 'gitGraph:\n' + rawContent.replace(/gitGraph:?/i, '');
          console.log("Fixed gitGraph format:", processedRawContent.substring(0, 100));
          }

          // Heuristic check for potentially complete graph definition
          // Looks for graph type declaration and some content beyond it.
          let looksPotentiallyComplete = processedRawContent.length > 10 && (
          processedRawContent.startsWith('graph') ||
          processedRawContent.startsWith('sequenceDiagram') ||
          processedRawContent.startsWith('classDiagram') ||
          processedRawContent.startsWith('stateDiagram') ||
          processedRawContent.startsWith('gantt') ||
          processedRawContent.startsWith('pie') ||
          processedRawContent.startsWith('flowchart') ||
          processedRawContent.startsWith('erDiagram') ||
          processedRawContent.startsWith('gitGraph') ||
          processedRawContent.startsWith('journey') ||
          processedRawContent.startsWith('mindmap') ||
          processedRawContent.startsWith('timeline') ||
          /^gitGraph\s*:*/.test(processedRawContent) ||
          /^graph\s+TD/.test(processedRawContent) ||
          /^graph\s+LR/.test(processedRawContent)
          );

          // Handle timeline type by converting it to gantt chart
          if (processedRawContent.startsWith('timeline')) {
          processedRawContent = processedRawContent
          .replace('timeline', 'gantt')
          .replace(/section\s+([^\n]+)/g, 'section $1')
          .replace(/\s*:\s*([^,]+),\s*([^,]+),\s*([^\n]+)/g, '    $1 :$2, $3');
          }

          // If content looks incomplete but contains gitGraph keyword, attempt to fix
          if (!looksPotentiallyComplete && processedRawContent.includes('gitGraph')) {
            looksPotentiallyComplete = true;
          }

          // Force attempt to render even if it looks incomplete
          if (!looksPotentiallyComplete) {
            // For short content, we still attempt to render
            if (processedRawContent.length > 5) {
            looksPotentiallyComplete = true;
            }
          }

          // Always attempt to render, no longer skipping
          looksPotentiallyComplete = true;


          if (!looksPotentiallyComplete) {
             console.log("Mermaid content might be incomplete, skipping render attempt:", rawContent);
             return;
          }

          const processedContent = processedRawContent
            .split('\n')
            .map(line => {
              const trimmedLine = line.trim();
              // Keep subgraph processing
              if (trimmedLine.startsWith('subgraph')) {
                const parts = trimmedLine.split(' ');
                if (parts.length > 1) {
                  const title = parts.slice(1).join(' ').replace(/[\"']/g, '');
                  return `subgraph "${title}"`;
                }
              }
              // Handle gitGraph branch names with spaces
              if (trimmedLine.startsWith('branch') || trimmedLine.startsWith('checkout')) {
                const parts = trimmedLine.split(' ');
                if (parts.length > 1) {
                  const branchName = parts.slice(1).join(' ').replace(/["']/g, '');
                  return `${parts[0]} "${branchName}"`;
                }
              }
              // Handling Chinese and special characters in commit messages
              if (trimmedLine.startsWith('commit') && trimmedLine.includes('"')) {
                return trimmedLine;
              } else if (trimmedLine.startsWith('commit') && trimmedLine.includes(' ')) {
                const parts = trimmedLine.split(' ');
                if (parts.length > 1) {
                  const message = parts.slice(1).join(' ');
                  return `commit "${message}"`;
                }
              }
              return trimmedLine;
            })
            .filter(line => !line.trim().startsWith('linkStyle')) // Keep filtering linkStyle
            .join('\n');

          const mermaidId = `mermaid-${Date.now()}`;
          mermaid.render(mermaidId, processedContent)
            .then(({ svg, bindFunctions }) => {
              if (mermaidRef.current === container) {
                container.innerHTML = svg;
                if (bindFunctions) {
                  try {
                    bindFunctions(container);
                  } catch (bindError) {
                    console.error('Mermaid bindFunctions error:', bindError);
                    container.innerHTML += `<p class="text-orange-500 text-xs">Diagram interactions might be limited.</p>`;
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
                handleMermaidError(container, error, processedContent);
              }
            });

        } catch (error) {
          console.error('Mermaid synchronous error (debounced):', error);
          console.error('Failed content (debounced):', String(children));
          if (mermaidRef.current === container) {
            handleMermaidError(container, error);
          }
        }
      }, 1000); // 1000ms debounce delay
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
      className={cn(className, 'mx-1 rounded-sm bg-muted px-1 py-0.5 font-mono text-sm')}
      {...props}
    >
      {children}
    </code>
  );
};

const handleMermaidError = (container: HTMLDivElement, error: unknown, content?: string) => {
  if (!container) return;
  const errorMessage = error instanceof Error ? error.message : String(error);

  let contentPreview = '';
  if (content) {
    contentPreview = content.length > 50 ? content.substring(0, 50) + '...' : content;
  }

  container.innerHTML = `
    <div class="flex flex-col gap-2 p-4 text-red-500 dark:text-red-400 text-sm">
      <div class="font-medium">Failed to render diagram</div>
      <div class="text-xs opacity-80">${errorMessage}</div>
      ${contentPreview ? `<div class="text-xs mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded overflow-x-auto"><code>${contentPreview.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></div>` : ''}
      <div class="text-xs mt-2">Try simplifying your diagram or check syntax.</div>
    </div>
  `;
};
