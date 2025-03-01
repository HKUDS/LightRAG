import { ReactNode, useCallback } from 'react'
import { Message } from '@/api/lightrag'
import useTheme from '@/hooks/useTheme'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeReact from 'rehype-react'
import remarkMath from 'remark-math'

import type { Element } from 'hast'

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'

import { LoaderIcon, CopyIcon } from 'lucide-react'

export type MessageWithError = Message & {
  isError?: boolean
}

export const ChatMessage = ({ message }: { message: MessageWithError }) => {
  const handleCopyMarkdown = useCallback(async () => {
    if (message.content) {
      try {
        await navigator.clipboard.writeText(message.content)
      } catch (err) {
        console.error('Failed to copy:', err)
      }
    }
  }, [message])

  return (
    <div
      className={`max-w-[80%] rounded-lg px-4 py-2 ${
        message.role === 'user'
          ? 'bg-primary text-primary-foreground'
          : message.isError
            ? 'bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400'
            : 'bg-muted'
      }`}
    >
      <pre className="relative break-words whitespace-pre-wrap">
        <ReactMarkdown
          className="dark:prose-invert max-w-none text-base text-sm"
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeReact]}
          skipHtml={false}
          components={{
            code: CodeHighlight
          }}
        >
          {message.content}
        </ReactMarkdown>
        {message.role === 'assistant' && message.content.length > 0 && (
          <Button
            onClick={handleCopyMarkdown}
            className="absolute right-0 bottom-0 size-6 rounded-md opacity-20 transition-opacity hover:opacity-100"
            tooltip="Copy to clipboard"
            variant="default"
            size="icon"
          >
            <CopyIcon />
          </Button>
        )}
      </pre>
      {message.content.length === 0 && <LoaderIcon className="animate-spin duration-2000" />}
    </div>
  )
}

interface CodeHighlightProps {
  inline?: boolean
  className?: string
  children?: ReactNode
  node?: Element
}

const isInlineCode = (node: Element): boolean => {
  const textContent = (node.children || [])
    .filter((child) => child.type === 'text')
    .map((child) => (child as any).value)
    .join('')

  return !textContent.includes('\n')
}

const CodeHighlight = ({ className, children, node, ...props }: CodeHighlightProps) => {
  const { theme } = useTheme()
  const match = className?.match(/language-(\w+)/)
  const language = match ? match[1] : undefined
  const inline = node ? isInlineCode(node) : false

  return !inline ? (
    <SyntaxHighlighter
      style={theme === 'dark' ? oneDark : oneLight}
      PreTag="div"
      language={language}
      {...props}
    >
      {String(children).replace(/\n$/, '')}
    </SyntaxHighlighter>
  ) : (
    <code
      className={cn(className, 'mx-1 rounded-xs bg-black/10 px-1 dark:bg-gray-100/20')}
      {...props}
    >
      {children}
    </code>
  )
}
