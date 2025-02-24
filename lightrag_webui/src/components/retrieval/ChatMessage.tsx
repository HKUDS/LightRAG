import { ReactNode, useCallback } from 'react'
import { Message } from '@/api/lightrag'
import useTheme from '@/hooks/useTheme'
import Button from '@/components/ui/Button'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeReact from 'rehype-react'
import remarkMath from 'remark-math'
import ShikiHighlighter, { isInlineCode, type Element } from 'react-shiki'

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
  className?: string | undefined
  children?: ReactNode | undefined
  node?: Element | undefined
}

const CodeHighlight = ({ className, children, node, ...props }: CodeHighlightProps) => {
  const { theme } = useTheme()
  const match = className?.match(/language-(\w+)/)
  const language = match ? match[1] : undefined
  const inline: boolean | undefined = node ? isInlineCode(node) : undefined

  return !inline ? (
    <ShikiHighlighter
      language={language}
      theme={theme === 'dark' ? 'houston' : 'github-light'}
      {...props}
    >
      {String(children)}
    </ShikiHighlighter>
  ) : (
    <code className={className} {...props}>
      {children}
    </code>
  )
}
