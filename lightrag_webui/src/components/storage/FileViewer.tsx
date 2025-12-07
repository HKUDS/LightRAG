import { s3Download } from '@/api/lightrag'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/Sheet'
import useTheme from '@/hooks/useTheme'
import { cn } from '@/lib/utils'
import {
  DownloadIcon,
  FileIcon,
  FileTextIcon,
  ImageIcon,
  Loader2Icon,
  FileCodeIcon,
  GripVerticalIcon,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/cjs/styles/prism'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import Button from '@/components/ui/Button'
import { ScrollArea } from '@/components/ui/ScrollArea'
import PDFViewer from './PDFViewer'

interface FileViewerProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  fileKey: string | null
  fileName: string
  fileSize: number
}

type FileType = 'text' | 'markdown' | 'json' | 'code' | 'image' | 'pdf' | 'unknown'

// Storage key for persisted width
const VIEWER_WIDTH_KEY = 'lightrag-viewer-width'
const DEFAULT_WIDTH = 672 // ~42rem
const MIN_WIDTH = 400
const MAX_WIDTH = 1200

// Get file type from extension
function getFileType(fileName: string): FileType {
  const ext = fileName.split('.').pop()?.toLowerCase() || ''

  // Text files
  if (['txt', 'log', 'csv', 'tsv'].includes(ext)) return 'text'

  // Markdown
  if (['md', 'markdown', 'mdx'].includes(ext)) return 'markdown'

  // JSON/YAML/Config
  if (['json', 'yaml', 'yml', 'toml', 'ini', 'conf', 'config'].includes(ext)) return 'json'

  // Code files
  if (
    [
      'js',
      'ts',
      'jsx',
      'tsx',
      'py',
      'java',
      'c',
      'cpp',
      'h',
      'hpp',
      'go',
      'rs',
      'rb',
      'php',
      'swift',
      'kt',
      'scala',
      'sql',
      'sh',
      'bash',
      'zsh',
      'css',
      'scss',
      'less',
      'html',
      'htm',
      'xml',
    ].includes(ext)
  )
    return 'code'

  // Images
  if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'ico', 'bmp'].includes(ext)) return 'image'

  // PDF
  if (ext === 'pdf') return 'pdf'

  return 'unknown'
}

// Get syntax highlighter language from extension
function getLanguage(fileName: string): string {
  const ext = fileName.split('.').pop()?.toLowerCase() || ''
  const langMap: Record<string, string> = {
    js: 'javascript',
    jsx: 'jsx',
    ts: 'typescript',
    tsx: 'tsx',
    py: 'python',
    java: 'java',
    c: 'c',
    cpp: 'cpp',
    h: 'c',
    hpp: 'cpp',
    go: 'go',
    rs: 'rust',
    rb: 'ruby',
    php: 'php',
    swift: 'swift',
    kt: 'kotlin',
    scala: 'scala',
    sql: 'sql',
    sh: 'bash',
    bash: 'bash',
    zsh: 'bash',
    css: 'css',
    scss: 'scss',
    less: 'less',
    html: 'html',
    htm: 'html',
    xml: 'xml',
    json: 'json',
    yaml: 'yaml',
    yml: 'yaml',
    toml: 'toml',
    ini: 'ini',
    md: 'markdown',
    markdown: 'markdown',
  }
  return langMap[ext] || 'text'
}

// Get icon for file type
function FileTypeIcon({ fileType, className }: { fileType: FileType; className?: string }) {
  switch (fileType) {
    case 'text':
      return <FileTextIcon className={className} />
    case 'markdown':
      return <FileTextIcon className={className} />
    case 'code':
    case 'json':
      return <FileCodeIcon className={className} />
    case 'image':
      return <ImageIcon className={className} />
    default:
      return <FileIcon className={className} />
  }
}

// Format bytes to human readable
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
}

export default function FileViewer({
  open,
  onOpenChange,
  fileKey,
  fileName,
  fileSize,
}: FileViewerProps) {
  const { t } = useTranslation()
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme === 'dark'

  const [content, setContent] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Resizable width state
  const [width, setWidth] = useState(() => {
    const saved = localStorage.getItem(VIEWER_WIDTH_KEY)
    return saved ? parseInt(saved, 10) : DEFAULT_WIDTH
  })
  const [isResizing, setIsResizing] = useState(false)
  const resizeRef = useRef<{ startX: number; startWidth: number } | null>(null)

  const fileType = getFileType(fileName)
  const language = getLanguage(fileName)

  // Handle resize start
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
    resizeRef.current = { startX: e.clientX, startWidth: width }
  }, [width])

  // Handle resize move
  useEffect(() => {
    if (!isResizing) return

    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeRef.current) return
      const delta = resizeRef.current.startX - e.clientX
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, resizeRef.current.startWidth + delta))
      setWidth(newWidth)
    }

    const handleMouseUp = () => {
      setIsResizing(false)
      localStorage.setItem(VIEWER_WIDTH_KEY, String(width))
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isResizing, width])

  // Fetch file content when opened
  useEffect(() => {
    if (!open || !fileKey) {
      setContent(null)
      setImageUrl(null)
      setError(null)
      return
    }

    const fetchContent = async () => {
      setLoading(true)
      setError(null)

      try {
        const response = await s3Download(fileKey)
        const presignedUrl = response.url

        if (fileType === 'image') {
          setImageUrl(presignedUrl)
        } else if (fileType === 'pdf') {
          // For PDF, we just set the URL - user can open in new tab
          setImageUrl(presignedUrl)
        } else if (fileType !== 'unknown') {
          // Fetch text content
          const textResponse = await fetch(presignedUrl)
          if (!textResponse.ok) {
            throw new Error(`Failed to fetch: ${textResponse.statusText}`)
          }
          const text = await textResponse.text()
          setContent(text)
        } else {
          // Unknown file type - just provide download link
          setImageUrl(presignedUrl)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load file')
      } finally {
        setLoading(false)
      }
    }

    fetchContent()
  }, [open, fileKey, fileType])

  const handleDownload = useCallback(async () => {
    if (!fileKey) return
    try {
      const response = await s3Download(fileKey)
      window.open(response.url, '_blank')
    } catch {
      // Error handled silently
    }
  }, [fileKey])

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="flex flex-col p-0"
        style={{ width: `${width}px`, maxWidth: '90vw' }}
      >
        {/* Resize handle */}
        <div
          className={cn(
            'absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-primary/50 transition-colors z-50 group',
            isResizing && 'bg-primary/50'
          )}
          onMouseDown={handleResizeStart}
        >
          <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVerticalIcon className="h-6 w-6 text-muted-foreground" />
          </div>
        </div>

        <div className="p-6 pb-0 flex-shrink-0">
          <SheetHeader>
            <div className="flex items-center gap-2 pr-8">
              <FileTypeIcon fileType={fileType} className="h-5 w-5 text-muted-foreground" />
              <SheetTitle className="truncate">{fileName}</SheetTitle>
            </div>
            <SheetDescription className="flex items-center justify-between">
              <span>
                {formatBytes(fileSize)} • {fileType.toUpperCase()}
              </span>
              <Button variant="outline" size="sm" onClick={handleDownload}>
                <DownloadIcon className="h-4 w-4 mr-1" />
                {t('storagePanel.actions.download')}
              </Button>
            </SheetDescription>
          </SheetHeader>
        </div>

        <div className="flex-1 mt-4 min-h-0 overflow-hidden px-6 pb-6">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-full text-destructive gap-2">
              <p>{error}</p>
              <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>
                {t('common.close')}
              </Button>
            </div>
          ) : fileType === 'image' && imageUrl ? (
            <div className="flex items-center justify-center h-full bg-muted/30 rounded-lg p-4">
              <img
                src={imageUrl}
                alt={fileName}
                className="max-w-full max-h-full object-contain rounded"
              />
            </div>
          ) : fileType === 'pdf' && imageUrl ? (
            <PDFViewer url={imageUrl} />
          ) : fileType === 'unknown' && imageUrl ? (
            <div className="flex flex-col items-center justify-center h-full gap-4">
              <FileIcon className="h-16 w-16 text-muted-foreground" />
              <p className="text-muted-foreground">{t('storagePanel.viewer.noPreview')}</p>
              <Button variant="default" onClick={handleDownload}>
                <DownloadIcon className="h-4 w-4 mr-1" />
                {t('storagePanel.actions.download')}
              </Button>
            </div>
          ) : fileType === 'markdown' && content ? (
            <ScrollArea className="h-full">
              <div className="prose prose-sm dark:prose-invert max-w-none p-4">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={{
                    code({ className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '')
                      const inline = !match
                      return !inline ? (
                        <SyntaxHighlighter
                          style={isDark ? oneDark : oneLight}
                          language={match?.[1] || 'text'}
                          PreTag="div"
                          customStyle={{
                            margin: 0,
                            borderRadius: '0.375rem',
                            fontSize: '0.875rem',
                          }}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={cn('bg-muted px-1 py-0.5 rounded', className)} {...props}>
                          {children}
                        </code>
                      )
                    },
                    // Allow images from external sources
                    img({ src, alt, ...props }) {
                      return (
                        <img
                          src={src}
                          alt={alt}
                          className="max-w-full h-auto rounded"
                          loading="lazy"
                          {...props}
                        />
                      )
                    },
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            </ScrollArea>
          ) : (fileType === 'code' || fileType === 'json') && content ? (
            <ScrollArea className="h-full">
              <SyntaxHighlighter
                style={isDark ? oneDark : oneLight}
                language={language}
                showLineNumbers
                customStyle={{
                  margin: 0,
                  borderRadius: '0.375rem',
                  fontSize: '0.8125rem',
                  minHeight: '100%',
                }}
              >
                {content}
              </SyntaxHighlighter>
            </ScrollArea>
          ) : content ? (
            <ScrollArea className="h-full">
              <pre className="p-4 text-sm whitespace-pre-wrap break-words font-mono bg-muted/30 rounded-lg min-h-full">
                {content}
              </pre>
            </ScrollArea>
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  )
}
