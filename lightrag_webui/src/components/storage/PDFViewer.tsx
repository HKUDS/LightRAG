import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  MinusIcon,
  PlusIcon,
  Loader2Icon,
  Maximize2Icon,
  MaximizeIcon,
  MinimizeIcon,
  SearchIcon,
  XIcon,
} from 'lucide-react'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { cn } from '@/lib/utils'

// Configure pdf.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

// Storage keys
const ZOOM_KEY = 'lightrag-pdf-zoom'

// Escape special regex characters
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

interface SearchMatch {
  pageNum: number
  matchIndex: number
}

interface PDFViewerProps {
  url: string
}

export default function PDFViewer({ url }: PDFViewerProps) {
  const [numPages, setNumPages] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [inputValue, setInputValue] = useState('1')
  const [pdfWidth, setPdfWidth] = useState<number | null>(null)
  const [fitMode, setFitMode] = useState<'manual' | 'width'>('manual')
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Search state
  const [showSearch, setShowSearch] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [textContent, setTextContent] = useState<Map<number, string>>(new Map())
  const [matches, setMatches] = useState<SearchMatch[]>([])
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0)

  // Persisted zoom
  const [scale, setScale] = useState(() => {
    const saved = localStorage.getItem(ZOOM_KEY)
    return saved ? parseFloat(saved) : 1.0
  })

  const viewerRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map())

  // Persist zoom to localStorage
  useEffect(() => {
    if (fitMode === 'manual') {
      localStorage.setItem(ZOOM_KEY, String(scale))
    }
  }, [scale, fitMode])

  // Calculate fit-to-width scale
  const calculateFitWidth = useCallback(() => {
    if (!containerRef.current || !pdfWidth) return 1.0
    const containerWidth = containerRef.current.clientWidth - 48 // padding
    return Math.min(2.0, Math.max(0.5, containerWidth / pdfWidth))
  }, [pdfWidth])

  // Update scale when fit mode is active and window resizes
  useEffect(() => {
    if (fitMode !== 'width') return

    const updateFitWidth = () => {
      setScale(calculateFitWidth())
    }

    updateFitWidth()
    window.addEventListener('resize', updateFitWidth)
    return () => window.removeEventListener('resize', updateFitWidth)
  }, [fitMode, calculateFitWidth])

  // Track which page is visible via IntersectionObserver
  useEffect(() => {
    if (numPages === 0) return

    const observer = new IntersectionObserver(
      (entries) => {
        let maxVisibility = 0
        let mostVisiblePage = currentPage

        entries.forEach((entry) => {
          if (entry.intersectionRatio > maxVisibility) {
            maxVisibility = entry.intersectionRatio
            const pageNum = parseInt(entry.target.getAttribute('data-page') || '1')
            mostVisiblePage = pageNum
          }
        })

        if (maxVisibility > 0.3) {
          setCurrentPage(mostVisiblePage)
          setInputValue(String(mostVisiblePage))
        }
      },
      {
        root: containerRef.current,
        threshold: [0, 0.25, 0.5, 0.75, 1],
      }
    )

    pageRefs.current.forEach((ref) => {
      if (ref) observer.observe(ref)
    })

    return () => observer.disconnect()
  }, [numPages, currentPage])

  // Fullscreen change listener
  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', handler)
    return () => document.removeEventListener('fullscreenchange', handler)
  }, [])

  // Search: Find matches when query changes
  useEffect(() => {
    if (!searchQuery.trim() || textContent.size === 0) {
      setMatches([])
      setCurrentMatchIndex(0)
      return
    }

    const results: SearchMatch[] = []
    const query = searchQuery.toLowerCase()

    textContent.forEach((text, pageNum) => {
      const lowerText = text.toLowerCase()
      let startIndex = 0
      let matchIndex = 0

      while ((startIndex = lowerText.indexOf(query, startIndex)) !== -1) {
        results.push({ pageNum, matchIndex })
        startIndex += query.length
        matchIndex++
      }
    })

    setMatches(results)
    setCurrentMatchIndex(0)

    // Navigate to first match
    if (results.length > 0) {
      goToPage(results[0].pageNum)
    }
  }, [searchQuery, textContent])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+F / Cmd+F to open search
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault()
        setShowSearch(true)
        setTimeout(() => searchInputRef.current?.focus(), 0)
        return
      }

      // If search is open, handle search-specific shortcuts
      if (showSearch && searchInputRef.current === document.activeElement) {
        if (e.key === 'Enter') {
          e.preventDefault()
          if (e.shiftKey) {
            prevMatch()
          } else {
            nextMatch()
          }
          return
        }
        if (e.key === 'Escape') {
          e.preventDefault()
          closeSearch()
          return
        }
        return // Don't handle other shortcuts when search input is focused
      }

      // Only handle if viewer is focused or in fullscreen
      if (!viewerRef.current?.contains(document.activeElement) && !isFullscreen) return

      switch (e.key) {
        case 'ArrowLeft':
        case 'ArrowUp':
          e.preventDefault()
          goToPage(currentPage - 1)
          break
        case 'ArrowRight':
        case 'ArrowDown':
        case ' ':
          e.preventDefault()
          goToPage(currentPage + 1)
          break
        case '+':
        case '=':
          e.preventDefault()
          zoomIn()
          break
        case '-':
          e.preventDefault()
          zoomOut()
          break
        case 'Home':
          e.preventDefault()
          goToPage(1)
          break
        case 'End':
          e.preventDefault()
          goToPage(numPages)
          break
        case 'Escape':
          if (showSearch) {
            e.preventDefault()
            closeSearch()
          } else if (isFullscreen) {
            e.preventDefault()
            toggleFullscreen()
          }
          break
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [currentPage, numPages, isFullscreen, showSearch, matches, currentMatchIndex])

  const goToPage = (page: number) => {
    const targetPage = Math.max(1, Math.min(page, numPages))
    const pageElement = pageRefs.current.get(targetPage)
    if (pageElement) {
      pageElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
    setCurrentPage(targetPage)
    setInputValue(String(targetPage))
  }

  const zoomIn = () => {
    setFitMode('manual')
    setScale((s) => Math.min(2, s + 0.25))
  }

  const zoomOut = () => {
    setFitMode('manual')
    setScale((s) => Math.max(0.5, s - 0.25))
  }

  const toggleFitWidth = () => {
    if (fitMode === 'width') {
      setFitMode('manual')
      const saved = localStorage.getItem(ZOOM_KEY)
      if (saved) setScale(parseFloat(saved))
    } else {
      setFitMode('width')
      setScale(calculateFitWidth())
    }
  }

  const toggleFullscreen = async () => {
    if (!document.fullscreenElement) {
      await viewerRef.current?.requestFullscreen()
    } else {
      await document.exitFullscreen()
    }
  }

  // Search navigation
  const nextMatch = () => {
    if (matches.length === 0) return
    const nextIndex = (currentMatchIndex + 1) % matches.length
    setCurrentMatchIndex(nextIndex)
    goToPage(matches[nextIndex].pageNum)
  }

  const prevMatch = () => {
    if (matches.length === 0) return
    const prevIndex = (currentMatchIndex - 1 + matches.length) % matches.length
    setCurrentMatchIndex(prevIndex)
    goToPage(matches[prevIndex].pageNum)
  }

  const closeSearch = () => {
    setShowSearch(false)
    setSearchQuery('')
    setMatches([])
    setCurrentMatchIndex(0)
    viewerRef.current?.focus()
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value)
  }

  const handleInputBlur = () => {
    const page = parseInt(inputValue) || 1
    goToPage(page)
  }

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      const page = parseInt(inputValue) || 1
      goToPage(page)
    }
  }

  // Extract text content from PDF for search
  const extractTextContent = async (pdf: PDFDocumentProxy) => {
    const textMap = new Map<number, string>()

    for (let i = 1; i <= pdf.numPages; i++) {
      try {
        const page = await pdf.getPage(i)
        const content = await page.getTextContent()
        const text = content.items
          .map((item) => ('str' in item ? item.str : ''))
          .join(' ')
        textMap.set(i, text)
      } catch {
        // Skip pages that fail to extract
      }
    }

    setTextContent(textMap)
  }

  const handleLoadSuccess = (pdf: PDFDocumentProxy) => {
    setNumPages(pdf.numPages)
    setLoading(false)
    // Extract text content for search
    extractTextContent(pdf)
  }

  const handlePageLoadSuccess = (page: { width: number }) => {
    if (!pdfWidth) {
      setPdfWidth(page.width)
    }
  }

  const handleLoadError = (err: Error) => {
    setError(err.message)
    setLoading(false)
  }

  // Custom text renderer for highlighting search matches
  const customTextRenderer = useCallback(
    ({ str }: { str: string }) => {
      if (!searchQuery.trim()) return str

      const regex = new RegExp(`(${escapeRegExp(searchQuery)})`, 'gi')
      const parts = str.split(regex)

      if (parts.length === 1) return str

      return parts
        .map((part, i) => {
          if (part.toLowerCase() === searchQuery.toLowerCase()) {
            return `<mark class="bg-yellow-300 dark:bg-yellow-600 rounded px-0.5">${part}</mark>`
          }
          return part
        })
        .join('')
    },
    [searchQuery]
  )

  return (
    <div
      ref={viewerRef}
      className={cn('flex flex-col h-full', isFullscreen && 'bg-background')}
      tabIndex={0}
    >
      {/* Search bar */}
      {showSearch && (
        <div className="flex items-center gap-2 p-2 border-b bg-background flex-shrink-0">
          <SearchIcon className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <Input
            ref={searchInputRef}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search in document..."
            className="flex-1 h-8"
            autoFocus
          />
          {matches.length > 0 ? (
            <>
              <Button
                size="icon"
                variant="ghost"
                onClick={prevMatch}
                className="h-8 w-8 flex-shrink-0"
                title="Previous match (Shift+Enter)"
              >
                <ChevronUpIcon className="h-4 w-4" />
              </Button>
              <span className="text-sm text-muted-foreground whitespace-nowrap min-w-[4rem] text-center">
                {currentMatchIndex + 1} of {matches.length}
              </span>
              <Button
                size="icon"
                variant="ghost"
                onClick={nextMatch}
                className="h-8 w-8 flex-shrink-0"
                title="Next match (Enter)"
              >
                <ChevronDownIcon className="h-4 w-4" />
              </Button>
            </>
          ) : searchQuery.trim() ? (
            <span className="text-sm text-muted-foreground whitespace-nowrap">No matches</span>
          ) : null}
          <Button
            size="icon"
            variant="ghost"
            onClick={closeSearch}
            className="h-8 w-8 flex-shrink-0"
            title="Close (Escape)"
          >
            <XIcon className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Scrollable PDF pages */}
      <div
        ref={containerRef}
        className={cn(
          'flex-1 overflow-auto bg-muted/30',
          'scrollbar-thin scrollbar-track-transparent',
          'scrollbar-thumb-transparent hover:scrollbar-thumb-muted-foreground/40'
        )}
      >
        {loading && (
          <div className="flex items-center justify-center h-full">
            <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        )}

        {error && (
          <div className="flex items-center justify-center h-full text-destructive">
            <p>Failed to load PDF: {error}</p>
          </div>
        )}

        <Document
          file={url}
          onLoadSuccess={handleLoadSuccess}
          onLoadError={handleLoadError}
          loading={null}
          error={null}
        >
          <div className="flex flex-col items-center py-4 gap-4 min-w-fit">
            {Array.from({ length: numPages }, (_, i) => (
              <div
                key={i}
                ref={(el) => {
                  if (el) pageRefs.current.set(i + 1, el)
                }}
                data-page={i + 1}
                className="shadow-lg"
              >
                <Page
                  pageNumber={i + 1}
                  scale={scale}
                  renderTextLayer
                  renderAnnotationLayer
                  onLoadSuccess={i === 0 ? handlePageLoadSuccess : undefined}
                  customTextRenderer={searchQuery.trim() ? customTextRenderer : undefined}
                  loading={
                    <div className="flex items-center justify-center p-8">
                      <Loader2Icon className="h-6 w-6 animate-spin text-muted-foreground" />
                    </div>
                  }
                />
              </div>
            ))}
          </div>
        </Document>
      </div>

      {/* Bottom toolbar */}
      {numPages > 0 && (
        <div className="flex items-center justify-center gap-2 p-2 border-t bg-background flex-shrink-0">
          {/* Search button */}
          <Button
            size="icon"
            variant={showSearch ? 'secondary' : 'ghost'}
            onClick={() => {
              setShowSearch(!showSearch)
              if (!showSearch) {
                setTimeout(() => searchInputRef.current?.focus(), 0)
              }
            }}
            className="h-8 w-8"
            title="Search (Ctrl+F)"
          >
            <SearchIcon className="h-4 w-4" />
          </Button>

          <div className="border-l h-6 mx-1" />

          {/* Page navigation */}
          <Button
            size="icon"
            variant="ghost"
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage <= 1}
            className="h-8 w-8"
            title="Previous page (←)"
          >
            <ChevronLeftIcon className="h-4 w-4" />
          </Button>

          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Page</span>
            <Input
              type="number"
              value={inputValue}
              onChange={handleInputChange}
              onBlur={handleInputBlur}
              onKeyDown={handleInputKeyDown}
              className="w-14 h-8 text-center"
              min={1}
              max={numPages}
            />
            <span className="text-sm text-muted-foreground">of {numPages}</span>
          </div>

          <Button
            size="icon"
            variant="ghost"
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage >= numPages}
            className="h-8 w-8"
            title="Next page (→)"
          >
            <ChevronRightIcon className="h-4 w-4" />
          </Button>

          <div className="border-l h-6 mx-2" />

          {/* Zoom controls */}
          <Button
            size="icon"
            variant="ghost"
            onClick={zoomOut}
            disabled={scale <= 0.5}
            className="h-8 w-8"
            title="Zoom out (−)"
          >
            <MinusIcon className="h-4 w-4" />
          </Button>
          <span className="text-sm w-12 text-center">{Math.round(scale * 100)}%</span>
          <Button
            size="icon"
            variant="ghost"
            onClick={zoomIn}
            disabled={scale >= 2}
            className="h-8 w-8"
            title="Zoom in (+)"
          >
            <PlusIcon className="h-4 w-4" />
          </Button>

          {/* Fit to width */}
          <Button
            size="icon"
            variant={fitMode === 'width' ? 'secondary' : 'ghost'}
            onClick={toggleFitWidth}
            className="h-8 w-8"
            title="Fit to width"
          >
            <Maximize2Icon className="h-4 w-4" />
          </Button>

          <div className="border-l h-6 mx-2" />

          {/* Fullscreen toggle */}
          <Button
            size="icon"
            variant="ghost"
            onClick={toggleFullscreen}
            className="h-8 w-8"
            title={isFullscreen ? 'Exit fullscreen (Esc)' : 'Enter fullscreen'}
          >
            {isFullscreen ? (
              <MinimizeIcon className="h-4 w-4" />
            ) : (
              <MaximizeIcon className="h-4 w-4" />
            )}
          </Button>
        </div>
      )}
    </div>
  )
}
