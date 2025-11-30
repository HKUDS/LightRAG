import { cn } from '@/lib/utils'
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronsLeftIcon,
  ChevronsRightIcon,
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import Button from './Button'
import Input from './Input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './Select'

export type PaginationControlsProps = {
  currentPage: number
  totalPages: number
  pageSize: number
  totalCount: number
  onPageChange: (page: number) => void
  onPageSizeChange: (pageSize: number) => void
  isLoading?: boolean
  compact?: boolean
  className?: string
}

const PAGE_SIZE_OPTIONS = [
  { value: 10, label: '10' },
  { value: 20, label: '20' },
  { value: 50, label: '50' },
  { value: 100, label: '100' },
  { value: 200, label: '200' },
]

export default function PaginationControls({
  currentPage,
  totalPages,
  pageSize,
  totalCount,
  onPageChange,
  onPageSizeChange,
  isLoading = false,
  compact = false,
  className,
}: PaginationControlsProps) {
  const { t } = useTranslation()
  const [inputPage, setInputPage] = useState(currentPage.toString())

  // Update input when currentPage changes
  useEffect(() => {
    setInputPage(currentPage.toString())
  }, [currentPage])

  // Handle page input change with debouncing
  const handlePageInputChange = useCallback((value: string) => {
    setInputPage(value)
  }, [])

  // Handle page input submit
  const handlePageInputSubmit = useCallback(() => {
    const pageNum = Number.parseInt(inputPage, 10)
    if (!isNaN(pageNum) && pageNum >= 1 && pageNum <= totalPages) {
      onPageChange(pageNum)
    } else {
      // Reset to current page if invalid
      setInputPage(currentPage.toString())
    }
  }, [inputPage, totalPages, onPageChange, currentPage])

  // Handle page input key press
  const handlePageInputKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        handlePageInputSubmit()
      }
    },
    [handlePageInputSubmit]
  )

  // Handle page size change
  const handlePageSizeChange = useCallback(
    (value: string) => {
      const newPageSize = Number.parseInt(value, 10)
      if (!isNaN(newPageSize)) {
        onPageSizeChange(newPageSize)
      }
    },
    [onPageSizeChange]
  )

  // Navigation handlers
  const goToFirstPage = useCallback(() => {
    if (currentPage > 1 && !isLoading) {
      onPageChange(1)
    }
  }, [currentPage, onPageChange, isLoading])

  const goToPrevPage = useCallback(() => {
    if (currentPage > 1 && !isLoading) {
      onPageChange(currentPage - 1)
    }
  }, [currentPage, onPageChange, isLoading])

  const goToNextPage = useCallback(() => {
    if (currentPage < totalPages && !isLoading) {
      onPageChange(currentPage + 1)
    }
  }, [currentPage, totalPages, onPageChange, isLoading])

  const goToLastPage = useCallback(() => {
    if (currentPage < totalPages && !isLoading) {
      onPageChange(totalPages)
    }
  }, [currentPage, totalPages, onPageChange, isLoading])

  if (totalPages <= 1) {
    return null
  }

  if (compact) {
    return (
      <div className={cn('flex items-center gap-2', className)}>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={goToPrevPage}
            disabled={currentPage <= 1 || isLoading}
            className="h-8 w-8 p-0"
          >
            <ChevronLeftIcon className="h-4 w-4" />
          </Button>

          <div className="flex items-center gap-1">
            <Input
              type="text"
              value={inputPage}
              onChange={(e) => handlePageInputChange(e.target.value)}
              onBlur={handlePageInputSubmit}
              onKeyPress={handlePageInputKeyPress}
              disabled={isLoading}
              className="h-8 w-12 text-center text-sm"
            />
            <span className="text-sm text-gray-500">/ {totalPages}</span>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={goToNextPage}
            disabled={currentPage >= totalPages || isLoading}
            className="h-8 w-8 p-0"
          >
            <ChevronRightIcon className="h-4 w-4" />
          </Button>
        </div>

        <Select
          value={pageSize.toString()}
          onValueChange={handlePageSizeChange}
          disabled={isLoading}
        >
          <SelectTrigger className="h-8 w-16">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PAGE_SIZE_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value.toString()}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    )
  }

  return (
    <div className={cn('flex items-center justify-between gap-4', className)}>
      <div className="text-sm text-gray-500">
        {t('pagination.showing', {
          start: Math.min((currentPage - 1) * pageSize + 1, totalCount),
          end: Math.min(currentPage * pageSize, totalCount),
          total: totalCount,
        })}
      </div>

      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={goToFirstPage}
            disabled={currentPage <= 1 || isLoading}
            className="h-8 w-8 p-0"
            tooltip={t('pagination.firstPage')}
          >
            <ChevronsLeftIcon className="h-4 w-4" />
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={goToPrevPage}
            disabled={currentPage <= 1 || isLoading}
            className="h-8 w-8 p-0"
            tooltip={t('pagination.prevPage')}
          >
            <ChevronLeftIcon className="h-4 w-4" />
          </Button>

          <div className="flex items-center gap-1">
            <span className="text-sm">{t('pagination.page')}</span>
            <Input
              type="text"
              value={inputPage}
              onChange={(e) => handlePageInputChange(e.target.value)}
              onBlur={handlePageInputSubmit}
              onKeyPress={handlePageInputKeyPress}
              disabled={isLoading}
              className="h-8 w-16 text-center text-sm"
            />
            <span className="text-sm">/ {totalPages}</span>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={goToNextPage}
            disabled={currentPage >= totalPages || isLoading}
            className="h-8 w-8 p-0"
            tooltip={t('pagination.nextPage')}
          >
            <ChevronRightIcon className="h-4 w-4" />
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={goToLastPage}
            disabled={currentPage >= totalPages || isLoading}
            className="h-8 w-8 p-0"
            tooltip={t('pagination.lastPage')}
          >
            <ChevronsRightIcon className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm">{t('pagination.pageSize')}</span>
          <Select
            value={pageSize.toString()}
            onValueChange={handlePageSizeChange}
            disabled={isLoading}
          >
            <SelectTrigger className="h-8 w-16">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PAGE_SIZE_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value.toString()}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  )
}
