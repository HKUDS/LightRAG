import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { Loader2 } from 'lucide-react'
import { useDebounce } from '@/hooks/useDebounce'

import { cn } from '@/lib/utils'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList
} from '@/components/ui/Command'

export interface Option {
  value: string
  label: string
  disabled?: boolean
  description?: string
  icon?: React.ReactNode
}

export interface AsyncSearchProps<T> {
  /** Async function to fetch options */
  fetcher: (query?: string) => Promise<T[]>
  /** Preload all data ahead of time */
  preload?: boolean
  /** Function to filter options */
  filterFn?: (option: T, query: string) => boolean
  /** Function to render each option */
  renderOption: (option: T) => React.ReactNode
  /** Function to get the value from an option */
  getOptionValue: (option: T) => string
  /** Custom not found message */
  notFound?: React.ReactNode
  /** Custom loading skeleton */
  loadingSkeleton?: React.ReactNode
  /** Currently selected value */
  value: string | null
  /** Callback when selection changes */
  onChange: (value: string) => void
  /** Callback when focus changes */
  onFocus: (value: string) => void
  /** Accessibility label for the search field */
  ariaLabel?: string
  /** Placeholder text when no selection */
  placeholder?: string
  /** Disable the entire select */
  disabled?: boolean
  /** Custom width for the popover */
  width?: string | number
  /** Custom class names */
  className?: string
  /** Custom trigger button class names */
  triggerClassName?: string
  /** Custom no results message */
  noResultsMessage?: string
  /** Allow clearing the selection */
  clearable?: boolean
}

export function AsyncSearch<T>({
  fetcher,
  preload,
  filterFn,
  renderOption,
  getOptionValue,
  notFound,
  loadingSkeleton,
  ariaLabel,
  placeholder = 'Select...',
  value,
  onChange,
  onFocus,
  disabled = false,
  className,
  noResultsMessage
}: AsyncSearchProps<T>) {
  const [open, setOpen] = useState(false)
  const [fetchedOptions, setFetchedOptions] = useState<T[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const debouncedSearchTerm = useDebounce(searchTerm, preload ? 0 : 150)
  const containerRef = useRef<HTMLDivElement>(null)

  // Handle clicks outside of the component
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node) &&
        open
      ) {
        setOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [open])

  const fetchOptions = useCallback(async (query: string) => {
    try {
      setLoading(true)
      setError(null)
      const data = await fetcher(query)
      setFetchedOptions(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch options')
    } finally {
      setLoading(false)
    }
  }, [fetcher])

  // Fetch options from server when not in preload mode (search term changes).
  // The fetch is a genuine external side effect; setState happens inside fetchOptions
  // after the async call resolves, so the rule fires on the synchronous loading flag.
  useEffect(() => {
    if (preload) return
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchOptions(debouncedSearchTerm)
  }, [preload, debouncedSearchTerm, fetchOptions])

  // Load initial value
  useEffect(() => {
    if (!value) return
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchOptions(value)
  }, [value, fetchOptions])

  // In preload mode, derive filtered options without mutating state
  const options = useMemo(() => {
    if (preload && debouncedSearchTerm) {
      return fetchedOptions.filter((option) =>
        filterFn ? filterFn(option, debouncedSearchTerm) : true
      )
    }
    return fetchedOptions
  }, [preload, debouncedSearchTerm, filterFn, fetchedOptions])

  const handleSelect = useCallback((currentValue: string) => {
    onChange(currentValue)
    requestAnimationFrame(() => {
      // Blur the input to ensure focus event triggers on next click
      const input = document.activeElement as HTMLElement
      input?.blur()
      // Close the dropdown
      setOpen(false)
    })
  }, [onChange])

  const handleFocus = useCallback(() => {
    setOpen(true)
    // Use current search term to fetch options
    fetchOptions(searchTerm)
  }, [searchTerm, fetchOptions])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement
    if (target.closest('.cmd-item')) {
      e.preventDefault()
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className={cn(disabled && 'cursor-not-allowed opacity-50', className)}
      onMouseDown={handleMouseDown}
    >
      <Command shouldFilter={false} className="bg-transparent">
        <div>
          <CommandInput
            placeholder={placeholder}
            value={searchTerm}
            className="max-h-8"
            aria-label={ariaLabel}
            onFocus={handleFocus}
            onValueChange={(value) => {
              setSearchTerm(value)
              if (!open) setOpen(true)
            }}
          />
          {loading && (
            <div className="absolute top-1/2 right-2 flex -translate-y-1/2 transform items-center">
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          )}
        </div>
        <CommandList hidden={!open}>
          {error && <div className="text-destructive p-4 text-center">{error}</div>}
          {loading && options.length === 0 && (loadingSkeleton || <DefaultLoadingSkeleton />)}
          {!loading &&
            !error &&
            options.length === 0 &&
            (notFound || (
              <CommandEmpty>{noResultsMessage || 'No results found.'}</CommandEmpty>
            ))}
          <CommandGroup>
            {options.map((option, idx) => (
              <React.Fragment key={getOptionValue(option) + `-fragment-${idx}`}>
                <CommandItem
                  key={getOptionValue(option) + `${idx}`}
                  value={getOptionValue(option)}
                  onSelect={handleSelect}
                  onMouseMove={() => onFocus(getOptionValue(option))}
                  className="truncate cmd-item"
                >
                  {renderOption(option)}
                </CommandItem>
                {idx !== options.length - 1 && (
                  <div key={`divider-${idx}`} className="bg-foreground/10 h-[1px]" />
                )}
              </React.Fragment>
            ))}
          </CommandGroup>
        </CommandList>
      </Command>
    </div>
  )
}

function DefaultLoadingSkeleton() {
  return (
    <CommandGroup>
      <CommandItem disabled>
        <div className="flex w-full items-center gap-2">
          <div className="bg-muted h-6 w-6 animate-pulse rounded-full" />
          <div className="flex flex-1 flex-col gap-1">
            <div className="bg-muted h-4 w-24 animate-pulse rounded" />
            <div className="bg-muted h-3 w-16 animate-pulse rounded" />
          </div>
        </div>
      </CommandItem>
    </CommandGroup>
  )
}
