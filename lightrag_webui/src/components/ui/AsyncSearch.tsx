import { useState, useEffect, useCallback } from 'react'
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
  /** Label for the select field */
  label: string
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
  label,
  placeholder = 'Select...',
  value,
  onChange,
  onFocus,
  disabled = false,
  className,
  noResultsMessage
}: AsyncSearchProps<T>) {
  const [mounted, setMounted] = useState(false)
  const [open, setOpen] = useState(false)
  const [options, setOptions] = useState<T[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedValue, setSelectedValue] = useState(value)
  const [focusedValue, setFocusedValue] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const debouncedSearchTerm = useDebounce(searchTerm, preload ? 0 : 150)
  const [originalOptions, setOriginalOptions] = useState<T[]>([])

  useEffect(() => {
    setMounted(true)
    setSelectedValue(value)
  }, [value])

  // Effect for initial fetch
  useEffect(() => {
    const initializeOptions = async () => {
      try {
        setLoading(true)
        setError(null)
        // If we have a value, use it for the initial search
        const data = value !== null ? await fetcher(value) : []
        setOriginalOptions(data)
        setOptions(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch options')
      } finally {
        setLoading(false)
      }
    }

    if (!mounted) {
      initializeOptions()
    }
  }, [mounted, fetcher, value])

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        setLoading(true)
        setError(null)
        const data = await fetcher(debouncedSearchTerm)
        setOriginalOptions(data)
        setOptions(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch options')
      } finally {
        setLoading(false)
      }
    }

    if (!mounted) {
      fetchOptions()
    } else if (!preload) {
      fetchOptions()
    } else if (preload) {
      if (debouncedSearchTerm) {
        setOptions(
          originalOptions.filter((option) =>
            filterFn ? filterFn(option, debouncedSearchTerm) : true
          )
        )
      } else {
        setOptions(originalOptions)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetcher, debouncedSearchTerm, mounted, preload, filterFn])

  const handleSelect = useCallback(
    (currentValue: string) => {
      if (currentValue !== selectedValue) {
        setSelectedValue(currentValue)
        onChange(currentValue)
      }
      setOpen(false)
    },
    [selectedValue, setSelectedValue, setOpen, onChange]
  )

  const handleFocus = useCallback(
    (currentValue: string) => {
      if (currentValue !== focusedValue) {
        setFocusedValue(currentValue)
        onFocus(currentValue)
      }
    },
    [focusedValue, setFocusedValue, onFocus]
  )

  return (
    <div
      className={cn(disabled && 'cursor-not-allowed opacity-50', className)}
      onFocus={() => {
        setOpen(true)
      }}
      onBlur={() => setOpen(false)}
    >
      <Command shouldFilter={false} className="bg-transparent">
        <div>
          <CommandInput
            placeholder={placeholder}
            value={searchTerm}
            className="max-h-8"
            onValueChange={(value) => {
              setSearchTerm(value)
              if (value && !open) setOpen(true)
            }}
          />
          {loading && options.length > 0 && (
            <div className="absolute top-1/2 right-2 flex -translate-y-1/2 transform items-center">
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          )}
        </div>
        <CommandList hidden={!open || debouncedSearchTerm.length === 0}>
          {error && <div className="text-destructive p-4 text-center">{error}</div>}
          {loading && options.length === 0 && (loadingSkeleton || <DefaultLoadingSkeleton />)}
          {!loading &&
            !error &&
            options.length === 0 &&
            (notFound || (
              <CommandEmpty>{noResultsMessage ?? `No ${label.toLowerCase()} found.`}</CommandEmpty>
            ))}
          <CommandGroup>
            {options.map((option, idx) => (
              <>
                <CommandItem
                  key={getOptionValue(option) + `${idx}`}
                  value={getOptionValue(option)}
                  onSelect={handleSelect}
                  onMouseEnter={() => handleFocus(getOptionValue(option))}
                  className="truncate"
                >
                  {renderOption(option)}
                </CommandItem>
                {idx !== options.length - 1 && (
                  <div key={idx} className="bg-foreground/10 h-[1px]" />
                )}
              </>
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
