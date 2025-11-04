import { useState, useEffect, useCallback } from 'react'
import { Check, ChevronsUpDown, Loader2 } from 'lucide-react'
import { useDebounce } from '@/hooks/useDebounce'

import { cn } from '@/lib/utils'
import Button from '@/components/ui/Button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList
} from '@/components/ui/Command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'

export interface Option {
  value: string
  label: string
  disabled?: boolean
  description?: string
  icon?: React.ReactNode
}

export interface AsyncSelectProps<T> {
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
  /** Function to get the display value for the selected option */
  getDisplayValue: (option: T) => React.ReactNode
  /** Custom not found message */
  notFound?: React.ReactNode
  /** Custom loading skeleton */
  loadingSkeleton?: React.ReactNode
  /** Currently selected value */
  value: string
  /** Callback when selection changes */
  onChange: (value: string) => void
  /** Callback before opening the dropdown (async supported) */
  onBeforeOpen?: () => void | Promise<void>
  /** Accessibility label for the select field */
  ariaLabel?: string
  /** Placeholder text when no selection */
  placeholder?: string
  /** Display text for search placeholder */
  searchPlaceholder?: string
  /** Disable the entire select */
  disabled?: boolean
  /** Custom width for the popover *
  width?: string | number
  /** Custom class names */
  className?: string
  /** Custom trigger button class names */
  triggerClassName?: string
  /** Custom search input class names */
  searchInputClassName?: string
  /** Custom no results message */
  noResultsMessage?: string
  /** Custom trigger tooltip */
  triggerTooltip?: string
  /** Allow clearing the selection */
  clearable?: boolean
  /** Debounce time in milliseconds */
  debounceTime?: number
}

export function AsyncSelect<T>({
  fetcher,
  preload,
  filterFn,
  renderOption,
  getOptionValue,
  getDisplayValue,
  notFound,
  loadingSkeleton,
  ariaLabel,
  placeholder = 'Select...',
  searchPlaceholder,
  value,
  onChange,
  onBeforeOpen,
  disabled = false,
  className,
  triggerClassName,
  searchInputClassName,
  noResultsMessage,
  triggerTooltip,
  clearable = true,
  debounceTime = 150
}: AsyncSelectProps<T>) {
  const [mounted, setMounted] = useState(false)
  const [open, setOpen] = useState(false)
  const [options, setOptions] = useState<T[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedValue, setSelectedValue] = useState(value)
  const [selectedOption, setSelectedOption] = useState<T | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const debouncedSearchTerm = useDebounce(searchTerm, preload ? 0 : debounceTime)
  const [originalOptions, setOriginalOptions] = useState<T[]>([])
  const [initialValueDisplay, setInitialValueDisplay] = useState<React.ReactNode | null>(null)

  useEffect(() => {
    setMounted(true)
    setSelectedValue(value)
  }, [value])

  // Add an effect to handle initial value display
  useEffect(() => {
    if (value && (!options.length || !selectedOption)) {
      // Create a temporary display until options are loaded
      setInitialValueDisplay(<div>{value}</div>)
    } else if (selectedOption) {
      // Once we find the actual selectedOption, clear the temporary display
      setInitialValueDisplay(null)
    }
  }, [value, options.length, selectedOption])

  // Initialize selectedOption when options are loaded and value exists
  useEffect(() => {
    if (value && options.length > 0) {
      const option = options.find((opt) => getOptionValue(opt) === value)
      if (option) {
        setSelectedOption(option)
      }
    }
  }, [value, options, getOptionValue])

  // Effect for initial fetch
  useEffect(() => {
    const initializeOptions = async () => {
      try {
        setLoading(true)
        setError(null)
        // Always use empty query for initial load to show search history
        const data = await fetcher('')
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
  }, [mounted, fetcher])

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
      const newValue = clearable && currentValue === selectedValue ? '' : currentValue
      setSelectedValue(newValue)
      setSelectedOption(options.find((option) => getOptionValue(option) === newValue) || null)
      onChange(newValue)
      setOpen(false)
    },
    [selectedValue, onChange, clearable, options, getOptionValue]
  )

  const handleOpenChange = useCallback(
    async (newOpen: boolean) => {
      if (newOpen && onBeforeOpen) {
        await onBeforeOpen()
      }
      setOpen(newOpen)
    },
    [onBeforeOpen]
  )

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          aria-label={ariaLabel}
          className={cn(
            'justify-between',
            disabled && 'cursor-not-allowed opacity-50',
            triggerClassName
          )}
          disabled={disabled}
          tooltip={triggerTooltip}
          side="bottom"
        >
          {value === '*' ? <div>*</div> : (selectedOption ? getDisplayValue(selectedOption) : (initialValueDisplay || placeholder))}
          <ChevronsUpDown className="opacity-50" size={10} />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className={cn('p-0', className)}
        onCloseAutoFocus={(e) => e.preventDefault()}
        align="start"
        sideOffset={8}
        collisionPadding={5}
      >
        <Command shouldFilter={false}>
          <div className="relative w-full border-b">
            <CommandInput
              placeholder={searchPlaceholder || 'Search...'}
              value={searchTerm}
              onValueChange={(value) => {
                setSearchTerm(value)
              }}
              className={searchInputClassName}
            />
            {loading && options.length > 0 && (
              <div className="absolute top-1/2 right-2 flex -translate-y-1/2 transform items-center">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
            )}
          </div>
          <CommandList>
            {error && <div className="text-destructive p-4 text-center">{error}</div>}
            {loading && options.length === 0 && (loadingSkeleton || <DefaultLoadingSkeleton />)}
            {!loading &&
              !error &&
              options.length === 0 &&
              (notFound || (
                <CommandEmpty>
                  {noResultsMessage || 'No results found.'}
                </CommandEmpty>
              ))}
            <CommandGroup>
              {options.map((option) => {
                const optionValue = getOptionValue(option);
                // Fix cmdk filtering issue: use empty string when search is empty
                // This ensures all items are shown when searchTerm is empty
                const itemValue = searchTerm.trim() === '' ? '' : optionValue;

                return (
                  <CommandItem
                    key={optionValue}
                    value={itemValue}
                    onSelect={() => {
                      handleSelect(optionValue);
                    }}
                    className="truncate"
                  >
                    {renderOption(option)}
                    <Check
                      className={cn(
                        'ml-auto h-3 w-3',
                        selectedValue === optionValue ? 'opacity-100' : 'opacity-0'
                      )}
                    />
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
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
