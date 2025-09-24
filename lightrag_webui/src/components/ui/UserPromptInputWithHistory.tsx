import React, { useState, useRef, useEffect, useCallback } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import Input from './Input'

interface UserPromptInputWithHistoryProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  id?: string
  history: string[]
  onSelectFromHistory: (prompt: string) => void
}

export default function UserPromptInputWithHistory({
  value,
  onChange,
  placeholder,
  className,
  id,
  history,
  onSelectFromHistory
}: UserPromptInputWithHistoryProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const [isHovered, setIsHovered] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
        setSelectedIndex(-1)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [])

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!isOpen) {
      if (e.key === 'ArrowDown' && history.length > 0) {
        e.preventDefault()
        setIsOpen(true)
        setSelectedIndex(0)
      }
      return
    }

    switch (e.key) {
    case 'ArrowDown':
      e.preventDefault()
      setSelectedIndex(prev =>
        prev < history.length - 1 ? prev + 1 : prev
      )
      break
    case 'ArrowUp':
      e.preventDefault()
      setSelectedIndex(prev => prev > 0 ? prev - 1 : -1)
      if (selectedIndex === 0) {
        setSelectedIndex(-1)
      }
      break
    case 'Enter':
      if (selectedIndex >= 0 && selectedIndex < history.length) {
        e.preventDefault()
        const selectedPrompt = history[selectedIndex]
        onSelectFromHistory(selectedPrompt)
        setIsOpen(false)
        setSelectedIndex(-1)
      }
      break
    case 'Escape':
      e.preventDefault()
      setIsOpen(false)
      setSelectedIndex(-1)
      break
    }
  }, [isOpen, selectedIndex, history, onSelectFromHistory])

  const handleInputClick = () => {
    if (history.length > 0) {
      setIsOpen(!isOpen)
      setSelectedIndex(-1)
    }
  }

  const handleDropdownItemClick = (prompt: string) => {
    onSelectFromHistory(prompt)
    setIsOpen(false)
    setSelectedIndex(-1)
    inputRef.current?.focus()
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value)
  }

  const handleMouseEnter = () => {
    setIsHovered(true)
  }

  const handleMouseLeave = () => {
    setIsHovered(false)
  }

  return (
    <div className="relative" ref={dropdownRef} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
      <div className="relative">
        <Input
          ref={inputRef}
          id={id}
          value={value}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onClick={handleInputClick}
          placeholder={placeholder}
          className={cn(isHovered && history.length > 0 ? 'pr-5' : 'pr-2', 'w-full', className)}
        />
        {isHovered && history.length > 0 && (
          <button
            type="button"
            onClick={handleInputClick}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-0 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            tabIndex={-1}
          >
            <ChevronDown
              className={cn(
                'h-3 w-3 transition-transform duration-200 text-gray-500',
                isOpen && 'rotate-180'
              )}
            />
          </button>
        )}
      </div>

      {/* Dropdown */}
      {isOpen && history.length > 0 && (
        <div className="absolute top-full left-0 right-0 z-50 mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg max-h-60 overflow-auto min-w-0">
          {history.map((prompt, index) => (
            <button
              key={index}
              type="button"
              onClick={() => handleDropdownItemClick(prompt)}
              className={cn(
                'w-full px-3 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
                'border-b border-gray-100 dark:border-gray-700 last:border-b-0',
                'focus:outline-none focus:bg-gray-100 dark:focus:bg-gray-700',
                selectedIndex === index && 'bg-gray-100 dark:bg-gray-700'
              )}
            >
              <div className="truncate" title={prompt}>
                {prompt}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
