import { ChevronDown, ChevronUp } from 'lucide-react'
import { forwardRef, useCallback, useEffect, useState } from 'react'
import { NumericFormat, type NumericFormatProps } from 'react-number-format'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { cn } from '@/lib/utils'

export interface NumberInputProps extends Omit<NumericFormatProps, 'value' | 'onValueChange'> {
  stepper?: number
  thousandSeparator?: string
  placeholder?: string
  defaultValue?: number
  min?: number
  max?: number
  value?: number // Controlled value
  suffix?: string
  prefix?: string
  onValueChange?: (value: number | undefined) => void
  fixedDecimalScale?: boolean
  decimalScale?: number
}

const NumberInput = forwardRef<HTMLInputElement, NumberInputProps>(
  (
    {
      stepper,
      thousandSeparator,
      placeholder,
      defaultValue,
      min = Number.NEGATIVE_INFINITY,
      max = Number.POSITIVE_INFINITY,
      onValueChange,
      fixedDecimalScale = false,
      decimalScale = 0,
      className = undefined,
      suffix,
      prefix,
      value: controlledValue,
      ...props
    },
    ref
  ) => {
    const [value, setValue] = useState<number | undefined>(controlledValue ?? defaultValue)

    const handleIncrement = useCallback(() => {
      setValue((prev) =>
        prev === undefined ? (stepper ?? 1) : Math.min(prev + (stepper ?? 1), max)
      )
    }, [stepper, max])

    const handleDecrement = useCallback(() => {
      setValue((prev) =>
        prev === undefined ? -(stepper ?? 1) : Math.max(prev - (stepper ?? 1), min)
      )
    }, [stepper, min])

    useEffect(() => {
      if (controlledValue !== undefined) {
        setValue(controlledValue)
      }
    }, [controlledValue])

    const handleChange = (values: { value: string; floatValue: number | undefined }) => {
      const newValue = values.floatValue === undefined ? undefined : values.floatValue
      setValue(newValue)
      if (onValueChange) {
        onValueChange(newValue)
      }
    }

    const handleBlur = () => {
      if (value !== undefined) {
        const inputRef = ref as React.RefObject<HTMLInputElement>
        if (value < min) {
          setValue(min)
          if (inputRef.current) inputRef.current.value = String(min)
        } else if (value > max) {
          setValue(max)
          if (inputRef.current) inputRef.current.value = String(max)
        }
      }
    }

    return (
      <div className="relative flex">
        <NumericFormat
          value={value}
          onValueChange={handleChange}
          thousandSeparator={thousandSeparator}
          decimalScale={decimalScale}
          fixedDecimalScale={fixedDecimalScale}
          allowNegative={min < 0}
          valueIsNumericString
          onBlur={handleBlur}
          max={max}
          min={min}
          suffix={suffix}
          prefix={prefix}
          customInput={(props) => <Input {...props} className={cn('w-full', className)} />}
          placeholder={placeholder}
          className="[appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          getInputRef={ref}
          {...props}
        />
        <div className="absolute top-0 right-0 bottom-0 flex flex-col">
          <Button
            aria-label="Increase value"
            className="border-input h-1/2 rounded-l-none rounded-br-none border-b border-l px-2 focus-visible:relative"
            variant="outline"
            onClick={handleIncrement}
            disabled={value === max}
          >
            <ChevronUp size={15} />
          </Button>
          <Button
            aria-label="Decrease value"
            className="border-input h-1/2 rounded-l-none rounded-tr-none border-b border-l px-2 focus-visible:relative"
            variant="outline"
            onClick={handleDecrement}
            disabled={value === min}
          >
            <ChevronDown size={15} />
          </Button>
        </div>
      </div>
    )
  }
)

NumberInput.displayName = 'NumberInput'

export default NumberInput
