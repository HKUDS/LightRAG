import { PencilIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { PropertyValue as PropertyValueType } from '@/api/lightrag'
import Text from '@/components/ui/Text'

interface PropertyNameProps {
  name: string
}

export const PropertyName = ({ name }: PropertyNameProps) => {
  const { t } = useTranslation()

  const getPropertyNameTranslation = (propName: string) => {
    const translationKey = `graphPanel.propertiesView.node.propertyNames.${propName}`
    const translation = t(translationKey)
    return translation === translationKey ? propName : translation
  }

  return (
    <span className="text-primary/60 tracking-wide whitespace-nowrap">
      {getPropertyNameTranslation(name)}
    </span>
  )
}

interface EditIconProps {
  onClick: () => void
}

export const EditIcon = ({ onClick }: EditIconProps) => (
  <div>
    <PencilIcon
      className="h-3 w-3 text-gray-500 hover:text-gray-700 cursor-pointer"
      onClick={onClick}
    />
  </div>
)

interface PropertyValueProps {
  value: PropertyValueType
  onClick?: () => void
  tooltip?: string
}

export const PropertyValue = ({ value, onClick, tooltip }: PropertyValueProps) => {
  // Convert PropertyValue to display string
  const displayValue = typeof value === 'string' ? value : JSON.stringify(value)

  return (
    <div className="flex items-center gap-1 overflow-hidden">
      <Text
        className="hover:bg-primary/20 rounded p-1 overflow-hidden text-ellipsis whitespace-nowrap"
        tooltipClassName="max-w-80 -translate-x-15"
        text={displayValue}
        tooltip={tooltip || (typeof value === 'string' ? value : JSON.stringify(value, null, 2))}
        side="left"
        onClick={onClick}
      />
    </div>
  )
}
