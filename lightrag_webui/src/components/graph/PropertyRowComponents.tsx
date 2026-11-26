import { PencilIcon } from 'lucide-react'
import Text from '@/components/ui/Text'
import { useTranslation } from 'react-i18next'

// Shared by the visible label and the edit button's accessible name (and
// mirrored by PropertyEditDialog's title) so a screen reader announces the
// same translated name the user sees. Unknown names fall back to the raw key.
const getPropertyNameTranslation = (t: (key: string) => string, propName: string): string => {
  const translationKey = `graphPanel.propertiesView.node.propertyNames.${propName}`
  const translation = t(translationKey)
  return translation === translationKey ? propName : translation
}

interface PropertyNameProps {
  name: string
}

export const PropertyName = ({ name }: PropertyNameProps) => {
  const { t } = useTranslation()

  return (
    <span className="text-primary/60 tracking-wide whitespace-nowrap">
      {getPropertyNameTranslation(t, name)}
    </span>
  )
}

interface EditIconProps {
  onClick: () => void
  propertyName: string
}

export const EditIcon = ({ onClick, propertyName }: EditIconProps) => {
  const { t } = useTranslation()

  // A native button is the keyboard path into property editing: Enter/Space
  // activate it and Tab reaches it, none of which a clickable svg offers.
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={t('graphPanel.propertiesView.editProperty', {
        property: getPropertyNameTranslation(t, propertyName)
      })}
      className="inline-flex appearance-none rounded-sm border-0 bg-transparent p-0 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
    >
      <PencilIcon className="h-3 w-3 text-gray-500 hover:text-gray-700 cursor-pointer" />
    </button>
  )
}

interface PropertyValueProps {
  value: any
  onClick?: () => void
  tooltip?: string
}

export const PropertyValue = ({ value, onClick, tooltip }: PropertyValueProps) => (
  <div className="flex items-center gap-1 overflow-hidden">
    <Text
      className="hover:bg-primary/20 rounded p-1 overflow-hidden text-ellipsis whitespace-nowrap"
      tooltipClassName="max-w-[min(20rem,calc(100vw-2rem))] -translate-x-15"
      text={value}
      tooltip={tooltip || (typeof value === 'string' ? value : JSON.stringify(value, null, 2))}
      side="left"
      onClick={onClick}
    />
  </div>
)
