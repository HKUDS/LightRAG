import { getPromptConfigVersions, PromptVersionRecord } from '@/api/lightrag'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { useEffect, useState } from 'react'

type RetrievalPromptVersionSelectorProps = {
  enabled: boolean
  value: string
  onChange: (value: string) => void
}

export default function RetrievalPromptVersionSelector({
  enabled,
  value,
  onChange
}: RetrievalPromptVersionSelectorProps) {
  const [versions, setVersions] = useState<PromptVersionRecord[]>([])

  useEffect(() => {
    if (!enabled) {
      setVersions([])
      return
    }

    getPromptConfigVersions('retrieval')
      .then((registry) => setVersions(registry.versions))
      .catch(() => setVersions([]))
  }, [enabled])

  return (
    <Select value={value} onValueChange={onChange} disabled={!enabled}>
      <SelectTrigger className="h-9">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="active">Use active version</SelectItem>
        <SelectItem value="custom">Custom / Draft</SelectItem>
        {versions.map((version) => (
          <SelectItem key={version.version_id} value={version.version_id}>
            {version.version_name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
