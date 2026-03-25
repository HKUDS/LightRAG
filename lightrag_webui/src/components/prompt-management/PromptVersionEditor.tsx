import { PromptConfigGroup, PromptVersionCreateRequest, PromptVersionRecord } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { formatVersionLineageLabel, buildPromptEditorSections } from '@/utils/promptVersioning'
import PromptListFieldEditor from './PromptListFieldEditor'
import { useEffect, useMemo, useState } from 'react'

type PromptVersionEditorProps = {
  groupType: PromptConfigGroup
  version: PromptVersionRecord | null
  versionsById: Record<string, PromptVersionRecord>
  activeVersionId: string | null
  onSaveVersion: (payload: PromptVersionCreateRequest) => Promise<void>
  onActivateVersion: (version: PromptVersionRecord) => Promise<void>
  onDeleteVersion: (version: PromptVersionRecord) => Promise<void>
  onShowDiff: (version: PromptVersionRecord) => Promise<void>
}

const getValueAtPath = (payload: Record<string, unknown>, path: string): unknown => {
  return path.split('.').reduce<unknown>((current, segment) => {
    if (!current || typeof current !== 'object') {
      return undefined
    }
    return (current as Record<string, unknown>)[segment]
  }, payload)
}

const setValueAtPath = (
  payload: Record<string, unknown>,
  path: string,
  value: unknown
): Record<string, unknown> => {
  const next = structuredClone(payload)
  const segments = path.split('.')
  let cursor: Record<string, unknown> = next

  segments.forEach((segment, index) => {
    if (index === segments.length - 1) {
      if (value === undefined || value === '') {
        delete cursor[segment]
      } else {
        cursor[segment] = value
      }
      return
    }

    const existing = cursor[segment]
    if (!existing || typeof existing !== 'object' || Array.isArray(existing)) {
      cursor[segment] = {}
    }
    cursor = cursor[segment] as Record<string, unknown>
  })

  return next
}

export default function PromptVersionEditor({
  groupType,
  version,
  versionsById,
  activeVersionId,
  onSaveVersion,
  onActivateVersion,
  onDeleteVersion,
  onShowDiff
}: PromptVersionEditorProps) {
  const sections = useMemo(() => buildPromptEditorSections(groupType), [groupType])
  const [versionName, setVersionName] = useState('')
  const [comment, setComment] = useState('')
  const [payload, setPayload] = useState<Record<string, unknown>>({})
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    setVersionName(version ? `${version.version_name}-copy` : '')
    setComment('')
    setPayload(version ? structuredClone(version.payload) : {})
  }, [version])

  if (!version) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>Version Editor</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Select a version from the left to inspect or save as a new version.
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="h-full">
      <CardHeader className="space-y-3">
        <CardTitle>{version.version_name}</CardTitle>
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-1">
            <label className="text-xs font-medium">Save As Version Name</label>
            <Input value={versionName} onChange={(event) => setVersionName(event.target.value)} />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium">Source</label>
            <div className="rounded-md border px-3 py-2 text-sm">
              {formatVersionLineageLabel(version, versionsById)}
            </div>
          </div>
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium">Comment</label>
          <Textarea
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            className="min-h-[72px]"
          />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            disabled={!versionName.trim() || saving}
            onClick={async () => {
              setSaving(true)
              try {
                await onSaveVersion({
                  version_name: versionName.trim(),
                  comment: comment.trim(),
                  payload,
                  source_version_id: version.version_id
                })
              } finally {
                setSaving(false)
              }
            }}
          >
            Save As New Version
          </Button>
          <Button type="button" variant="outline" onClick={() => onShowDiff(version)}>
            View Diff
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={activeVersionId === version.version_id}
            onClick={() => onActivateVersion(version)}
          >
            Set Active
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={activeVersionId === version.version_id}
            onClick={() => onDeleteVersion(version)}
          >
            Delete
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 overflow-auto">
        {sections.map((section) => {
          const value = getValueAtPath(payload, section.key)
          return (
            <div key={section.key} className="space-y-2">
              <label className="text-xs font-medium">{section.title}</label>
              {section.type === 'list' ? (
                <PromptListFieldEditor
                  value={Array.isArray(value) ? (value as string[]) : []}
                  onChange={(nextValue) => setPayload((current) => setValueAtPath(current, section.key, nextValue))}
                  placeholder={section.title}
                />
              ) : section.type === 'input' ? (
                <Input
                  value={typeof value === 'string' ? value : ''}
                  onChange={(event) =>
                    setPayload((current) => setValueAtPath(current, section.key, event.target.value))
                  }
                />
              ) : (
                <Textarea
                  value={typeof value === 'string' ? value : ''}
                  onChange={(event) =>
                    setPayload((current) => setValueAtPath(current, section.key, event.target.value))
                  }
                  className="min-h-[120px]"
                />
              )}
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}
