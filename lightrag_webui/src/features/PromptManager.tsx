import { useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'

import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import Badge from '@/components/ui/Badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import { cn } from '@/lib/utils'
import { useAuthStore } from '@/stores/state'

import {
  getPromptTemplate,
  listPromptTemplates,
  upsertPromptTemplate,
  PromptTemplateDetail,
  PromptTemplateInfo
} from '@/api/lightrag'

export default function PromptManager() {
  const isGuestMode = useAuthStore((s) => s.isGuestMode)

  const [templates, setTemplates] = useState<PromptTemplateInfo[]>([])
  const [loadingList, setLoadingList] = useState(false)

  const [selected, setSelected] = useState<PromptTemplateDetail | null>(null)
  const [editorValue, setEditorValue] = useState('')
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)

  const [search, setSearch] = useState('')
  const [commitMessage, setCommitMessage] = useState('')

  const filteredTemplates = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return templates
    return templates.filter((t) => `${t.type}-${t.name}`.toLowerCase().includes(q))
  }, [templates, search])

  const refreshList = async () => {
    setLoadingList(true)
    try {
      const data = await listPromptTemplates({ resolved: true })
      setTemplates(data)
    } catch (e: any) {
      toast.error(e?.message || 'Failed to load prompt templates')
    } finally {
      setLoadingList(false)
    }
  }

  const openTemplate = async (info: PromptTemplateInfo) => {
    try {
      const detail = await getPromptTemplate(info.type, info.name)
      setSelected(detail)
      setEditorValue(detail.content)
      setCommitMessage('')
      setDirty(false)
    } catch (e: any) {
      toast.error(e?.message || 'Failed to load template')
    }
  }

  const saveTemplate = async () => {
    if (!selected) return
    setSaving(true)
    try {
      const resp = await upsertPromptTemplate(selected.type, selected.name, {
        content: editorValue,
        commit_message: commitMessage?.trim() || undefined
      })
      setSelected(resp.template)
      setTemplates((prev) =>
        prev.map((t) =>
          t.type === resp.template.type && t.name === resp.template.name
            ? {
                type: resp.template.type,
                name: resp.template.name,
                origin: resp.template.origin,
                file_path: resp.template.file_path
              }
            : t
        )
      )
      setDirty(false)
      toast.success(`Saved ${resp.template.type}-${resp.template.name}`)
    } catch (e: any) {
      const msg = e?.message || 'Failed to save template'
      toast.error(msg)
    } finally {
      setSaving(false)
    }
  }

  useEffect(() => {
    refreshList()
  }, [])

  return (
    <div className="flex h-full w-full overflow-hidden">
      <div className="w-[340px] shrink-0 border-r border-border/60 overflow-auto p-4">
        <div className="flex items-center gap-2 mb-3">
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search templates..."
          />
          <Button
            variant="secondary"
            onClick={refreshList}
            disabled={loadingList}
            tooltip="Refresh"
          >
            Refresh
          </Button>
        </div>

        <div className="space-y-1">
          {filteredTemplates.map((t) => {
            const isActive = selected?.type === t.type && selected?.name === t.name
            return (
              <button
                key={`${t.type}-${t.name}`}
                className={cn(
                  'w-full text-left rounded-md px-2 py-2 border transition-colors',
                  isActive
                    ? 'bg-emerald-50 border-emerald-200 dark:bg-emerald-950 dark:border-emerald-900'
                    : 'bg-background hover:bg-muted/40 border-transparent'
                )}
                onClick={() => openTemplate(t)}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="font-medium text-sm truncate">{`${t.type}-${t.name}`}</div>
                  <Badge variant={t.origin === 'user' ? 'default' : 'secondary'}>{t.origin}</Badge>
                </div>
                <div className="text-xs text-muted-foreground truncate">{t.file_path}</div>
              </button>
            )
          })}

          {!loadingList && filteredTemplates.length === 0 && (
            <div className="text-sm text-muted-foreground">No templates found.</div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4">
        {isGuestMode && (
          <Alert className="mb-4">
            <AlertTitle>Read-only mode</AlertTitle>
            <AlertDescription>
              Prompt editing is disabled when authentication accounts are not configured on the server.
            </AlertDescription>
          </Alert>
        )}

        {!selected ? (
          <div className="text-sm text-muted-foreground">Select a template from the left.</div>
        ) : (
          <div className="flex flex-col gap-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-lg font-semibold">{`${selected.type}-${selected.name}`}</div>
                <div className="text-xs text-muted-foreground">
                  {selected.origin} · {selected.file_path}
                </div>
              </div>
              <Button
                onClick={saveTemplate}
                disabled={saving || !dirty || isGuestMode}
                tooltip={isGuestMode ? 'Admin login required' : dirty ? 'Save' : 'No changes'}
              >
                Save
              </Button>
            </div>

            <Input
              value={commitMessage}
              onChange={(e) => setCommitMessage(e.target.value)}
              placeholder="Commit message (optional)"
            />

            <Textarea
              value={editorValue}
              onChange={(e) => {
                setEditorValue(e.target.value)
                setDirty(true)
              }}
              className="min-h-[520px] font-mono text-xs"
            />
          </div>
        )}
      </div>
    </div>
  )
}
