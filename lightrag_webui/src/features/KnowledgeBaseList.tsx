import * as React from 'react'
import { useTranslation } from 'react-i18next'
import {
  BookOpen,
  Plus,
  Trash2,
  Pencil,
  FileText,
  ChevronRight,
  Loader2
} from 'lucide-react'
import { toast } from 'sonner'

import { cn, errorMessage } from '@/lib/utils'
import { Card } from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import {
  getKnowledgeBases,
  createKnowledgeBase,
  deleteKnowledgeBase,
  renameKnowledgeBase,
  type KnowledgeBaseSummary
} from '@/api/lightrag'

interface KnowledgeBaseListProps {
  /** Currently selected KB id, used to highlight the active card. */
  activeId?: string
  /** Navigate into a knowledge base detail view. */
  onOpen: (kbId: string) => void
}

/**
 * Knowledge base catalogue. Each card is a self-contained workspace that the
 * user can open, create or delete. Deleting prompts for confirmation because
 * drops the underlying data.
 */
export default function KnowledgeBaseList({
  activeId,
  onOpen
}: KnowledgeBaseListProps) {
  const { t } = useTranslation()
  const [items, setItems] = React.useState<KnowledgeBaseSummary[]>([])
  const [loading, setLoading] = React.useState(true)
  const [createOpen, setCreateOpen] = React.useState(false)
  const [newId, setNewId] = React.useState('')
  const [creating, setCreating] = React.useState(false)
  const [pendingDelete, setPendingDelete] = React.useState<
    KnowledgeBaseSummary | null
  >(null)
  const [deleting, setDeleting] = React.useState(false)
  const [renameTarget, setRenameTarget] = React.useState<
    KnowledgeBaseSummary | null
  >(null)
  const [renameValue, setRenameValue] = React.useState('')
  const [renaming, setRenaming] = React.useState(false)
  const [reloadToken, setReloadToken] = React.useState(0)

  React.useEffect(() => {
    let cancelled = false

    const run = async () => {
      setLoading(true)
      try {
        const data = await getKnowledgeBases()
        if (cancelled) return
        setItems(data ?? [])
      } catch (err) {
        if (!cancelled) {
          toast.error(
            errorMessage(err) || t('knowledgeBase.loadError', 'Failed to load knowledge bases')
          )
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void run()
    return () => {
      cancelled = true
    }
  }, [t, reloadToken])

  const handleCreate = async () => {
    const id = newId.trim()
    if (!id) {
      toast.error(t('knowledgeBase.idRequired', 'Knowledge base ID is required'))
      return
    }
    setCreating(true)
    try {
      await createKnowledgeBase({ id })
      toast.success(
        t('knowledgeBase.created', 'Knowledge base "{{id}}" created').replace(
          '{{id}}',
          id
        )
      )
      setNewId('')
      setCreateOpen(false)
      setReloadToken((v) => v + 1)
      onOpen(id)
    } catch (err) {
      toast.error(errorMessage(err) || t('knowledgeBase.createError', 'Failed to create'))
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async () => {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      await deleteKnowledgeBase(pendingDelete.id)
      toast.success(
        t('knowledgeBase.deleted', 'Knowledge base "{{id}}" deleted').replace(
          '{{id}}',
          pendingDelete.id
        )
      )
      setPendingDelete(null)
      setReloadToken((v) => v + 1)
    } catch (err) {
      toast.error(errorMessage(err) || t('knowledgeBase.deleteError', 'Failed to delete'))
    } finally {
      setDeleting(false)
    }
  }

  const handleRename = async () => {
    if (!renameTarget) return
    const name = renameValue.trim()
    if (!name) {
      toast.error(t('knowledgeBase.renameRequired', 'Name must not be empty'))
      return
    }
    setRenaming(true)
    try {
      await renameKnowledgeBase(renameTarget.id, name)
      toast.success(
        t('knowledgeBase.renamed', 'Knowledge base renamed to "{{name}}"').replace(
          '{{name}}',
          name
        )
      )
      setRenameTarget(null)
      setReloadToken((v) => v + 1)
    } catch (err) {
      toast.error(errorMessage(err) || t('knowledgeBase.renameError', 'Failed to rename'))
    } finally {
      setRenaming(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1600px] space-y-5 p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">
              {t('knowledgeBase.title', 'Knowledge Bases')}
            </h2>
            <p className="text-muted-foreground mt-1 text-sm">
              {t(
                'knowledgeBase.subtitle',
                'Each knowledge base is an isolated workspace for your documents and graph.'
              )}
            </p>
          </div>
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="size-4" />
            {t('knowledgeBase.new', 'New Knowledge Base')}
          </Button>
        </div>

      {loading ? (
        <div className="flex items-center justify-center py-16 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          <span className="ml-2">{t('common.loading', 'Loading…')}</span>
        </div>
      ) : items.length === 0 ? (
        <Card variant="glass" className="flex flex-col items-center justify-center gap-3 py-16 text-center">
          <BookOpen className="size-10 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            {t('knowledgeBase.empty', 'No knowledge bases yet. Create your first one.')}
          </p>
          <Button variant="outline" onClick={() => setCreateOpen(true)}>
            <Plus className="size-4" />
            {t('knowledgeBase.new', 'New Knowledge Base')}
          </Button>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((kb) => (
            <Card
              key={kb.id}
              variant="glass"
              interactive
              className={cn(
                'glass-sheen group relative overflow-hidden p-5',
                activeId === kb.id && 'ring-2 ring-cyan-400/60'
              )}
              onClick={() => onOpen(kb.id)}
            >
              <div className="flex items-start gap-3">
                <div className="bg-cyan-500/12 text-cyan-400 ring-cyan-500/22 flex size-11 shrink-0 items-center justify-center rounded-xl ring-1 ring-inset">
                  <BookOpen className="size-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-semibold">{kb.name || kb.id}</p>
                  <p className="text-muted-foreground mt-1 flex items-center gap-1.5 text-xs">
                    <FileText className="size-3.5" />
                    {kb.document_count.toLocaleString()}{' '}
                    {t('knowledgeBase.docs', 'documents')}
                  </p>
                </div>
                <button
                  type="button"
                  aria-label={t('knowledgeBase.rename', 'Rename')}
                  title={t('knowledgeBase.rename', 'Rename')}
                  onClick={(e) => {
                    e.stopPropagation()
                    setRenameTarget(kb)
                    setRenameValue(kb.name || kb.id)
                  }}
                  className="text-muted-foreground hover:text-cyan-400 rounded-md p-1.5 transition-colors"
                >
                  <Pencil className="size-4" />
                </button>
                <button
                  type="button"
                  aria-label={t('knowledgeBase.delete', 'Delete')}
                  onClick={(e) => {
                    e.stopPropagation()
                    setPendingDelete(kb)
                  }}
                  className="text-muted-foreground hover:text-rose-400 rounded-md p-1.5 transition-colors"
                >
                  <Trash2 className="size-4" />
                </button>
              </div>
              <div className="mt-4 flex items-center justify-end text-sm font-medium text-cyan-400">
                {t('knowledgeBase.open', 'Open')}
                <ChevronRight className="size-4 transition-transform group-hover:translate-x-0.5" />
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Create dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('knowledgeBase.new', 'New Knowledge Base')}</DialogTitle>
            <DialogDescription>
              {t(
                'knowledgeBase.newHint',
                'The ID becomes the workspace name. Use letters, numbers, hyphens and underscores.'
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2 py-2">
            <label className="text-sm font-medium">
              {t('knowledgeBase.id', 'Knowledge Base ID')}
            </label>
            <Input
              autoFocus
              value={newId}
              placeholder="safety-standards"
              onChange={(e) => setNewId(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void handleCreate()
              }}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button onClick={() => void handleCreate()} disabled={creating}>
              {creating && <Loader2 className="size-4 animate-spin" />}
              {t('knowledgeBase.create', 'Create')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Rename dialog */}
      <Dialog
        open={!!renameTarget}
        onOpenChange={(o) => {
          if (!o) setRenameTarget(null)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('knowledgeBase.renameTitle', 'Rename knowledge base')}</DialogTitle>
            <DialogDescription>
              {t(
                'knowledgeBase.renameHint',
                'Changes the display name only; the workspace id stays unchanged.'
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2 py-2">
            <label className="text-sm font-medium">
              {t('knowledgeBase.name', 'Name')}
            </label>
            <Input
              autoFocus
              value={renameValue}
              placeholder={renameTarget?.id ?? ''}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void handleRename()
              }}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameTarget(null)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button onClick={() => void handleRename()} disabled={renaming}>
              {renaming && <Loader2 className="size-4 animate-spin" />}
              {t('knowledgeBase.save', 'Save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirm */}
      <Dialog open={!!pendingDelete} onOpenChange={(o) => !o && setPendingDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('knowledgeBase.deleteTitle', 'Delete knowledge base?')}</DialogTitle>
            <DialogDescription>
              {t('knowledgeBase.deleteConfirm', 'This permanently removes all data in "{{id}}".').replace(
                '{{id}}',
                pendingDelete?.id ?? ''
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              {t('common.cancel', 'Cancel')}
            </Button>
            <Button variant="destructive" onClick={() => void handleDelete()} disabled={deleting}>
              {deleting && <Loader2 className="size-4 animate-spin" />}
              {t('knowledgeBase.delete', 'Delete')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
    </div>
  )
}
