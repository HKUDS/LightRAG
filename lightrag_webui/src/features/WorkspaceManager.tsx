import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  FolderKanbanIcon,
  MoreHorizontalIcon,
  PencilIcon,
  PlusIcon,
  RefreshCwIcon,
  Trash2Icon
} from 'lucide-react'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/Card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle
} from '@/components/ui/AlertDialog'
import { createWorkspace, deleteWorkspace, getWorkspaces, updateWorkspace } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useWorkspaceStore, type Workspace } from '@/stores/workspace'
import SiteHeader from '@/features/SiteHeader'

type WorkspaceForm = {
  displayName: string
  description: string
}

const emptyForm: WorkspaceForm = { displayName: '', description: '' }

const formatDate = (value: string, locale: string): string => {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(locale)
}

const workspaceRoute = (workspaceId: string): string => `/workspaces/${workspaceId}`

export default function WorkspaceManager() {
  const { t, i18n } = useTranslation()
  const navigate = useNavigate()
  const clearSelectedWorkspace = useWorkspaceStore.use.clearSelectedWorkspace()
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [editingWorkspace, setEditingWorkspace] = useState<Workspace | null>(null)
  const [deletingWorkspace, setDeletingWorkspace] = useState<Workspace | null>(null)
  const [deleteConfirmation, setDeleteConfirmation] = useState('')
  const [form, setForm] = useState<WorkspaceForm>(emptyForm)

  const refreshWorkspaces = useCallback(
    async (showSuccess: boolean = false) => {
      try {
        setIsLoading(true)
        const items = await getWorkspaces()
        setWorkspaces(items)
        if (showSuccess) toast.success(t('workspace.refreshSuccess'))
      } catch (error) {
        toast.error(t('workspace.loadFailed', { error: errorMessage(error) }))
      } finally {
        setIsLoading(false)
      }
    },
    [t]
  )

  useEffect(() => {
    clearSelectedWorkspace()
    const timer = window.setTimeout(() => void refreshWorkspaces(), 0)
    return () => window.clearTimeout(timer)
  }, [clearSelectedWorkspace, refreshWorkspaces])

  const isDeleting = useMemo(
    () => workspaces.some((workspace) => workspace.status === 'deleting'),
    [workspaces]
  )

  useEffect(() => {
    if (!isDeleting) return
    const timer = window.setTimeout(() => void refreshWorkspaces(), 1500)
    return () => window.clearTimeout(timer)
  }, [isDeleting, refreshWorkspaces])

  const openCreate = () => {
    setForm(emptyForm)
    setCreateOpen(true)
  }

  const openEdit = (workspace: Workspace) => {
    setForm({
      displayName: workspace.display_name,
      description: workspace.description ?? ''
    })
    setEditingWorkspace(workspace)
  }

  const saveWorkspace = async () => {
    const displayName = form.displayName.trim()
    if (!displayName) {
      toast.error(t('workspace.nameRequired'))
      return
    }

    setIsSaving(true)
    try {
      const request = {
        display_name: displayName,
        description: form.description.trim() || null
      }
      const workspace = editingWorkspace
        ? await updateWorkspace(editingWorkspace.id, request)
        : await createWorkspace(request)

      await refreshWorkspaces()
      setCreateOpen(false)
      setEditingWorkspace(null)
      toast.success(editingWorkspace ? t('workspace.updateSuccess') : t('workspace.createSuccess'))

      if (!editingWorkspace) {
        navigate(workspaceRoute(workspace.id))
      }
    } catch (error) {
      toast.error(t('workspace.saveFailed', { error: errorMessage(error) }))
    } finally {
      setIsSaving(false)
    }
  }

  const scheduleDelete = async () => {
    if (!deletingWorkspace) return
    try {
      const response = await deleteWorkspace(deletingWorkspace.id)
      toast.success(response.message)
      setDeletingWorkspace(null)
      setDeleteConfirmation('')
      await refreshWorkspaces()
    } catch (error) {
      toast.error(t('workspace.deleteFailed', { error: errorMessage(error) }))
    }
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <SiteHeader showTabs={false} />
      <div className="bg-muted/20 flex-1 overflow-auto">
        <div className="mx-auto flex min-h-full w-full max-w-6xl flex-col gap-7 px-6 py-8 lg:px-10">
          <section className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div className="space-y-2">
              <div className="text-primary flex items-center gap-2">
                <FolderKanbanIcon className="size-5" aria-hidden="true" />
                <span className="text-sm font-medium">{t('workspace.eyebrow')}</span>
              </div>
              <h1 className="text-3xl font-semibold tracking-tight">{t('workspace.title')}</h1>
              <p className="text-muted-foreground max-w-2xl">{t('workspace.description')}</p>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => void refreshWorkspaces(true)}
                disabled={isLoading}
              >
                <RefreshCwIcon className={isLoading ? 'animate-spin' : ''} />
                {t('workspace.refresh')}
              </Button>
              <Button size="sm" onClick={openCreate}>
                <PlusIcon />
                {t('workspace.create')}
              </Button>
            </div>
          </section>

          {isLoading && workspaces.length === 0 ? (
            <div className="text-muted-foreground flex flex-1 items-center justify-center py-24">
              <RefreshCwIcon className="mr-2 size-4 animate-spin" />
              {t('workspace.loading')}
            </div>
          ) : (
            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {workspaces.map((workspace) => {
                const isAvailable = workspace.status === 'active'
                return (
                  <Card
                    key={workspace.id}
                    className={`group flex min-h-56 flex-col transition-shadow ${
                      isAvailable ? 'cursor-pointer hover:shadow-md' : 'opacity-70'
                    }`}
                    onClick={() => isAvailable && navigate(workspaceRoute(workspace.id))}
                  >
                    <CardHeader className="gap-3">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex min-w-0 items-center gap-3">
                          <div className="bg-primary/10 text-primary rounded-lg p-2">
                            <FolderKanbanIcon className="size-5" aria-hidden="true" />
                          </div>
                          <div className="min-w-0">
                            <CardTitle className="truncate">{workspace.display_name}</CardTitle>
                            <p className="text-muted-foreground mt-1 text-xs">
                              {workspace.is_default
                                ? t('workspace.defaultWorkspace')
                                : t('workspace.workspace')}
                            </p>
                          </div>
                        </div>
                        <span
                          className={`rounded-full px-2 py-1 text-xs font-medium ${
                            workspace.status === 'active'
                              ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                              : 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300'
                          }`}
                        >
                          {t(`workspace.status.${workspace.status}`)}
                        </span>
                      </div>
                      <CardDescription className="line-clamp-3 min-h-10">
                        {workspace.description || t('workspace.noDescription')}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="flex-1">
                      <p className="text-muted-foreground text-xs">
                        {t('workspace.updatedAt', {
                          date: formatDate(workspace.updated_at, i18n.language)
                        })}
                      </p>
                    </CardContent>
                    <CardFooter className="justify-between gap-2 border-t pt-4">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(event) => {
                          event.stopPropagation()
                          openEdit(workspace)
                        }}
                        disabled={workspace.status !== 'active'}
                      >
                        <PencilIcon />
                        {t('workspace.edit')}
                      </Button>
                      <div className="flex items-center gap-2">
                        {!workspace.is_default && (
                          <Button
                            variant="ghost"
                            size="icon"
                            tooltip={t('workspace.delete')}
                            onClick={(event) => {
                              event.stopPropagation()
                              setDeleteConfirmation('')
                              setDeletingWorkspace(workspace)
                            }}
                            disabled={workspace.status !== 'active'}
                          >
                            <Trash2Icon className="text-destructive" />
                          </Button>
                        )}
                        {isAvailable ? (
                          <Button
                            variant="link"
                            size="sm"
                            className="h-auto px-0 py-0"
                            onClick={(event) => {
                              event.stopPropagation()
                              navigate(workspaceRoute(workspace.id))
                            }}
                          >
                            {t('workspace.open')}
                          </Button>
                        ) : (
                          <MoreHorizontalIcon className="size-4" />
                        )}
                      </div>
                    </CardFooter>
                  </Card>
                )
              })}
            </section>
          )}
        </div>
      </div>

      <Dialog
        open={createOpen || editingWorkspace !== null}
        onOpenChange={(open) => {
          if (!open && !isSaving) {
            setCreateOpen(false)
            setEditingWorkspace(null)
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {editingWorkspace ? t('workspace.editTitle') : t('workspace.createTitle')}
            </DialogTitle>
            <DialogDescription>{t('workspace.formDescription')}</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-2">
            <label className="grid gap-2 text-sm font-medium">
              {t('workspace.name')}
              <Input
                value={form.displayName}
                maxLength={120}
                autoFocus
                onChange={(event) =>
                  setForm((current) => ({ ...current, displayName: event.target.value }))
                }
              />
            </label>
            <label className="grid gap-2 text-sm font-medium">
              {t('workspace.descriptionLabel')}
              <Textarea
                value={form.description}
                maxLength={1000}
                rows={4}
                onChange={(event) =>
                  setForm((current) => ({ ...current, description: event.target.value }))
                }
              />
            </label>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setCreateOpen(false)
                setEditingWorkspace(null)
              }}
              disabled={isSaving}
            >
              {t('common.cancel')}
            </Button>
            <Button onClick={() => void saveWorkspace()} disabled={isSaving}>
              {isSaving ? t('common.saving') : t('common.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={deletingWorkspace !== null}
        onOpenChange={(open) => {
          if (!open) {
            setDeletingWorkspace(null)
            setDeleteConfirmation('')
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('workspace.deleteTitle')}</AlertDialogTitle>
            <AlertDialogDescription>
              {t('workspace.deleteDescription', { name: deletingWorkspace?.display_name })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <label className="grid gap-2 text-sm font-medium">
            {t('workspace.deletePrompt', { name: deletingWorkspace?.display_name })}
            <Input
              value={deleteConfirmation}
              onChange={(event) => setDeleteConfirmation(event.target.value)}
            />
          </label>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={deleteConfirmation !== deletingWorkspace?.display_name}
              onClick={(event) => {
                event.preventDefault()
                void scheduleDelete()
              }}
            >
              {t('workspace.delete')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
