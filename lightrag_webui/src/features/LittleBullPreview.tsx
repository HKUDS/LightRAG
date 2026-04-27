import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ActivityIcon,
  BotIcon,
  CheckCircle2Icon,
  FileTextIcon,
  FolderOpenIcon,
  HomeIcon,
  Loader2Icon,
  LockIcon,
  MessageCircleIcon,
  RefreshCwIcon,
  SearchIcon,
  SettingsIcon,
  ShieldCheckIcon,
  Trash2Icon,
  UploadCloudIcon,
  type LucideIcon
} from 'lucide-react'
import { toast } from 'sonner'
import {
  approveLittleBullApproval,
  deleteLittleBullDocument,
  getLittleBullActivity,
  getLittleBullApprovals,
  getLittleBullAreas,
  getLittleBullAssistants,
  getLittleBullAuditEvents,
  getLittleBullDocuments,
  getLittleBullMe,
  queryLittleBull,
  rejectLittleBullApproval,
  uploadLittleBullDocument,
  type LittleBullActivityItem,
  type LittleBullArea,
  type LittleBullApproval,
  type LittleBullAssistant,
  type LittleBullAuditEvent,
  type LittleBullDocument,
  type LittleBullPrincipal,
  type QueryMode
} from '@/api/lightrag'
import { cn, errorMessage } from '@/lib/utils'

type Page = 'inicio' | 'perguntar' | 'conhecimento' | 'areas' | 'assistentes' | 'atividade' | 'admin'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  references?: Array<Record<string, any>>
}

const pageLabels: Record<Page, string> = {
  inicio: 'Início',
  perguntar: 'Perguntar',
  conhecimento: 'Conhecimento',
  areas: 'Áreas',
  assistentes: 'Assistentes',
  atividade: 'Atividade',
  admin: 'Admin'
}

const navItems: Array<{ id: Page; icon: LucideIcon }> = [
  { id: 'inicio', icon: HomeIcon },
  { id: 'perguntar', icon: MessageCircleIcon },
  { id: 'conhecimento', icon: FolderOpenIcon },
  { id: 'areas', icon: FolderOpenIcon },
  { id: 'assistentes', icon: BotIcon },
  { id: 'atividade', icon: ActivityIcon },
  { id: 'admin', icon: SettingsIcon }
]

const permissionMap = {
  readDocuments: 'little_bull.documents.read',
  uploadDocuments: 'little_bull.documents.upload',
  deleteDocuments: 'little_bull.documents.delete',
  query: 'little_bull.query',
  decideApprovals: 'little_bull.approvals.decide',
  readAudit: 'little_bull.audit.read'
}

const hasPermission = (principal: LittleBullPrincipal | null, permission: string) => {
  if (!principal) return false
  return principal.is_master_global || principal.permissions.includes('*') || principal.permissions.includes(permission)
}

const formatDate = (value?: string | null) => {
  if (!value) return 'Sem data'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('pt-BR', {
    dateStyle: 'short',
    timeStyle: 'short'
  }).format(date)
}

const statusLabel = (status: string) => {
  const normalized = status.toLowerCase()
  if (normalized.includes('processed')) return 'Processado'
  if (normalized.includes('processing')) return 'Processando'
  if (normalized.includes('pending')) return 'Pendente'
  if (normalized.includes('failed')) return 'Falhou'
  return status || 'Desconhecido'
}

function IconButton({
  children,
  onClick,
  disabled,
  tone = 'dark'
}: {
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  tone?: 'dark' | 'light' | 'danger' | 'blue'
}) {
  const tones = {
    dark: 'bg-slate-950 text-white hover:bg-slate-800',
    light: 'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50',
    danger: 'bg-red-600 text-white hover:bg-red-700',
    blue: 'bg-blue-600 text-white hover:bg-blue-700'
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'inline-flex h-10 items-center justify-center gap-2 rounded-lg px-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50',
        tones[tone]
      )}
    >
      {children}
    </button>
  )
}

function Stat({ label, value, helper }: { label: string; value: string; helper: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-semibold text-slate-950">{value}</p>
      <p className="mt-1 text-sm text-slate-500">{helper}</p>
    </div>
  )
}

function EmptyState({ icon: Icon, label }: { icon: LucideIcon; label: string }) {
  return (
    <div className="grid min-h-32 place-items-center rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500">
      <div>
        <Icon className="mx-auto mb-2 size-6 text-slate-400" />
        {label}
      </div>
    </div>
  )
}

function Shell({
  page,
  setPage,
  areas,
  activeWorkspaceId,
  setActiveWorkspaceId,
  principal,
  searchText,
  setSearchText,
  onSearchSubmit,
  children
}: {
  page: Page
  setPage: (page: Page) => void
  areas: LittleBullArea[]
  activeWorkspaceId: string
  setActiveWorkspaceId: (workspaceId: string) => void
  principal: LittleBullPrincipal | null
  searchText: string
  setSearchText: (value: string) => void
  onSearchSubmit: () => void
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <div className="flex min-h-screen">
        <aside className="hidden w-72 shrink-0 border-r border-slate-200 bg-white p-4 lg:block">
          <div className="flex items-center gap-3 rounded-lg bg-slate-950 p-3 text-white">
            <ShieldCheckIcon className="size-6 text-yellow-300" />
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-yellow-300">Little Bull</p>
              <h1 className="font-semibold">Knowledge</h1>
            </div>
          </div>
          <nav className="mt-5 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const selected = page === item.id
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setPage(item.id)}
                  className={cn(
                    'flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm font-semibold transition',
                    selected ? 'bg-blue-600 text-white' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-950'
                  )}
                >
                  <Icon className="size-4" />
                  {pageLabels[item.id]}
                </button>
              )
            })}
          </nav>
        </aside>

        <main className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 px-4 py-3 backdrop-blur">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex min-w-0 flex-1 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3">
                <SearchIcon className="size-4 shrink-0 text-slate-400" />
                <input
                  value={searchText}
                  onChange={(event) => setSearchText(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') onSearchSubmit()
                  }}
                  className="h-10 min-w-0 flex-1 bg-transparent text-sm outline-none"
                  placeholder="Perguntar ao workspace ativo"
                />
              </div>
              <select
                value={activeWorkspaceId}
                onChange={(event) => setActiveWorkspaceId(event.target.value)}
                className="h-10 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold outline-none"
                aria-label="Selecionar workspace"
              >
                {areas.map((area) => (
                  <option key={area.id} value={area.id}>
                    {area.label}
                  </option>
                ))}
              </select>
              <div className="hidden rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600 md:block">
                {principal?.sub ?? 'Usuário'} · {principal?.is_master_global ? 'MASTER' : principal?.roles.join(', ')}
              </div>
            </div>
          </header>

          <div className="flex-1 p-4">{children}</div>

          <nav className="sticky bottom-0 grid grid-cols-7 border-t border-slate-200 bg-white p-1 lg:hidden">
            {navItems.map((item) => {
              const Icon = item.icon
              const selected = page === item.id
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setPage(item.id)}
                  className={cn(
                    'grid min-h-14 place-items-center rounded-lg text-[10px] font-semibold',
                    selected ? 'bg-slate-950 text-white' : 'text-slate-500'
                  )}
                >
                  <Icon className="size-5" />
                  <span className="hidden sm:block">{pageLabels[item.id]}</span>
                </button>
              )
            })}
          </nav>
        </main>
      </div>
    </div>
  )
}

function HomePage({
  areas,
  docs,
  activity,
  setPage,
  setActiveWorkspaceId
}: {
  areas: LittleBullArea[]
  docs: LittleBullDocument[]
  activity: LittleBullActivityItem[]
  setPage: (page: Page) => void
  setActiveWorkspaceId: (workspaceId: string) => void
}) {
  const processed = docs.filter((doc) => doc.status.toLowerCase().includes('processed')).length
  const processing = docs.filter((doc) => doc.status.toLowerCase().includes('processing') || doc.status.toLowerCase().includes('pending')).length

  return (
    <div className="space-y-4">
      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-blue-600">Operação local-first</p>
            <h2 className="mt-2 text-3xl font-semibold tracking-tight">Little Bull funcional sobre LightRAG</h2>
          </div>
          <IconButton onClick={() => setPage('conhecimento')}>
            <UploadCloudIcon className="size-4" />
            Enviar arquivo
          </IconButton>
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-4">
          <Stat label="Áreas" value={String(areas.length)} helper="Workspaces permitidos" />
          <Stat label="Documentos" value={String(docs.length)} helper={`${processed} processados`} />
          <Stat label="Fila" value={String(processing)} helper="Pendente ou processando" />
          <Stat label="Eventos" value={String(activity.length)} helper="Auditoria recente" />
        </div>
      </section>

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {areas.map((area) => (
          <button
            key={area.id}
            type="button"
            onClick={() => {
              setActiveWorkspaceId(area.id)
              setPage('perguntar')
            }}
            className="rounded-lg border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-blue-300 hover:shadow"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="grid size-11 place-items-center rounded-lg text-xl" style={{ backgroundColor: `${area.accent}22` }}>
                {area.emoji}
              </span>
              <span className="rounded-lg bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-600">
                {area.privacy}
              </span>
            </div>
            <h3 className="mt-4 text-lg font-semibold">{area.label}</h3>
            <p className="mt-1 min-h-10 text-sm text-slate-500">{area.description || 'Workspace ativo'}</p>
            <p className="mt-4 text-sm font-semibold text-slate-700">{area.document_count} documentos</p>
          </button>
        ))}
      </section>
    </div>
  )
}

function AskPage({
  activeWorkspaceId,
  prompt,
  setPrompt,
  messages,
  setMessages,
  refreshActivity
}: {
  activeWorkspaceId: string
  prompt: string
  setPrompt: (value: string) => void
  messages: ChatMessage[]
  setMessages: (messages: ChatMessage[]) => void
  refreshActivity: () => Promise<void>
}) {
  const [mode, setMode] = useState<QueryMode>('mix')
  const [confidentiality, setConfidentiality] = useState<'normal' | 'sensivel' | 'privado'>('normal')
  const [modelProfile, setModelProfile] = useState('equilibrado')
  const [loading, setLoading] = useState(false)

  const submit = async () => {
    const query = prompt.trim()
    if (!query) return
    setLoading(true)
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: query }
    setMessages([...messages, userMessage])
    setPrompt('')
    try {
      const response = await queryLittleBull({
        workspace_id: activeWorkspaceId,
        query,
        mode,
        confidentiality,
        model_profile: modelProfile,
        include_references: true
      })
      setMessages([
        ...messages,
        userMessage,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response.response,
          references: response.references
        }
      ])
      await refreshActivity()
    } catch (error) {
      toast.error(errorMessage(error))
      setMessages(messages)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
      <section className="flex min-h-[70vh] flex-col rounded-lg border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-200 p-4">
          <p className="text-sm font-semibold uppercase tracking-wide text-blue-600">Perguntar</p>
          <h2 className="mt-1 text-2xl font-semibold">Resposta com fontes do workspace ativo</h2>
        </div>
        <div className="flex-1 space-y-4 overflow-auto p-4">
          {messages.length === 0 ? (
            <EmptyState icon={MessageCircleIcon} label="Faça a primeira pergunta para este workspace." />
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  'max-w-3xl rounded-lg p-4 text-sm leading-6',
                  message.role === 'user'
                    ? 'ml-auto bg-slate-950 text-white'
                    : 'border border-slate-200 bg-slate-50 text-slate-900'
                )}
              >
                {message.content}
                {!!message.references?.length && (
                  <div className="mt-4 space-y-2">
                    {message.references.map((reference, index) => (
                      <div key={`${reference.reference_id ?? index}`} className="rounded-lg bg-white p-3 text-slate-700 shadow-sm">
                        <p className="font-semibold">Fonte {reference.reference_id ?? index + 1}</p>
                        <p className="text-xs text-slate-500">{reference.file_path ?? 'Sem arquivo'}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
        <div className="border-t border-slate-200 p-4">
          <div className="mb-3 grid gap-2 md:grid-cols-3">
            <select value={mode} onChange={(event) => setMode(event.target.value as QueryMode)} className="h-10 rounded-lg border border-slate-200 px-3 text-sm">
              {['mix', 'hybrid', 'local', 'global', 'naive', 'bypass'].map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
            <select value={confidentiality} onChange={(event) => setConfidentiality(event.target.value as 'normal' | 'sensivel' | 'privado')} className="h-10 rounded-lg border border-slate-200 px-3 text-sm">
              <option value="normal">Normal</option>
              <option value="sensivel">Sensível</option>
              <option value="privado">Privado</option>
            </select>
            <select value={modelProfile} onChange={(event) => setModelProfile(event.target.value)} className="h-10 rounded-lg border border-slate-200 px-3 text-sm">
              <option value="equilibrado">Equilibrado</option>
              <option value="rapido">Rápido</option>
              <option value="privado">Privado/local</option>
            </select>
          </div>
          <div className="flex gap-2">
            <input
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') submit()
              }}
              className="h-11 min-w-0 flex-1 rounded-lg border border-slate-200 px-3 text-sm outline-none focus:border-blue-400"
              placeholder="Escreva sua pergunta"
            />
            <IconButton onClick={submit} disabled={loading} tone="blue">
              {loading ? <Loader2Icon className="size-4 animate-spin" /> : <MessageCircleIcon className="size-4" />}
              Enviar
            </IconButton>
          </div>
        </div>
      </section>

      <aside className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="flex items-center gap-2 font-semibold">
          <LockIcon className="size-4 text-blue-600" />
          Política ativa
        </h3>
        <div className="mt-4 space-y-3 text-sm text-slate-600">
          <p>Dados sensíveis ou privados exigem o perfil Privado/local.</p>
          <p>Consultas registram ator, workspace, modelo e resultado na auditoria.</p>
        </div>
      </aside>
    </div>
  )
}

function KnowledgePage({
  activeWorkspaceId,
  docs,
  setDocs,
  principal,
  refreshActivity
}: {
  activeWorkspaceId: string
  docs: LittleBullDocument[]
  setDocs: (docs: LittleBullDocument[]) => void
  principal: LittleBullPrincipal | null
  refreshActivity: () => Promise<void>
}) {
  const [loading, setLoading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})
  const canUpload = hasPermission(principal, permissionMap.uploadDocuments)
  const canDelete = hasPermission(principal, permissionMap.deleteDocuments)

  const refreshDocuments = useCallback(async () => {
    if (!activeWorkspaceId) return
    const response = await getLittleBullDocuments(activeWorkspaceId)
    setDocs(response.documents)
  }, [activeWorkspaceId, setDocs])

  const uploadFiles = async (files: FileList | null) => {
    if (!files?.length || !canUpload) return
    setLoading(true)
    try {
      for (const file of Array.from(files)) {
        await uploadLittleBullDocument(activeWorkspaceId, file, 'normal', (percent) => {
          setUploadProgress((current) => ({ ...current, [file.name]: percent }))
        })
      }
      toast.success('Upload enviado para processamento')
      await refreshDocuments()
      await refreshActivity()
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setLoading(false)
    }
  }

  const requestDelete = async (documentId: string) => {
    if (!canDelete) return
    try {
      const response = await deleteLittleBullDocument(activeWorkspaceId, documentId)
      toast.info(response.message ?? 'Ação registrada')
      await refreshActivity()
    } catch (error) {
      toast.error(errorMessage(error))
    }
  }

  return (
    <div className="space-y-4">
      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-blue-600">Conhecimento</p>
            <h2 className="mt-1 text-2xl font-semibold">Documentos reais do LightRAG</h2>
          </div>
          <IconButton onClick={refreshDocuments} tone="light">
            <RefreshCwIcon className="size-4" />
            Atualizar
          </IconButton>
        </div>
        <label className={cn(
          'mt-5 grid cursor-pointer place-items-center rounded-lg border border-dashed border-slate-300 bg-slate-50 p-8 text-center',
          !canUpload && 'cursor-not-allowed opacity-60'
        )}>
          <UploadCloudIcon className="mb-2 size-8 text-blue-600" />
          <span className="text-sm font-semibold">{canUpload ? 'Selecionar arquivos' : 'Upload bloqueado por permissão'}</span>
          <input
            type="file"
            multiple
            className="hidden"
            disabled={!canUpload || loading}
            onChange={(event) => uploadFiles(event.target.files)}
          />
        </label>
        {!!Object.keys(uploadProgress).length && (
          <div className="mt-4 space-y-2">
            {Object.entries(uploadProgress).map(([fileName, percent]) => (
              <div key={fileName} className="text-sm text-slate-600">
                <div className="flex justify-between"><span>{fileName}</span><span>{percent}%</span></div>
                <div className="mt-1 h-2 overflow-hidden rounded-lg bg-slate-100">
                  <div className="h-full bg-blue-600" style={{ width: `${percent}%` }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
        {docs.length === 0 ? (
          <EmptyState icon={FileTextIcon} label="Nenhum documento retornado para este workspace." />
        ) : (
          docs.map((doc) => (
            <div key={doc.id} className="grid gap-3 border-b border-slate-200 p-4 last:border-b-0 lg:grid-cols-[1fr_140px_160px_80px]">
              <div className="min-w-0">
                <p className="truncate font-semibold">{doc.title}</p>
                <p className="mt-1 line-clamp-2 text-sm text-slate-500">{doc.content_summary || doc.file_path}</p>
              </div>
              <span className="text-sm font-semibold text-slate-600">{statusLabel(doc.status)}</span>
              <span className="text-sm text-slate-500">{formatDate(doc.updated_at)}</span>
              <button
                type="button"
                disabled={!canDelete}
                onClick={() => requestDelete(doc.id)}
                className="inline-flex size-9 items-center justify-center rounded-lg text-slate-500 transition hover:bg-red-50 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-40"
                aria-label="Solicitar exclusão"
              >
                <Trash2Icon className="size-4" />
              </button>
            </div>
          ))
        )}
      </section>
    </div>
  )
}

function AreasPage({
  areas,
  activeWorkspaceId,
  setActiveWorkspaceId,
  setPage
}: {
  areas: LittleBullArea[]
  activeWorkspaceId: string
  setActiveWorkspaceId: (workspaceId: string) => void
  setPage: (page: Page) => void
}) {
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {areas.map((area) => (
        <button
          key={area.id}
          type="button"
          onClick={() => {
            setActiveWorkspaceId(area.id)
            setPage('conhecimento')
          }}
          className={cn(
            'rounded-lg border bg-white p-4 text-left shadow-sm transition hover:border-blue-300',
            activeWorkspaceId === area.id ? 'border-blue-500 ring-2 ring-blue-100' : 'border-slate-200'
          )}
        >
          <div className="flex items-center gap-3">
            <span className="grid size-10 place-items-center rounded-lg text-xl" style={{ backgroundColor: `${area.accent}22` }}>
              {area.emoji}
            </span>
            <div>
              <h3 className="font-semibold">{area.label}</h3>
              <p className="text-sm text-slate-500">{area.slug}</p>
            </div>
          </div>
          <p className="mt-3 text-sm text-slate-600">{area.description}</p>
          <div className="mt-4 flex gap-2 text-xs font-semibold text-slate-500">
            <span>{area.document_count} docs</span>
            <span>{area.ready_count} prontos</span>
            <span>{area.processing_count} na fila</span>
          </div>
        </button>
      ))}
    </div>
  )
}

function AssistantsPage({
  assistants,
  setPage
}: {
  assistants: LittleBullAssistant[]
  setPage: (page: Page) => void
}) {
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {assistants.map((assistant) => (
        <div key={assistant.id} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <BotIcon className="size-6 text-blue-600" />
            <span className={cn(
              'rounded-lg px-2 py-1 text-xs font-semibold',
              assistant.enabled ? 'bg-green-50 text-green-700' : 'bg-slate-100 text-slate-500'
            )}>
              {assistant.enabled ? 'Ativo' : 'Pausado'}
            </span>
          </div>
          <h3 className="mt-4 text-lg font-semibold">{assistant.name}</h3>
          <p className="mt-1 text-sm text-slate-500">{assistant.description}</p>
          <div className="mt-4 space-y-2">
            {assistant.response_rules.map((rule) => (
              <div key={rule} className="flex items-start gap-2 text-sm text-slate-600">
                <CheckCircle2Icon className="mt-0.5 size-4 shrink-0 text-green-600" />
                {rule}
              </div>
            ))}
          </div>
          <IconButton onClick={() => setPage('perguntar')} tone="light">
            <MessageCircleIcon className="size-4" />
            Usar agora
          </IconButton>
        </div>
      ))}
    </div>
  )
}

function ActivityPage({
  activity,
  refresh
}: {
  activity: LittleBullActivityItem[]
  refresh: () => Promise<void>
}) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-blue-600">Atividade</p>
          <h2 className="mt-1 text-2xl font-semibold">Eventos do workspace</h2>
        </div>
        <IconButton onClick={refresh} tone="light">
          <RefreshCwIcon className="size-4" />
          Atualizar
        </IconButton>
      </div>
      <div className="mt-5 space-y-2">
        {activity.length === 0 ? (
          <EmptyState icon={ActivityIcon} label="Nenhum evento registrado ainda." />
        ) : (
          activity.map((item) => (
            <div key={item.id} className="rounded-lg border border-slate-200 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="font-semibold">{item.action}</p>
                  <p className="mt-1 text-sm text-slate-500">{item.result}</p>
                </div>
                <span className="text-xs font-semibold text-slate-500">{formatDate(item.created_at)}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  )
}

function AdminPage({
  approvals,
  auditEvents,
  principal,
  refreshAdmin
}: {
  approvals: LittleBullApproval[]
  auditEvents: LittleBullAuditEvent[]
  principal: LittleBullPrincipal | null
  refreshAdmin: () => Promise<void>
}) {
  const canDecide = hasPermission(principal, permissionMap.decideApprovals)

  const decide = async (approval: LittleBullApproval, decision: 'approve' | 'reject') => {
    try {
      if (decision === 'approve') {
        await approveLittleBullApproval(approval.approval_id)
      } else {
        await rejectLittleBullApproval(approval.approval_id)
      }
      await refreshAdmin()
      toast.success('Aprovação atualizada')
    } catch (error) {
      toast.error(errorMessage(error))
    }
  }

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-blue-600">Aprovações</p>
            <h2 className="mt-1 text-2xl font-semibold">Fila humana</h2>
          </div>
          <IconButton onClick={refreshAdmin} tone="light">
            <RefreshCwIcon className="size-4" />
            Atualizar
          </IconButton>
        </div>
        <div className="mt-5 space-y-3">
          {approvals.length === 0 ? (
            <EmptyState icon={ShieldCheckIcon} label="Nenhuma aprovação pendente ou recente." />
          ) : (
            approvals.map((approval) => (
              <div key={approval.approval_id} className="rounded-lg border border-slate-200 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="font-semibold">{approval.action}</p>
                    <p className="mt-1 text-sm text-slate-500">{approval.reason}</p>
                    <p className="mt-1 text-xs text-slate-400">{approval.status} · {formatDate(approval.requested_at)}</p>
                  </div>
                  {approval.status === 'pending' && (
                    <div className="flex gap-2">
                      <IconButton disabled={!canDecide} onClick={() => decide(approval, 'approve')} tone="blue">
                        Aprovar
                      </IconButton>
                      <IconButton disabled={!canDecide} onClick={() => decide(approval, 'reject')} tone="light">
                        Rejeitar
                      </IconButton>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <p className="text-sm font-semibold uppercase tracking-wide text-blue-600">Auditoria</p>
        <h2 className="mt-1 text-2xl font-semibold">Eventos duráveis</h2>
        <div className="mt-5 space-y-2">
          {auditEvents.length === 0 ? (
            <EmptyState icon={ActivityIcon} label="Nenhum evento de auditoria retornado." />
          ) : (
            auditEvents.map((event) => (
              <div key={event.event_id} className="rounded-lg border border-slate-200 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="font-semibold">{event.action}</p>
                    <p className="mt-1 text-sm text-slate-500">{event.result} · {event.workspace_id ?? 'global'}</p>
                  </div>
                  <span className="text-xs font-semibold text-slate-500">{formatDate(event.created_at)}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  )
}

export default function LittleBullPreview() {
  const [page, setPage] = useState<Page>('inicio')
  const [principal, setPrincipal] = useState<LittleBullPrincipal | null>(null)
  const [areas, setAreas] = useState<LittleBullArea[]>([])
  const [activeWorkspaceId, setActiveWorkspaceId] = useState('')
  const [docs, setDocs] = useState<LittleBullDocument[]>([])
  const [activity, setActivity] = useState<LittleBullActivityItem[]>([])
  const [assistants, setAssistants] = useState<LittleBullAssistant[]>([])
  const [approvals, setApprovals] = useState<LittleBullApproval[]>([])
  const [auditEvents, setAuditEvents] = useState<LittleBullAuditEvent[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [searchText, setSearchText] = useState('')
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(true)
  const [fatalError, setFatalError] = useState<string | null>(null)

  const refreshAreas = useCallback(async () => {
    const [me, loadedAreas] = await Promise.all([getLittleBullMe(), getLittleBullAreas()])
    setPrincipal(me)
    setAreas(loadedAreas)
    setActiveWorkspaceId((current) => current || loadedAreas[0]?.id || '')
  }, [])

  const refreshDocuments = useCallback(async () => {
    if (!activeWorkspaceId) return
    const response = await getLittleBullDocuments(activeWorkspaceId)
    setDocs(response.documents)
  }, [activeWorkspaceId])

  const refreshActivity = useCallback(async () => {
    if (!activeWorkspaceId) return
    setActivity(await getLittleBullActivity(activeWorkspaceId))
  }, [activeWorkspaceId])

  const refreshAssistants = useCallback(async () => {
    if (!activeWorkspaceId) return
    setAssistants(await getLittleBullAssistants(activeWorkspaceId))
  }, [activeWorkspaceId])

  const refreshAdmin = useCallback(async () => {
    const [approvalItems, auditItems] = await Promise.all([
      getLittleBullApprovals(),
      getLittleBullAuditEvents()
    ])
    setApprovals(approvalItems)
    setAuditEvents(auditItems)
  }, [])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      setLoading(true)
      try {
        await refreshAreas()
        if (!cancelled) setFatalError(null)
      } catch (error) {
        if (!cancelled) setFatalError(errorMessage(error))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [refreshAreas])

  useEffect(() => {
    if (!activeWorkspaceId) return
    refreshDocuments().catch((error) => toast.error(errorMessage(error)))
    refreshActivity().catch((error) => toast.error(errorMessage(error)))
    refreshAssistants().catch((error) => toast.error(errorMessage(error)))
  }, [activeWorkspaceId, refreshActivity, refreshAssistants, refreshDocuments])

  useEffect(() => {
    if (page === 'admin') {
      refreshAdmin().catch((error) => toast.error(errorMessage(error)))
    }
  }, [page, refreshAdmin])

  const onSearchSubmit = () => {
    const value = searchText.trim()
    if (!value) return
    setPrompt(value)
    setPage('perguntar')
  }

  const content = useMemo(() => {
    if (loading) {
      return (
        <div className="grid min-h-[70vh] place-items-center">
          <Loader2Icon className="size-8 animate-spin text-blue-600" />
        </div>
      )
    }

    if (fatalError) {
      return (
        <div className="rounded-lg border border-red-200 bg-red-50 p-5 text-red-800">
          <p className="font-semibold">Little Bull indisponível</p>
          <p className="mt-1 text-sm">{fatalError}</p>
        </div>
      )
    }

    if (!activeWorkspaceId) {
      return <EmptyState icon={FolderOpenIcon} label="Nenhum workspace disponível para este usuário." />
    }

    switch (page) {
      case 'inicio':
        return (
          <HomePage
            areas={areas}
            docs={docs}
            activity={activity}
            setPage={setPage}
            setActiveWorkspaceId={setActiveWorkspaceId}
          />
        )
      case 'perguntar':
        return (
          <AskPage
            activeWorkspaceId={activeWorkspaceId}
            prompt={prompt}
            setPrompt={setPrompt}
            messages={messages}
            setMessages={setMessages}
            refreshActivity={refreshActivity}
          />
        )
      case 'conhecimento':
        return (
          <KnowledgePage
            activeWorkspaceId={activeWorkspaceId}
            docs={docs}
            setDocs={setDocs}
            principal={principal}
            refreshActivity={refreshActivity}
          />
        )
      case 'areas':
        return (
          <AreasPage
            areas={areas}
            activeWorkspaceId={activeWorkspaceId}
            setActiveWorkspaceId={setActiveWorkspaceId}
            setPage={setPage}
          />
        )
      case 'assistentes':
        return <AssistantsPage assistants={assistants} setPage={setPage} />
      case 'atividade':
        return <ActivityPage activity={activity} refresh={refreshActivity} />
      case 'admin':
        return (
          <AdminPage
            approvals={approvals}
            auditEvents={auditEvents}
            principal={principal}
            refreshAdmin={refreshAdmin}
          />
        )
      default:
        return null
    }
  }, [
    activeWorkspaceId,
    activity,
    approvals,
    areas,
    auditEvents,
    assistants,
    docs,
    fatalError,
    loading,
    messages,
    page,
    principal,
    prompt,
    refreshActivity,
    refreshAdmin
  ])

  return (
    <Shell
      page={page}
      setPage={setPage}
      areas={areas}
      activeWorkspaceId={activeWorkspaceId}
      setActiveWorkspaceId={setActiveWorkspaceId}
      principal={principal}
      searchText={searchText}
      setSearchText={setSearchText}
      onSearchSubmit={onSearchSubmit}
    >
      {content}
    </Shell>
  )
}
