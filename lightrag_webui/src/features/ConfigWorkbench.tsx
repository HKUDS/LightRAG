import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ConfigEnvField,
  ConfigPromptStage,
  ConfigWorkbenchResponse,
  getConfigWorkbench,
  pickWorkspaceFolder,
  updateEntityTypePrompt,
  updateEnvConfig
} from '@/api/lightrag'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import {
  ChunkKey,
  ParserPresetKey,
  buildParserRuleForPreset,
  getParserRuleHint
} from '@/features/configWorkbenchRules'
import { cn } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import { useTranslation } from 'react-i18next'
import {
  BoxIcon,
  CheckIcon,
  DatabaseIcon,
  FileTextIcon,
  FolderIcon,
  Layers3Icon,
  PlusIcon,
  RefreshCwIcon,
  RotateCcwIcon,
  SaveIcon,
  Settings2Icon,
  SparklesIcon,
  WandSparklesIcon
} from 'lucide-react'
import { toast } from 'sonner'

type ViewKey = 'parser' | 'chunking' | 'models' | 'entity_prompt'

const WORKSPACE_LIST_KEY = 'LIGHTRAG_WEBUI_WORKSPACES'
const CHUNK_KEYS: ChunkKey[] = ['F', 'R', 'V', 'P']

const UI = {
  loading: '正在加载配置...',
  unavailable: '配置不可用',
  loadFailed: '加载配置失败',
  restartRequired: '需要重启 Server',
  restartDescription: '这些设置写入 .env 或提示词文件后，需要重启 LightRAG server 才会生效。',
  reload: '重新加载',
  reset: '重置',
  save: '保存设置',
  savePrompt: '保存提示词',
  savedEnv: '配置已保存，重启 server 后生效。',
  saveEnvFailed: '保存配置失败',
  savedPrompt: '提示词已保存，重启 server 后生效。',
  savePromptFailed: '保存提示词失败',
  workspace: 'Workspace',
  workspaceName: 'Workspace 名称',
  addWorkspace: '添加',
  pickWorkspaceFolder: '选择 Workspace 文件夹',
  removeWorkspace: '移除',
  workingDir: '数据目录',
  inputDir: '文件目录',
  folderPickCancelled: '已取消选择文件夹',
  removeKeepsDirs: '仅从列表移除，不删除目录',
  parser: '解析器选择',
  parserPreset: '解析器',
  parserPresetNative: 'LightRAG 默认',
  parserPresetLegacy: 'Legacy 文本解析',
  parserPresetRagAnything: 'RAG-Anything',
  parserPresetMinerU: 'MinerU HTTP',
  parserPresetDocling: 'Docling',
  customParserRule: '自定义规则',
  chunking: '分块策略',
  models: '模型选择',
  prompt: '实体关系抽取提示词',
  ruleHint: '规则提示',
  applicableFileRange: '适用文件范围',
  compatibleChunkRange: '适配切块范围',
  fallbackChunkStrategy: '兜底切块策略',
  entityPromptProfile: '提示词配置',
  secretPlaceholder: '已配置；留空表示不修改',
  noWorkspace: '暂无 Workspace',
  invalidWorkspace: 'Workspace 名称不能包含 /、\\，也不能是 . 或 ..'
}

const EN_UI = {
  ...UI,
  loading: 'Loading configuration...',
  unavailable: 'Configuration unavailable',
  loadFailed: 'Failed to load configuration',
  restartRequired: 'Server restart required',
  restartDescription:
    'These settings are written to .env or prompt files and require a LightRAG server restart.',
  reload: 'Reload',
  reset: 'Reset',
  save: 'Save settings',
  savePrompt: 'Save prompt',
  savedEnv: 'Configuration saved. Restart the server to apply changes.',
  saveEnvFailed: 'Failed to save configuration',
  savedPrompt: 'Prompt saved. Restart the server to apply changes.',
  savePromptFailed: 'Failed to save prompt',
  addWorkspace: 'Add',
  pickWorkspaceFolder: 'Choose workspace folder',
  removeWorkspace: 'Remove',
  workingDir: 'Working dir',
  inputDir: 'Input dir',
  folderPickCancelled: 'Folder selection cancelled',
  removeKeepsDirs: 'Removes from list only; directories are kept.',
  parser: 'Parser',
  parserPreset: 'Parser',
  parserPresetNative: 'LightRAG default',
  parserPresetLegacy: 'Legacy text parser',
  parserPresetRagAnything: 'RAG-Anything',
  parserPresetMinerU: 'MinerU HTTP',
  parserPresetDocling: 'Docling',
  customParserRule: 'Custom rule',
  chunking: 'Chunking',
  models: 'Models',
  prompt: 'Entity relation prompt',
  ruleHint: 'Rule hint',
  applicableFileRange: 'File scope',
  compatibleChunkRange: 'Compatible chunking',
  fallbackChunkStrategy: 'Fallback chunking',
  entityPromptProfile: 'Prompt profile',
  noWorkspace: 'No workspace',
  invalidWorkspace: 'Workspace name cannot contain / or \\ and cannot be . or ..'
}

const CHUNK_COPY: Record<
  ChunkKey,
  { title: string; subtitle: string; fields: string[] }
> = {
  F: {
    title: 'F 固定 Token',
    subtitle: '按 token 窗口切分，适合通用文本。',
    fields: [
      'CHUNK_F_SIZE',
      'CHUNK_F_OVERLAP_SIZE',
      'CHUNK_F_SPLIT_BY_CHARACTER',
      'CHUNK_F_SPLIT_BY_CHARACTER_ONLY'
    ]
  },
  R: {
    title: 'R 递归字符',
    subtitle: '按分隔符递归切分，适合中文、多层级文档。',
    fields: ['CHUNK_R_SIZE', 'CHUNK_R_OVERLAP_SIZE', 'CHUNK_R_SEPARATORS']
  },
  V: {
    title: 'V 语义向量',
    subtitle: '按句间语义距离切分，适合语义段落边界不明显的文本。',
    fields: [
      'CHUNK_V_SIZE',
      'CHUNK_V_BREAKPOINT_THRESHOLD_TYPE',
      'CHUNK_V_BREAKPOINT_THRESHOLD_AMOUNT',
      'CHUNK_V_BUFFER_SIZE',
      'CHUNK_V_SENTENCE_SPLIT_REGEX'
    ]
  },
  P: {
    title: 'P 段落语义',
    subtitle: '先保留段落结构，再做语义合并，适合报告、论文、长文档。',
    fields: ['CHUNK_P_SIZE', 'CHUNK_P_OVERLAP_SIZE']
  }
}

const FIELD_LABELS: Record<string, string> = {
  WORKSPACE: '当前 Workspace',
  WORKING_DIR: '数据根目录',
  INPUT_DIR: '文件根目录',
  CHUNK_F_SIZE: '块大小',
  CHUNK_F_OVERLAP_SIZE: '重叠大小',
  CHUNK_F_SPLIT_BY_CHARACTER: '预切分字符',
  CHUNK_F_SPLIT_BY_CHARACTER_ONLY: '只按字符切分',
  CHUNK_R_SIZE: '块大小',
  CHUNK_R_OVERLAP_SIZE: '重叠大小',
  CHUNK_R_SEPARATORS: '递归分隔符',
  CHUNK_V_SIZE: '块大小上限',
  CHUNK_V_BREAKPOINT_THRESHOLD_TYPE: '断点阈值类型',
  CHUNK_V_BREAKPOINT_THRESHOLD_AMOUNT: '断点阈值数值',
  CHUNK_V_BUFFER_SIZE: '句子缓冲数',
  CHUNK_V_SENTENCE_SPLIT_REGEX: '句子切分正则',
  CHUNK_P_SIZE: '块大小',
  CHUNK_P_OVERLAP_SIZE: '重叠大小',
  LLM_BINDING: 'LLM Provider',
  LLM_MODEL: 'LLM Model',
  LLM_BINDING_HOST: 'LLM Host',
  LLM_BINDING_API_KEY: 'LLM API Key',
  EMBEDDING_BINDING: 'Embedding Provider',
  EMBEDDING_MODEL: 'Embedding Model',
  EMBEDDING_BINDING_HOST: 'Embedding Host',
  EMBEDDING_BINDING_API_KEY: 'Embedding API Key'
}

const MODEL_GROUPS = [
  {
    title: 'LLM',
    icon: WandSparklesIcon,
    fields: ['LLM_BINDING', 'LLM_MODEL', 'LLM_BINDING_HOST', 'LLM_BINDING_API_KEY']
  },
  {
    title: 'Embedding',
    icon: DatabaseIcon,
    fields: [
      'EMBEDDING_BINDING',
      'EMBEDDING_MODEL',
      'EMBEDDING_BINDING_HOST',
      'EMBEDDING_BINDING_API_KEY'
    ]
  }
]

const getEditableFields = (workbench: ConfigWorkbenchResponse | null): ConfigEnvField[] => {
  if (!workbench) return []
  return workbench.env.sections.flatMap((section) =>
    section.fields.filter((field) => field.editable)
  )
}

const buildFieldMap = (workbench: ConfigWorkbenchResponse | null): Record<string, ConfigEnvField> => {
  const fields: Record<string, ConfigEnvField> = {}
  workbench?.env.sections.forEach((section) => {
    section.fields.forEach((field) => {
      fields[field.key] = field
    })
  })
  return fields
}

const workspaceNamesFromPayload = (workbench: ConfigWorkbenchResponse | null): string[] => {
  const names = workbench?.workspace.available?.map((item) => item.name) || []
  const current = workbench?.workspace.current || ''
  return current && !names.includes(current) ? [current, ...names] : names
}

const buildEnvDraft = (workbench: ConfigWorkbenchResponse): Record<string, string> => {
  const draft: Record<string, string> = {}
  workbench.env.sections.forEach((section) => {
    section.fields.forEach((field) => {
      draft[field.key] = field.value
    })
  })
  draft[WORKSPACE_LIST_KEY] = workspaceNamesFromPayload(workbench).join(',')
  return draft
}

const splitWorkspaceList = (value: string): string[] => {
  const seen = new Set<string>()
  return value
    .split(',')
    .map((item) => item.trim())
    .filter((item) => {
      if (!item || seen.has(item)) return false
      seen.add(item)
      return true
    })
}

const findEntityPrompt = (workbench: ConfigWorkbenchResponse | null): ConfigPromptStage | null => {
  return workbench?.prompts.stages.find((stage) => stage.key === 'entity_type') || null
}

const deriveChunkStrategy = (parserRule: string): ChunkKey => {
  for (const rule of parserRule.replaceAll(';', ',').split(',')) {
    if (!rule.includes(':')) continue
    const target = rule.split(':', 2)[1]
    if (!target.includes('-')) continue
    const options = target.split('-', 2)[1]
    for (const char of options) {
      if ((CHUNK_KEYS as string[]).includes(char)) {
        return char as ChunkKey
      }
    }
  }
  return 'F'
}

const applyChunkStrategyToParserRule = (parserRule: string, strategy: ChunkKey): string => {
  const trimmed = parserRule.trim()
  if (!trimmed) {
    return `*:native-te${strategy},*:legacy-${strategy}`
  }

  return trimmed
    .split(/([,;])/)
    .map((part) => {
      if (part === ',' || part === ';' || !part.trim() || !part.includes(':')) {
        return part
      }

      const [pattern, rawTarget] = part.split(':', 2)
      const [engine, rawOptions = ''] = rawTarget.split('-', 2)
      const options = rawOptions.replace(/[FRVP]/g, '')
      return `${pattern}:${engine}-${options}${strategy}`
    })
    .join('')
}

const deriveParserPreset = (parserRule: string): ParserPresetKey => {
  const engines = new Set<string>()
  parserRule
    .replaceAll(';', ',')
    .split(',')
    .forEach((rule) => {
      if (!rule.includes(':')) return
      const target = rule.split(':', 2)[1].trim()
      if (!target) return
      engines.add(target.split('-', 1)[0].trim().toLowerCase())
    })

  if (engines.has('raganything')) return 'raganything'
  if (engines.has('mineru')) return 'mineru'
  if (engines.has('docling')) return 'docling'
  if (engines.has('native')) return 'native'
  if (engines.size === 1 && engines.has('legacy')) return 'legacy'
  return parserRule.trim() ? 'custom' : 'native'
}

const ConfigWorkbench = () => {
  const { i18n } = useTranslation()
  const currentTab = useSettingsStore.use.currentTab()
  const ui = i18n.language.startsWith('zh') ? UI : EN_UI
  const [workbench, setWorkbench] = useState<ConfigWorkbenchResponse | null>(null)
  const [envDraft, setEnvDraft] = useState<Record<string, string>>({})
  const [promptDraft, setPromptDraft] = useState('')
  const [selectedPromptProfile, setSelectedPromptProfile] = useState('')
  const [selectedView, setSelectedView] = useState<ViewKey>('chunking')
  const [selectedStrategy, setSelectedStrategy] = useState<ChunkKey>('F')
  const [pickingWorkspace, setPickingWorkspace] = useState(false)
  const [loading, setLoading] = useState(true)
  const [savingEnv, setSavingEnv] = useState(false)
  const [savingPrompt, setSavingPrompt] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const applyWorkbenchData = useCallback((data: ConfigWorkbenchResponse) => {
    const draft = buildEnvDraft(data)
    const entityPrompt = findEntityPrompt(data)

    setWorkbench(data)
    setEnvDraft(draft)
    setPromptDraft(entityPrompt?.content || '')
    setSelectedPromptProfile(
      data.prompts.entity_type_active_profile || data.prompts.entity_type_profiles[0]?.name || ''
    )
    setSelectedStrategy(
      ((data.chunking?.active_strategy as ChunkKey | undefined) ||
        deriveChunkStrategy(draft.LIGHTRAG_PARSER || '')) as ChunkKey
    )
  }, [])

  const loadWorkbench = useCallback(
    async (promptProfile?: string) => {
      setLoading(true)
      setError(null)
      try {
        const data = await getConfigWorkbench('.env', promptProfile)
        applyWorkbenchData(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : ui.loadFailed)
      } finally {
        setLoading(false)
      }
    },
    [applyWorkbenchData, ui.loadFailed]
  )

  useEffect(() => {
    if (currentTab !== 'config' || workbench) return
    const timeoutId = window.setTimeout(() => {
      void loadWorkbench()
    }, 0)
    return () => window.clearTimeout(timeoutId)
  }, [currentTab, loadWorkbench, workbench])

  const fieldMap = useMemo(() => buildFieldMap(workbench), [workbench])
  const originalDraft = useMemo(
    () => (workbench ? buildEnvDraft(workbench) : {}),
    [workbench]
  )
  const workspaceList = useMemo(
    () => splitWorkspaceList(envDraft[WORKSPACE_LIST_KEY] || ''),
    [envDraft]
  )
  const activeWorkspace = envDraft.WORKSPACE ?? workbench?.workspace.current ?? ''
  const entityPrompt = useMemo(() => findEntityPrompt(workbench), [workbench])
  const promptDirty = Boolean(entityPrompt?.editable && promptDraft !== entityPrompt.content)
  const selectedParserPreset = useMemo(
    () => deriveParserPreset(envDraft.LIGHTRAG_PARSER || ''),
    [envDraft.LIGHTRAG_PARSER]
  )
  const parserPresetOptions = useMemo(
    () => [
      { key: 'native' as const, label: ui.parserPresetNative },
      { key: 'raganything' as const, label: ui.parserPresetRagAnything },
      { key: 'mineru' as const, label: ui.parserPresetMinerU },
      { key: 'docling' as const, label: ui.parserPresetDocling },
      { key: 'legacy' as const, label: ui.parserPresetLegacy },
      { key: 'custom' as const, label: ui.customParserRule }
    ],
    [ui]
  )
  const parserRuleHint = useMemo(
    () => getParserRuleHint(selectedParserPreset, selectedStrategy),
    [selectedParserPreset, selectedStrategy]
  )

  const dirtyEnvValues = useMemo(() => {
    if (!workbench) return {}
    const dirty: Record<string, string> = {}

    getEditableFields(workbench).forEach((field) => {
      const nextValue = envDraft[field.key] ?? ''
      if (nextValue !== field.value) {
        dirty[field.key] = nextValue
      }
    })

    const originalWorkspaces = originalDraft[WORKSPACE_LIST_KEY] || ''
    const nextWorkspaces = envDraft[WORKSPACE_LIST_KEY] || ''
    if (nextWorkspaces !== originalWorkspaces) {
      dirty[WORKSPACE_LIST_KEY] = nextWorkspaces
    }

    return dirty
  }, [envDraft, originalDraft, workbench])

  const dirtyEnvCount = Object.keys(dirtyEnvValues).length

  const updateDraftValue = (key: string, value: string) => {
    setEnvDraft((draft) => ({
      ...draft,
      [key]: value
    }))
  }

  const selectWorkspace = (name: string) => {
    updateDraftValue('WORKSPACE', name)
  }

  const selectChunkStrategy = (strategy: ChunkKey) => {
    setSelectedStrategy(strategy)
    updateDraftValue(
      'LIGHTRAG_PARSER',
      applyChunkStrategyToParserRule(envDraft.LIGHTRAG_PARSER || '', strategy)
    )
  }

  const selectParserPreset = (preset: ParserPresetKey) => {
    const nextRule = buildParserRuleForPreset(preset, selectedStrategy)
    if (!nextRule) return
    updateDraftValue('LIGHTRAG_PARSER', nextRule)
  }

  const chooseWorkspaceFolder = async () => {
    if (pickingWorkspace) return

    setPickingWorkspace(true)
    try {
      const response = await pickWorkspaceFolder({
        initial_dir: envDraft.INPUT_DIR || workbench?.workspace.input_dir || ''
      })

      if (!response.selected_path || !response.workspace) {
        toast.info(ui.folderPickCancelled)
        return
      }

      setEnvDraft((draft) => {
        const workspace = response.workspace || ''
        const names = splitWorkspaceList(draft[WORKSPACE_LIST_KEY] || '')
        const nextNames = names.includes(workspace)
          ? names
          : [...names, workspace]

        return {
          ...draft,
          [WORKSPACE_LIST_KEY]: nextNames.join(','),
          WORKSPACE: workspace || draft.WORKSPACE || '',
          ...(response.input_dir ? { INPUT_DIR: response.input_dir } : {})
        }
      })
    } catch (err) {
      toast.error(err instanceof Error ? err.message : ui.loadFailed)
    } finally {
      setPickingWorkspace(false)
    }
  }

  const resetEnv = () => {
    if (!workbench) return
    const draft = buildEnvDraft(workbench)
    setEnvDraft(draft)
    setSelectedStrategy(
      ((workbench.chunking?.active_strategy as ChunkKey | undefined) ||
        deriveChunkStrategy(draft.LIGHTRAG_PARSER || '')) as ChunkKey
    )
  }

  const saveEnv = async () => {
    if (!dirtyEnvCount) return
    setSavingEnv(true)
    try {
      const response = await updateEnvConfig(dirtyEnvValues)
      if (response.workbench) {
        applyWorkbenchData(response.workbench)
      } else {
        await loadWorkbench(selectedPromptProfile || undefined)
      }
      toast.success(ui.savedEnv)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : ui.saveEnvFailed)
    } finally {
      setSavingEnv(false)
    }
  }

  const changePromptProfile = async (profile: string) => {
    setSelectedPromptProfile(profile)
    await loadWorkbench(profile || undefined)
    setSelectedView('entity_prompt')
  }

  const savePrompt = async () => {
    if (!entityPrompt?.editable || !promptDirty) return
    const profile =
      entityPrompt.profile ||
      selectedPromptProfile ||
      workbench?.prompts.entity_type_active_profile ||
      workbench?.prompts.entity_type_profiles[0]?.name ||
      'entity_type_prompt.yml'

    setSavingPrompt(true)
    try {
      const response = await updateEntityTypePrompt({
        profile,
        entity_types_guidance: promptDraft
      })
      if (response.workbench) {
        applyWorkbenchData(response.workbench)
      } else {
        await loadWorkbench(response.profile)
      }
      toast.success(ui.savedPrompt)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : ui.savePromptFailed)
    } finally {
      setSavingPrompt(false)
    }
  }

  const renderField = (key: string, options: { multiline?: boolean } = {}) => {
    const field = fieldMap[key]
    const value = envDraft[key] ?? ''
    const isDirty = value !== (originalDraft[key] ?? field?.value ?? '')
    const disabled = field ? !field.editable : false
    const label = FIELD_LABELS[key] || key
    const inputClass = cn(
      'w-full',
      isDirty && 'border-emerald-300 bg-emerald-50 dark:bg-emerald-950/30'
    )

    return (
      <div key={key} className="grid gap-1.5">
        <div className="flex items-baseline justify-between gap-3">
          <label className="text-sm font-medium">{label}</label>
          <code className="text-muted-foreground truncate text-xs">{key}</code>
        </div>
        {options.multiline ? (
          <Textarea
            value={value}
            disabled={disabled}
            onChange={(event) => updateDraftValue(key, event.target.value)}
            className={cn(inputClass, 'min-h-24 resize-y font-mono text-sm')}
          />
        ) : (
          <Input
            value={value}
            disabled={disabled}
            type={field?.sensitive ? 'password' : 'text'}
            placeholder={field?.sensitive && field.configured ? ui.secretPlaceholder : ''}
            onChange={(event) => updateDraftValue(key, event.target.value)}
            className={inputClass}
          />
        )}
      </div>
    )
  }

  const navItems: Array<{ key: ViewKey; label: string; icon: typeof Settings2Icon }> = [
    { key: 'parser', label: ui.parser, icon: FileTextIcon },
    { key: 'chunking', label: ui.chunking, icon: Layers3Icon },
    { key: 'models', label: ui.models, icon: Settings2Icon },
    { key: 'entity_prompt', label: ui.prompt, icon: SparklesIcon }
  ]
  const viewTitle =
    selectedView === 'parser'
      ? ui.parser
      : selectedView === 'chunking'
        ? ui.chunking
        : selectedView === 'models'
          ? ui.models
          : ui.prompt

  if (loading && !workbench) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-muted-foreground flex items-center gap-2 text-sm">
          <RefreshCwIcon className="size-4 animate-spin" />
          {ui.loading}
        </div>
      </div>
    )
  }

  if (error && !workbench) {
    return (
      <div className="p-6">
        <Alert variant="destructive">
          <BoxIcon className="size-4" />
          <AlertTitle>{ui.unavailable}</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <div className="bg-muted/20 flex h-full min-h-0">
      <aside className="bg-background flex w-80 shrink-0 flex-col overflow-auto border-r p-3">
        <section className="rounded-md border p-3">
          <div className="mb-3 flex items-center gap-2">
            <FolderIcon className="text-muted-foreground size-4" />
            <h2 className="text-sm font-semibold tracking-normal">{ui.workspace}</h2>
          </div>

          <div className="mt-2 flex gap-2">
            <select
              className="bg-background h-9 min-w-0 flex-1 rounded-md border px-2 text-sm"
              value={activeWorkspace}
              onChange={(event) => selectWorkspace(event.target.value)}
            >
              {workspaceList.length ? (
                workspaceList.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))
              ) : (
                <option value="">{ui.noWorkspace}</option>
              )}
            </select>
            <Button
              type="button"
              size="icon"
              tooltip={ui.pickWorkspaceFolder}
              aria-label={ui.pickWorkspaceFolder}
              disabled={pickingWorkspace}
              onClick={chooseWorkspaceFolder}
            >
              <PlusIcon className={cn('size-4', pickingWorkspace && 'animate-pulse')} />
            </Button>
          </div>

        </section>

        <nav className="mt-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <button
                key={item.key}
                type="button"
                onClick={() => setSelectedView(item.key)}
                className={cn(
                  'hover:bg-muted flex h-10 w-full items-center gap-2 rounded-md px-3 text-left text-sm',
                  selectedView === item.key
                    ? 'bg-muted text-foreground shadow-[inset_3px_0_0_#34d399]'
                    : 'text-muted-foreground'
                )}
              >
                <Icon className="size-4" />
                {item.label}
              </button>
            )
          })}
        </nav>
      </aside>

      <main className="min-w-0 flex-1 overflow-auto p-5">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-normal">{viewTitle}</h1>
            <p className="text-muted-foreground mt-1 text-sm">
              Workspace: {activeWorkspace || '-'}
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => loadWorkbench(selectedPromptProfile || undefined)}
            >
              <RefreshCwIcon className="size-4" />
              {ui.reload}
            </Button>
            {selectedView === 'entity_prompt' ? (
              <>
                <Button
                  variant="outline"
                  disabled={!promptDirty}
                  onClick={() => setPromptDraft(entityPrompt?.content || '')}
                >
                  <RotateCcwIcon className="size-4" />
                  {ui.reset}
                </Button>
                <Button
                  disabled={!entityPrompt?.editable || !promptDirty || savingPrompt}
                  onClick={savePrompt}
                >
                  <CheckIcon className="size-4" />
                  {ui.savePrompt}
                </Button>
              </>
            ) : (
              <>
                <Button variant="outline" disabled={!dirtyEnvCount} onClick={resetEnv}>
                  <RotateCcwIcon className="size-4" />
                  {ui.reset}
                </Button>
                <Button disabled={!dirtyEnvCount || savingEnv} onClick={saveEnv}>
                  <SaveIcon className="size-4" />
                  {ui.save}
                </Button>
              </>
            )}
          </div>
        </div>

        <Alert className="mb-4">
          <Settings2Icon className="size-4" />
          <AlertTitle>{ui.restartRequired}</AlertTitle>
          <AlertDescription>{ui.restartDescription}</AlertDescription>
        </Alert>

        {selectedView === 'parser' && (
          <section className="bg-background rounded-md border">
            <div className="grid gap-4 p-4 md:grid-cols-[240px_1fr]">
              <div className="grid gap-1.5">
                <label className="text-sm font-medium">{ui.parserPreset}</label>
                <select
                  className="bg-background h-9 rounded-md border px-2 text-sm"
                  value={selectedParserPreset}
                  onChange={(event) => selectParserPreset(event.target.value as ParserPresetKey)}
                >
                  {parserPresetOptions.map((preset) => (
                    <option key={preset.key} value={preset.key}>
                      {preset.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="bg-muted/25 grid gap-3 rounded-md border p-3">
                <div className="flex items-center gap-2">
                  <FileTextIcon className="text-muted-foreground size-4" />
                  <h2 className="text-sm font-semibold tracking-normal">{ui.ruleHint}</h2>
                </div>
                <div className="grid gap-2 text-sm">
                  <div className="grid gap-1 md:grid-cols-[140px_1fr]">
                    <span className="text-muted-foreground">{ui.applicableFileRange}</span>
                    <span>{parserRuleHint.applicableFileRange}</span>
                  </div>
                  <div className="grid gap-1 md:grid-cols-[140px_1fr]">
                    <span className="text-muted-foreground">{ui.compatibleChunkRange}</span>
                    <span>{parserRuleHint.compatibleChunkRange}</span>
                  </div>
                  <div className="grid gap-1 md:grid-cols-[140px_1fr]">
                    <span className="text-muted-foreground">{ui.fallbackChunkStrategy}</span>
                    <span>{parserRuleHint.fallbackChunkStrategy}</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {selectedView === 'chunking' && (
          <section className="bg-background rounded-md border">
            <div className="border-b p-4">
              <div className="grid grid-cols-4 gap-2">
                {CHUNK_KEYS.map((key) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => selectChunkStrategy(key)}
                    className={cn(
                      'hover:bg-muted rounded-md border p-3 text-left transition-colors',
                      selectedStrategy === key && 'border-emerald-400 bg-emerald-50 dark:bg-emerald-950/30'
                    )}
                  >
                    <div className="text-sm font-semibold">{CHUNK_COPY[key].title}</div>
                    <div className="text-muted-foreground mt-1 text-xs">
                      {CHUNK_COPY[key].subtitle}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-5 p-4">
              <div className="grid gap-4 md:grid-cols-2">
                {CHUNK_COPY[selectedStrategy].fields.map((key) =>
                  renderField(key, {
                    multiline: key === 'CHUNK_R_SEPARATORS' || key === 'CHUNK_V_SENTENCE_SPLIT_REGEX'
                  })
                )}
              </div>
            </div>
          </section>
        )}

        {selectedView === 'models' && (
          <section className="grid gap-4 lg:grid-cols-2">
            {MODEL_GROUPS.map((group) => {
              const Icon = group.icon
              return (
                <div key={group.title} className="bg-background rounded-md border">
                  <div className="flex items-center gap-2 border-b p-4">
                    <Icon className="text-muted-foreground size-4" />
                    <h2 className="text-lg font-semibold tracking-normal">{group.title}</h2>
                  </div>
                  <div className="grid gap-4 p-4">
                    {group.fields.map((key) => renderField(key))}
                  </div>
                </div>
              )
            })}
          </section>
        )}

        {selectedView === 'entity_prompt' && (
          <section className="bg-background rounded-md border">
            <div className="flex items-center justify-between gap-3 border-b p-4">
              <div className="grid gap-1">
                <label className="text-sm font-medium">{ui.entityPromptProfile}</label>
                <select
                  className="bg-background h-9 min-w-64 rounded-md border px-2 text-sm"
                  value={selectedPromptProfile}
                  onChange={(event) => changePromptProfile(event.target.value)}
                >
                  {workbench?.prompts.entity_type_profiles.length ? (
                    workbench.prompts.entity_type_profiles.map((profile) => (
                      <option key={profile.name} value={profile.name}>
                        {profile.name}
                      </option>
                    ))
                  ) : (
                    <option value="">entity_type_prompt.yml</option>
                  )}
                </select>
              </div>
              <code className="text-muted-foreground text-xs">
                {entityPrompt?.field || 'entity_types_guidance'}
              </code>
            </div>
            <div className="p-4">
              <Textarea
                value={promptDraft}
                readOnly={!entityPrompt?.editable}
                onChange={(event) => setPromptDraft(event.target.value)}
                className="min-h-[520px] resize-y font-mono text-sm"
              />
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default ConfigWorkbench
