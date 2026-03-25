import {
  activatePromptConfigVersion,
  createPromptConfigVersion,
  deletePromptConfigVersion,
  diffPromptConfigVersion,
  getPromptConfigVersions,
  initializePromptConfig,
  PromptConfigGroup,
  PromptVersionCreateRequest,
  PromptVersionRecord
} from '@/api/lightrag'
import PromptGroupSwitcher from '@/components/prompt-management/PromptGroupSwitcher'
import PromptVersionDiffDialog from '@/components/prompt-management/PromptVersionDiffDialog'
import PromptVersionEditor from '@/components/prompt-management/PromptVersionEditor'
import PromptVersionList from '@/components/prompt-management/PromptVersionList'
import EmptyCard from '@/components/ui/EmptyCard'
import Button from '@/components/ui/Button'
import { useSettingsStore } from '@/stores/settings'
import { useMemo, useEffect, useState, useCallback } from 'react'
import { toast } from 'sonner'

export default function PromptManagement() {
  const language = useSettingsStore.use.language()
  const groupType = useSettingsStore.use.promptManagementGroup()
  const selectedVersionId = useSettingsStore.use.promptManagementSelectedVersionId()
  const setGroupType = useSettingsStore.use.setPromptManagementGroup()
  const setSelectedVersionId = useSettingsStore.use.setPromptManagementSelectedVersionId()

  const [registry, setRegistry] = useState<{ active_version_id: string | null; versions: PromptVersionRecord[] } | null>(null)
  const [loading, setLoading] = useState(true)
  const [diffOpen, setDiffOpen] = useState(false)
  const [diffData, setDiffData] = useState<{ changes: Record<string, { before: unknown; after: unknown }> } | null>(null)

  const loadVersions = useCallback(async () => {
    setLoading(true)
    try {
      const nextRegistry = await getPromptConfigVersions(groupType)
      setRegistry(nextRegistry)
      const nextSelectedVersionId = nextRegistry.versions.some((version) => version.version_id === selectedVersionId)
        ? selectedVersionId
        : nextRegistry.active_version_id || nextRegistry.versions[0]?.version_id || null
      setSelectedVersionId(nextSelectedVersionId)
    } catch (error) {
      toast.error(String(error))
    } finally {
      setLoading(false)
    }
  }, [groupType, selectedVersionId, setSelectedVersionId])

  useEffect(() => {
    loadVersions()
  }, [loadVersions])

  const versions = registry?.versions || []
  const activeVersionId = registry?.active_version_id || null
  const selectedVersion = versions.find((version) => version.version_id === selectedVersionId) || null
  const versionsById = useMemo(
    () => Object.fromEntries(versions.map((version) => [version.version_id, version])),
    [versions]
  )

  const locale = language.startsWith('zh') ? 'zh' : 'en'

  if (!loading && versions.length === 0) {
    return (
      <EmptyCard
        className="h-full"
        title="Prompt Management"
        description="No prompt versions exist for this workspace yet."
        action={
          <Button
            type="button"
            onClick={async () => {
              await initializePromptConfig(locale)
              await loadVersions()
            }}
          >
            Initialize Seed Versions
          </Button>
        }
      />
    )
  }

  const handleSaveVersion = async (payload: PromptVersionCreateRequest) => {
    const savedVersion = await createPromptConfigVersion(groupType, payload)
    toast.success(`Saved ${savedVersion.version_name}`)
    await loadVersions()
    setSelectedVersionId(savedVersion.version_id)
  }

  const handleActivateVersion = async (version: PromptVersionRecord) => {
    if (
      groupType === 'indexing' &&
      !window.confirm(
        'Changing indexing configuration can create mixed-schema graph data unless you rebuild the workspace. Continue?'
      )
    ) {
      return
    }

    const response = await activatePromptConfigVersion(groupType, version.version_id)
    if (response.warning) {
      toast.warning(response.warning)
    } else {
      toast.success(`Activated ${version.version_name}`)
    }
    await loadVersions()
  }

  const handleDeleteVersion = async (version: PromptVersionRecord) => {
    if (!window.confirm(`Delete ${version.version_name}?`)) {
      return
    }
    await deletePromptConfigVersion(groupType, version.version_id)
    toast.success(`Deleted ${version.version_name}`)
    await loadVersions()
  }

  const handleShowDiff = async (version: PromptVersionRecord) => {
    const nextDiff = await diffPromptConfigVersion(
      groupType,
      version.version_id,
      activeVersionId && activeVersionId !== version.version_id ? activeVersionId : undefined
    )
    setDiffData(nextDiff)
    setDiffOpen(true)
  }

  return (
    <div className="grid h-full grid-cols-[320px_1fr] gap-4 p-4">
      <div className="space-y-4">
        <PromptGroupSwitcher
          value={groupType}
          onChange={(nextGroup: PromptConfigGroup) => {
            setGroupType(nextGroup)
            setSelectedVersionId(null)
          }}
        />
        <PromptVersionList
          versions={versions}
          activeVersionId={activeVersionId}
          selectedVersionId={selectedVersionId}
          onSelectVersion={setSelectedVersionId}
        />
      </div>

      <PromptVersionEditor
        groupType={groupType}
        version={selectedVersion}
        versionsById={versionsById}
        activeVersionId={activeVersionId}
        onSaveVersion={handleSaveVersion}
        onActivateVersion={handleActivateVersion}
        onDeleteVersion={handleDeleteVersion}
        onShowDiff={handleShowDiff}
      />

      <PromptVersionDiffDialog open={diffOpen} onOpenChange={setDiffOpen} diffData={diffData} />
    </div>
  )
}
