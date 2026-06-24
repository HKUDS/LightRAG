import type { DisplayArtifactItem } from './ArtifactFileSection'
import { artifactsForStep } from './kgMaintenanceArtifacts'
import {
  isGeneratedDisplayArtifact,
  type KGMaintenanceDisplayArtifacts
} from './kgIterationLoadUtils'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'

const EXPLICIT_SOURCE_ARTIFACT_KEYS = new Set([
  'quality_score',
  'approval_queue',
  'deferred_changes',
  'accepted_changes',
  'rejected_changes',
  'accepted_changes_apply_result',
  'llm_review_trace',
  'deterministic_proposal_report',
  'proposals_generated',
  'quality_rules',
  'known_issues'
])

export function buildDisplayArtifactItems({
  step,
  displayArtifacts,
  sourceArtifacts,
  artifactExists
}: {
  step: KGMaintenanceSection
  displayArtifacts: KGMaintenanceDisplayArtifacts
  sourceArtifacts: Record<string, string | undefined>
  artifactExists: Map<string, boolean>
}): DisplayArtifactItem[] {
  return artifactsForStep(step).map((artifact) => {
    const displayArtifact = displayArtifacts[artifact.key]
    const sourceContent = resolveArtifactSourceContent(
      artifact.key,
      displayArtifact,
      sourceArtifacts
    )
    return {
      key: artifact.key,
      title: artifact.title,
      sourceFile: artifact.sourceFile,
      zhFile: displayArtifact?.display?.zhFile ?? artifact.zhFile,
      contentType: displayArtifact?.contentType ?? artifact.contentType,
      displayStatus: displayArtifactStatus({
        displayArtifact,
        hasSource: Boolean(sourceContent || artifactExists.get(artifact.key))
      }),
      generatedAt: displayArtifact?.display?.generatedAt,
      model: displayArtifact?.display?.model,
      content: stringifyArtifactContent(artifactContent(displayArtifact)),
      originalContent: sourceContent
    }
  })
}

function resolveArtifactSourceContent(
  artifactKey: string,
  displayArtifact: KGMaintenanceDisplayArtifacts[string] | undefined,
  sourceArtifacts: Record<string, string | undefined>
): string | undefined {
  if (EXPLICIT_SOURCE_ARTIFACT_KEYS.has(artifactKey)) {
    const sourceContent = sourceArtifacts[artifactKey]
    if (sourceContent) return sourceContent
  }

  if (displayArtifact?.display?.fallbackToSource) {
    return stringifyArtifactContent(artifactContent(displayArtifact))
  }

  return undefined
}

function displayArtifactStatus({
  displayArtifact,
  hasSource
}: {
  displayArtifact: KGMaintenanceDisplayArtifacts[string] | undefined
  hasSource: boolean
}): string {
  if (isGeneratedDisplayArtifact(displayArtifact)) return '中文已生成'
  if (displayArtifact?.display?.fallbackToSource || hasSource) return '原始文件'
  return '缺失'
}

function artifactContent(artifact: KGMaintenanceDisplayArtifacts[string] | undefined): unknown {
  if (!artifact) return ''
  if ('payload' in artifact) return artifact.payload
  return artifact.content
}

export function stringifyArtifactContent(value: unknown): string | undefined {
  if (value === null || value === undefined || value === '') return undefined
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}
