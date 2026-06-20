import type { DisplayArtifactItem } from './ArtifactFileSection'

export type ArtifactViewMode = 'zh' | 'source'

export function resolveArtifactViewState(
  artifact: DisplayArtifactItem,
  viewMode: ArtifactViewMode
) {
  const selectedFile = viewMode === 'zh' ? artifact.zhFile : artifact.sourceFile
  const selectedContent = viewMode === 'zh' ? artifact.content : artifact.originalContent
  const emptyMessage =
    viewMode === 'source' && !selectedContent ? '暂无原始文件内容。' : undefined

  return { selectedFile, selectedContent, emptyMessage }
}
