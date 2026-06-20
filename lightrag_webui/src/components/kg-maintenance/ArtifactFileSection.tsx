import { useState } from 'react'
import { RefreshCwIcon } from 'lucide-react'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'

export type DisplayArtifactItem = {
  key: string
  title: string
  sourceFile: string
  zhFile: string
  contentType: 'application/json' | 'text/markdown' | string
  displayStatus: string
  generatedAt?: string
  model?: string
  content?: string
  originalContent?: string
}

type ArtifactViewMode = 'zh' | 'source'

interface ArtifactFileSectionProps {
  title: string
  artifacts: DisplayArtifactItem[]
  onRegenerate?: (artifactKey: string) => void
}

export function ArtifactFileSection({
  title,
  artifacts,
  onRegenerate
}: ArtifactFileSectionProps) {
  const [viewModes, setViewModes] = useState<Record<string, ArtifactViewMode>>({})

  const setArtifactView = (artifactKey: string, mode: ArtifactViewMode) => {
    setViewModes((current) => ({ ...current, [artifactKey]: mode }))
  }

  return (
    <details open className="border-border/70 bg-background rounded-md border">
      <summary className="hover:bg-muted/40 flex cursor-pointer items-center justify-between gap-3 rounded-md px-3 py-2 text-sm font-semibold">
        <span>{title}</span>
        <span className="text-muted-foreground text-xs font-normal">{artifacts.length} 个产物</span>
      </summary>
      <div className="space-y-2 border-t px-3 py-3">
        {artifacts.length === 0 ? (
          <p className="text-muted-foreground text-sm">暂无产物。</p>
        ) : (
          artifacts.map((artifact) => {
            const viewMode = viewModes[artifact.key] ?? 'zh'
            const selectedFile = viewMode === 'zh' ? artifact.zhFile : artifact.sourceFile
            const selectedContent =
              viewMode === 'zh' ? artifact.content : artifact.originalContent ?? artifact.content

            return (
              <div
                key={artifact.key}
                className="border-border/70 bg-muted/10 rounded-md border px-3 py-2"
              >
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="text-sm font-medium">{artifact.title}</h3>
                      <span className="border-border bg-background rounded px-1.5 py-0.5 text-xs">
                        {artifact.displayStatus}
                      </span>
                    </div>
                    <div className="text-muted-foreground mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs">
                      <span>中文文件：{artifact.zhFile}</span>
                      <span>原始文件：{artifact.sourceFile}</span>
                      <span>{artifact.contentType}</span>
                    </div>
                  </div>
                  {onRegenerate && (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => onRegenerate(artifact.key)}
                    >
                      <RefreshCwIcon className="size-4" />
                      重新生成
                    </Button>
                  )}
                </div>

                {(artifact.generatedAt || artifact.model) && (
                  <div className="text-muted-foreground mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs">
                    {artifact.generatedAt && <span>生成时间：{artifact.generatedAt}</span>}
                    {artifact.model && <span>模型：{artifact.model}</span>}
                  </div>
                )}

                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <div className="border-border inline-flex rounded-md border bg-background p-0.5">
                    <button
                      type="button"
                      aria-pressed={viewMode === 'zh'}
                      onClick={() => setArtifactView(artifact.key, 'zh')}
                      className={cn(
                        'rounded px-2 py-1 text-xs transition-colors',
                        viewMode === 'zh'
                          ? 'bg-emerald-500 text-white'
                          : 'text-muted-foreground hover:bg-muted'
                      )}
                    >
                      中文显示
                    </button>
                    <button
                      type="button"
                      aria-pressed={viewMode === 'source'}
                      onClick={() => setArtifactView(artifact.key, 'source')}
                      className={cn(
                        'rounded px-2 py-1 text-xs transition-colors',
                        viewMode === 'source'
                          ? 'bg-emerald-500 text-white'
                          : 'text-muted-foreground hover:bg-muted'
                      )}
                    >
                      原始文件
                    </button>
                  </div>
                  <span className="text-muted-foreground truncate text-xs">{selectedFile}</span>
                </div>

                {selectedContent && (
                  <pre className="bg-background text-foreground mt-2 max-h-40 overflow-auto rounded border p-2 text-xs whitespace-pre-wrap">
                    {selectedContent}
                  </pre>
                )}
              </div>
            )
          })
        )}
      </div>
    </details>
  )
}
