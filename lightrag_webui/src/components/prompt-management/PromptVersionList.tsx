import { PromptVersionRecord } from '@/api/lightrag'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { cn } from '@/lib/utils'

type PromptVersionListProps = {
  versions: PromptVersionRecord[]
  activeVersionId: string | null
  selectedVersionId: string | null
  onSelectVersion: (versionId: string) => void
}

export default function PromptVersionList({
  versions,
  activeVersionId,
  selectedVersionId,
  onSelectVersion
}: PromptVersionListProps) {
  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle>Versions</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {versions.map((version) => (
          <button
            key={version.version_id}
            type="button"
            onClick={() => onSelectVersion(version.version_id)}
            className={cn(
              'w-full rounded-lg border p-3 text-left transition-colors',
              selectedVersionId === version.version_id
                ? 'border-emerald-400 bg-emerald-50/80 dark:bg-emerald-950/20'
                : 'hover:bg-muted/40'
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="font-medium">{version.version_name}</div>
              {activeVersionId === version.version_id ? (
                <span className="rounded-full bg-emerald-400 px-2 py-0.5 text-[10px] font-semibold text-white">
                  ACTIVE
                </span>
              ) : null}
            </div>
            <div className="mt-1 text-[11px] text-muted-foreground">
              v{version.version_number}
            </div>
            {version.comment ? (
              <div className="mt-2 text-xs text-muted-foreground line-clamp-2">
                {version.comment}
              </div>
            ) : null}
          </button>
        ))}
      </CardContent>
    </Card>
  )
}
