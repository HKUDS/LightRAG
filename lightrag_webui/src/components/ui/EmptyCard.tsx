import { cn } from '@/lib/utils'
import { Card, CardDescription, CardTitle } from '@/components/ui/Card'
import { FilesIcon } from 'lucide-react'

interface EmptyCardProps extends React.ComponentPropsWithoutRef<typeof Card> {
  title: string
  description?: string
  action?: React.ReactNode
  icon?: React.ComponentType<{ className?: string }>
}

export default function EmptyCard({
  title,
  description,
  icon: Icon = FilesIcon,
  action,
  className,
  ...props
}: EmptyCardProps) {
  return (
    <Card
      className={cn(
        'flex w-full h-full flex-col items-center justify-center space-y-6 bg-transparent p-16 border-none shadow-none',
        className
      )}
      {...props}
    >
      <div className="shrink-0 rounded-2xl bg-gradient-to-br from-muted/80 to-muted/40 p-6 ring-1 ring-border/50">
        <Icon className="text-muted-foreground/70 size-10" aria-hidden="true" />
      </div>
      <div className="flex flex-col items-center gap-2 text-center max-w-sm">
        <CardTitle className="text-lg font-semibold">{title}</CardTitle>
        {description ? (
          <CardDescription className="text-sm text-muted-foreground/80">
            {description}
          </CardDescription>
        ) : null}
      </div>
      {action ? <div className="mt-2">{action}</div> : null}
    </Card>
  )
}
