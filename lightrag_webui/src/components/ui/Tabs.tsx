import * as React from 'react'
import * as TabsPrimitive from '@radix-ui/react-tabs'

import { cn } from '@/lib/utils'

const Tabs = TabsPrimitive.Root

const TabsList = React.forwardRef<
  React.ComponentRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      'bg-muted text-muted-foreground inline-flex h-10 items-center justify-center rounded-md p-1',
      className
    )}
    {...props}
  />
))
TabsList.displayName = TabsPrimitive.List.displayName

const TabsTrigger = React.forwardRef<
  React.ComponentRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      'ring-offset-background focus-visible:ring-ring data-[state=active]:bg-background data-[state=active]:text-foreground inline-flex items-center justify-center rounded-sm px-3 py-1.5 text-sm font-medium whitespace-nowrap transition-all focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50 data-[state=active]:shadow-sm',
      className
    )}
    {...props}
  />
))
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName

const TabsContent = React.forwardRef<
  React.ComponentRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => {
  const innerRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    const el = innerRef.current
    if (!el) return
    const observer = new MutationObserver(() => {
      const state = el.getAttribute('data-state')
      el.style.display = state === 'inactive' ? 'none' : ''
      el.style.visibility = state === 'inactive' ? 'hidden' : 'visible'
      el.style.position = state === 'inactive' ? 'absolute' : ''
      el.style.pointerEvents = state === 'inactive' ? 'none' : ''
    })
    observer.observe(el, { attributes: true, attributeFilter: ['data-state'] })
    // Initial sync
    const state = el.getAttribute('data-state')
    el.style.display = state === 'inactive' ? 'none' : ''
    el.style.visibility = state === 'inactive' ? 'hidden' : 'visible'
    el.style.position = state === 'inactive' ? 'absolute' : ''
    el.style.pointerEvents = state === 'inactive' ? 'none' : ''
    return () => observer.disconnect()
  }, [])

  return (
    <TabsPrimitive.Content
      ref={(node) => {
        innerRef.current = node
        if (typeof ref === 'function') ref(node)
        else if (ref) ref.current = node
      }}
      className={cn(
        'ring-offset-background focus-visible:ring-ring focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none',
        'h-full w-full',
        className
      )}
      forceMount
      {...props}
    />
  )
})
TabsContent.displayName = TabsPrimitive.Content.displayName

export { Tabs, TabsList, TabsTrigger, TabsContent }
