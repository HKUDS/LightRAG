import { useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { LogOut, KeyRound, UserCircle } from 'lucide-react'
import AppSettings from '@/components/AppSettings'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore, useBackendState } from '@/stores/state'
import { cn } from '@/lib/utils'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import Separator from '@/components/ui/Separator'
import { navigationService } from '@/services/navigation'

export default function SiteHeader() {
  const { t } = useTranslation()
  const { coreVersion, apiVersion, username, isGuestMode } =
    useAuthStore()
  const health = useBackendState.use.health()

  const versionDisplay = coreVersion && apiVersion ? `${coreVersion}/${apiVersion}` : null

  // apiVersion ends with a warning symbol when the built frontend is stale.
  const hasWarning = apiVersion?.endsWith('\u{FE0F}') // ⚠️
  const versionTooltip = hasWarning
    ? t('header.frontendNeedsRebuild')
    : versionDisplay
      ? `v${versionDisplay}`
      : ''

  const initials = username ? username.slice(0, 1).toUpperCase() : '?'

  const handleNavigateToProfile = useCallback(() => {
    useSettingsStore.getState().setCurrentTab('users' as any)
  }, [])

  const handleLogout = useCallback(() => {
    navigationService.navigateToLogin()
  }, [])

  return (
    <header className="glass-panel sticky top-0 z-20 flex h-14 w-full shrink-0 items-center gap-4 rounded-none border-x-0 border-t-0 px-5">
      {/* Spacer */}
      <div className="flex min-w-0 flex-1" />

      {/* Global status + settings */}
      <div className="flex shrink-0 items-center gap-3">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                className={cn(
                  'flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset',
                  health
                    ? 'bg-emerald-500/10 text-emerald-400 ring-emerald-500/20'
                    : 'bg-rose-500/10 text-rose-400 ring-rose-500/20'
                )}
              >
                <span
                  className={cn(
                    'size-1.5 rounded-full',
                    health ? 'bg-emerald-400' : 'bg-rose-400'
                  )}
                  aria-hidden="true"
                />
                {health ? t('header.online', 'Online') : t('header.offline', 'Offline')}
              </span>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              {health
                ? t('header.backendHealthy', 'Backend is healthy')
                : t('header.backendUnavailable', 'Backend unavailable')}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {versionDisplay && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-muted-foreground hidden cursor-default text-xs tabular-nums sm:inline">
                  v{versionDisplay}
                </span>
              </TooltipTrigger>
              <TooltipContent side="bottom">{versionTooltip}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}

        {/* User profile avatar dropdown */}
        <Popover>
          <PopoverTrigger asChild>
            <button
              type="button"
              className={cn(
                'flex size-8 items-center justify-center rounded-full text-xs font-semibold ring-1 ring-inset transition-colors hover:ring-2',
                'bg-primary/12 text-primary ring-primary/20 hover:bg-primary/20'
              )}
              title={username || t('userManagement.unknownUser', 'Unknown user')}
            >
              {initials}
            </button>
          </PopoverTrigger>
          <PopoverContent side="bottom" align="end" className="w-56 p-2">
            <div className="flex flex-col gap-1">
              {/* User info header */}
              <div className="flex items-center gap-3 px-2 py-2">
                <span className="bg-primary/12 text-primary ring-primary/20 flex size-9 shrink-0 items-center justify-center rounded-full text-sm font-semibold ring-1 ring-inset">
                  {initials}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold">
                    {username || t('userManagement.unknownUser', 'Unknown user')}
                  </p>
                  {isGuestMode && (
                    <p className="text-muted-foreground truncate text-xs">
                      {t('login.guestMode', 'Guest')}
                    </p>
                  )}
                </div>
              </div>

              <Separator />

              {/* Profile link */}
              <button
                type="button"
                onClick={handleNavigateToProfile}
                className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-sm transition-colors hover:bg-foreground/5"
              >
                <UserCircle className="size-4" aria-hidden="true" />
                {t('userManagement.profile', 'Profile')}
              </button>

              {/* Change password */}
              <button
                type="button"
                onClick={handleNavigateToProfile}
                className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-sm transition-colors hover:bg-foreground/5"
              >
                <KeyRound className="size-4" aria-hidden="true" />
                {t('userManagement.changePassword', 'Change Password')}
              </button>

              <Separator />

              {/* Logout */}
              <button
                type="button"
                onClick={handleLogout}
                className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-sm text-rose-400 transition-colors hover:bg-rose-500/10"
              >
                <LogOut className="size-4" aria-hidden="true" />
                {t('header.logout', 'Logout')}
              </button>
            </div>
          </PopoverContent>
        </Popover>

        <AppSettings />
      </div>
    </header>
  )
}