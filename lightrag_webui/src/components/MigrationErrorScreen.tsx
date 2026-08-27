import { useTranslation } from 'react-i18next'

/**
 * Shown instead of the app when the settings-storage split migration did not
 * complete (see migrations/splitSettingsStorage.ts rule 7): dependent stores
 * skipped hydration, so running the UI would write defaults over keys whose
 * legacy values are still pending migration. The retry is a full reload —
 * the migration re-runs before any store is evaluated and converges.
 */
export default function MigrationErrorScreen() {
  const { t } = useTranslation()
  return (
    <div className="flex h-dvh w-screen flex-col items-center justify-center gap-4 p-6 text-center">
      <h1 className="text-xl font-semibold">
        {t('migration.failedTitle', 'Failed to prepare local settings')}
      </h1>
      <p className="text-muted-foreground max-w-md text-sm">
        {t(
          'migration.failedBody',
          'Your saved settings could not be migrated to the new storage format. Nothing was lost — reload to try again.'
        )}
      </p>
      <button
        type="button"
        className="bg-primary text-primary-foreground rounded-md px-4 py-2 text-sm"
        onClick={() => window.location.reload()}
      >
        {t('migration.retry', 'Reload')}
      </button>
    </div>
  )
}
