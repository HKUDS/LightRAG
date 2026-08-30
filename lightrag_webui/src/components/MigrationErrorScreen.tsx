import { useTranslation } from 'react-i18next'
import {
  acceptSettingsDataLoss,
  getSettingsMigrationError,
  isSettingsDataLostError
} from '@/migrations/splitSettingsStorage'

/**
 * Shown instead of the app when the settings-storage split migration did not
 * complete (see migrations/splitSettingsStorage.ts rule 7): dependent stores
 * skipped hydration, so running the UI would write defaults over keys whose
 * legacy values are still pending migration.
 *
 * TWO outcomes, and they must not share a screen. The ordinary one is
 * transient — the legacy envelope survived intact, a reload re-runs the
 * migration before any store is evaluated, and it converges. The other is a
 * store so far over quota that the envelope could only be written back with
 * fields shed, or not at all; those settings are gone, no reload brings them
 * back, and the migration now refuses to complete over what is left. Offering
 * "reload to try again" there loops the user on a promise ("Nothing was
 * lost") that is no longer true, so that case says what happened and offers
 * the only move left — continue, by an explicit choice, with whatever
 * survived plus defaults for the rest.
 */
export default function MigrationErrorScreen() {
  const { t } = useTranslation()
  const dataLost = isSettingsDataLostError(getSettingsMigrationError())

  // Clearing the marker is not enough on its own: `skipHydration` was decided
  // when the stores were created, so they stay un-hydrated for the rest of
  // this session. Reload after accepting.
  const continueAfterLoss = () => {
    acceptSettingsDataLoss()
    window.location.reload()
  }

  return (
    <div className="flex h-dvh w-screen flex-col items-center justify-center gap-4 p-6 text-center">
      <h1 className="text-xl font-semibold">
        {dataLost
          ? t('migration.dataLostTitle', 'Some saved settings could not be recovered')
          : t('migration.failedTitle', 'Failed to prepare local settings')}
      </h1>
      <p className="text-muted-foreground max-w-md text-sm">
        {dataLost
          ? t(
            'migration.dataLostBody',
            'Browser storage was full, so some of your saved settings (theme, language, API key) could not be preserved and are no longer recoverable. Continuing will keep whatever survived and use defaults for the rest.'
          )
          : t(
            'migration.failedBody',
            'Your saved settings could not be migrated to the new storage format. Nothing was lost — reload to try again.'
          )}
      </p>
      <button
        type="button"
        className="bg-primary text-primary-foreground min-h-11 rounded-md px-4 py-2 text-sm"
        onClick={dataLost ? continueAfterLoss : () => window.location.reload()}
      >
        {dataLost
          ? t('migration.continueAfterLoss', 'Continue')
          : t('migration.retry', 'Reload')}
      </button>
    </div>
  )
}
