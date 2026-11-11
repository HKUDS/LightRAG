/**
 * Seed the UI-customization store for component tests.
 *
 * Pages behind `useCustomizedContent` (login, workspace welcome) render a
 * placeholder until the first customization response settles — §8.8 of the
 * workspace-entry PRD: never flash the default content before knowing
 * whether a bundle is active. A test that renders one of those pages without
 * seeding therefore finds an empty page and no useful roles at all.
 *
 * Seeding the STORE rather than stubbing the request keeps the resolution
 * logic (`needsCustomizationLoad`, locale targeting, the brand-title chain)
 * real: `load` is never called because the target locale is already loaded.
 */
import type { UICustomization, UICustomizationBrand } from '@/api/customization'
import { useCustomizationStore } from '@/stores/customization'
import { resolveUiLanguage } from '@/lib/browserLanguage'
import { useSettingsStore } from '@/stores/settings'

/**
 * The locale `useCustomizedContent` will target, computed the same way the
 * hook computes it. Seeding any OTHER locale leaves `needsCustomizationLoad`
 * true, and the effect fires a real request for the difference.
 */
const targetedLocale = (): string => {
  const { language, languageUserSelected } = useSettingsStore.getState()
  return resolveUiLanguage(languageUserSelected, language)
}

const INITIAL = {
  status: 'loading',
  snapshot: null,
  targetLocale: null,
  loadedLocale: null,
  pendingLocale: null,
  failedAttempts: 0
} as const

/**
 * Present the store as "a bundle for the current locale is loaded".
 *
 * Pass `customized: false` in `extra` for the uncustomized-deployment case —
 * a settled response that declares no bundle, which is a different state from
 * "still loading" and the one that makes the frontend defaults take over.
 */
export const seedCustomization = (
  brand: UICustomizationBrand = {},
  extra: Partial<UICustomization> = {}
): void => {
  const locale = targetedLocale()
  const snapshot: UICustomization = {
    customized: true,
    locale,
    brand,
    ...extra
  }

  useCustomizationStore.setState({
    status: 'ready',
    snapshot,
    // Both must name the SAME locale the hook will target, or the effect
    // fires a real `load` for the difference.
    targetLocale: locale,
    loadedLocale: locale,
    pendingLocale: null,
    failedAttempts: 0
  })
}

/** Return the store to its module-load state, for the next test file. */
export const resetCustomization = (): void => {
  useCustomizationStore.setState({ ...INITIAL })
}
