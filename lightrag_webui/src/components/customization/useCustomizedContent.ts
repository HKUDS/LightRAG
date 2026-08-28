import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import { needsCustomizationLoad, useCustomizationStore } from '@/stores/customization'
import { resolveUiLanguage } from '@/lib/browserLanguage'
import defaultLogoUrl from '@/assets/logo.svg'

/**
 * Resolves the branding content the workspace entry shows (welcome page and
 * query empty state), following the atomicity rule of the workspace-entry
 * PRD §8.3: a locale's representation comes ENTIRELY from the active bundle,
 * or ENTIRELY from the frontend defaults (no bundle → `customized: false`,
 * or a hard first-load failure) — never a field-by-field mix.
 */

// Which UI languages lay out right-to-left. Used only for the DEFAULT
// content path; a bundle's direction comes from the server response.
const RTL_LANGUAGES = new Set(['ar'])

export interface CustomizedContent {
  /** True until the first customization response settles (§8.8: never flash
   * the default content before knowing whether a bundle is active). */
  loading: boolean
  direction: 'ltr' | 'rtl'
  logoUrl: string | null
  logoAlt: string
  welcomeMarkdown: string
  queryEmptyMarkdown: string
  brandTitle: string
  brandDescription: string | null
}

export function useCustomizedContent(): CustomizedContent {
  const { t } = useTranslation()
  const language = useSettingsStore.use.language()
  const languageUserSelected = useSettingsStore.use.languageUserSelected()
  const status = useCustomizationStore((s) => s.status)
  const snapshot = useCustomizationStore((s) => s.snapshot)
  const targetLocale = useCustomizationStore((s) => s.targetLocale)
  const loadedLocale = useCustomizationStore((s) => s.loadedLocale)
  const failedAttempts = useCustomizationStore((s) => s.failedAttempts)

  // The SHARED language chain — the same function the i18n bootstrap runs on
  // (explicit persisted choice > browser language > DEFAULT_UI_LANGUAGE), so
  // branding and UI chrome always speak one language. Requesting no locale
  // here instead would hand the last step to the bundle's `default_locale`
  // while the chrome, which has no async fallback, was already English.
  // Asking for a concrete locale loses nothing: the server still answers a
  // locale the bundle does not declare with its own default.
  const locale = resolveUiLanguage(languageUserSelected, language)

  // (Re-)target on mount and whenever the user switches language, AND retry
  // whenever this locale is not the one actually loaded — a failed request
  // leaves the target already set to it, so keying on the target alone would
  // never try again (see `needsCustomizationLoad`). `failedAttempts` is a
  // dependency so a failure RE-ARMS this effect without a remount, and it is
  // also the bound that stops the re-arm from looping. The store decides
  // whether a network request is needed: a re-target alone still invalidates
  // a stale in-flight response (fast A → B → A switching), and repeated calls
  // while a request is out are deduped there.
  useEffect(() => {
    if (needsCustomizationLoad(locale, targetLocale, loadedLocale, failedAttempts)) {
      void useCustomizationStore.getState().load(locale)
    }
  }, [locale, targetLocale, loadedLocale, failedAttempts])

  if (status === 'loading') {
    return {
      loading: true,
      direction: 'ltr',
      logoUrl: null,
      logoAlt: '',
      welcomeMarkdown: '',
      queryEmptyMarkdown: '',
      brandTitle: '',
      brandDescription: null
    }
  }

  if (status === 'ready' && snapshot?.customized) {
    return {
      loading: false,
      direction: snapshot.direction === 'rtl' ? 'rtl' : 'ltr',
      // Explicit `"logo": null` in the bundle means NO logo — the layout
      // shows only the text, never the LightRAG default.
      logoUrl: snapshot.brand.logo_url ?? null,
      logoAlt: snapshot.brand.logo_alt ?? '',
      welcomeMarkdown: snapshot.welcome?.content ?? '',
      queryEmptyMarkdown: snapshot.query_empty?.content ?? '',
      brandTitle: snapshot.brand.title || 'LightRAG',
      brandDescription: snapshot.brand.description ?? null
    }
  }

  // `customized: false` (the normal no-bundle deployment) or a hard failure:
  // the WHOLE frontend default representation.
  return {
    loading: false,
    direction: RTL_LANGUAGES.has(language) ? 'rtl' : 'ltr',
    logoUrl: defaultLogoUrl,
    logoAlt: 'LightRAG',
    welcomeMarkdown: t('workspace.welcome.defaultMarkdown'),
    queryEmptyMarkdown: t('workspace.queryEmpty.defaultMarkdown'),
    brandTitle: snapshot?.brand?.title || 'LightRAG',
    brandDescription: snapshot?.brand?.description ?? null
  }
}
