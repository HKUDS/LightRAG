import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import {
  needsCustomizationLoad,
  SERVER_DEFAULT_LOCALE,
  useCustomizationStore
} from '@/stores/customization'
import { detectBrowserLanguage } from '@/lib/browserLanguage'
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

  // Language priority (PRD): explicit persisted choice > browser language >
  // bundle default. Without an explicit choice the store's language already
  // mirrors the browser detection (i18n bootstrap); when even that found no
  // supported match, send NO locale and let the server resolve the bundle's
  // default_locale.
  const locale = languageUserSelected
    ? language
    : (detectBrowserLanguage() ?? SERVER_DEFAULT_LOCALE)

  // (Re-)target on mount and whenever the user switches language, AND retry
  // whenever this locale is not the one actually loaded — a failed request
  // leaves the target already set to it, so keying on the target alone would
  // never try again (see `needsCustomizationLoad`). The store decides whether
  // a network request is needed: a re-target alone still invalidates a stale
  // in-flight response (fast A → B → A switching), and repeated calls while a
  // request is out are deduped there.
  useEffect(() => {
    if (needsCustomizationLoad(locale, targetLocale, loadedLocale)) {
      void useCustomizationStore.getState().load(locale)
    }
  }, [locale, targetLocale, loadedLocale])

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
