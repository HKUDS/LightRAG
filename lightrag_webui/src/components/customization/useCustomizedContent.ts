import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import { useCustomizationStore } from '@/stores/customization'
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
  const status = useCustomizationStore((s) => s.status)
  const snapshot = useCustomizationStore((s) => s.snapshot)
  const loadedLanguage = useCustomizationStore((s) => s.loadedLanguage)

  // (Re-)request on mount and whenever the user switches language.
  useEffect(() => {
    if (loadedLanguage !== language) {
      void useCustomizationStore.getState().load(language)
    }
  }, [language, loadedLanguage])

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
