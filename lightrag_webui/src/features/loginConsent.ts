/**
 * The login page's consent gate, as pure decisions.
 *
 * A deployment that ships BOTH a customized login blurb and a user-agreement
 * document (one document covering the privacy policy and the model service
 * agreement) requires the visitor to tick "I agree" before they can sign in.
 * Whether that gate exists at all is the SERVER's decision — `consentRequired`
 * arrives as `consent_required` on `GET /ui/customization` — so nothing here
 * re-derives it from the presence of the documents.
 *
 * Split out of the component because it is the part that must not be wrong:
 * the repo's frontend tests cover logic, not rendering (there is no DOM test
 * setup), so the rules a login-blocking control depends on live where they can
 * be tested directly.
 */

export interface LoginConsentState {
  /** True until the customization response settles. */
  loading: boolean
  /** Server's verdict; meaningless while `loading`. */
  consentRequired: boolean
  /** Whether the visitor has ticked the box. */
  agreed: boolean
}

/**
 * Whether the consent control should be on screen.
 *
 * Only once the customization response has settled: rendering it earlier
 * would mean guessing, and both guesses are wrong — a checkbox that vanishes
 * on a deployment that has no gate, or a form that briefly looks ungated on
 * one that does.
 */
export function shouldShowLoginConsent(state: LoginConsentState): boolean {
  return !state.loading && state.consentRequired
}

/**
 * Whether the gate currently BLOCKS a sign-in attempt.
 *
 * Blocking while `loading` is the load-bearing case, not a nicety: the
 * customization request runs in parallel with the auth-status probe, so
 * without it a visitor who submits inside that window would sign in on a
 * gated deployment having agreed to nothing. Unknown therefore blocks, and
 * the flag is believed only once it is known.
 *
 * A hard customization failure is NOT unknown — the store settles to the
 * default representation with `consentRequired: false`, and the gate is off.
 * That is deliberate: the agreement text lives in the bundle, so a deployment
 * whose branding endpoint is unreachable has nothing to show a user and must
 * not become unloggable-into.
 */
export function isLoginBlockedByConsent(state: LoginConsentState): boolean {
  if (state.loading) return true
  return state.consentRequired && !state.agreed
}

/**
 * Sentinel passed as the `documents` interpolation value of
 * `login.consentLabel`, so the translated sentence can be split around the
 * link without assuming where the document name sits in it. Word order
 * differs per language — Chinese and English lead with the verb, Japanese
 * ends with it — so the position is READ back out of the translation rather
 * than assumed by concatenation.
 *
 * U+0000 cannot occur in a translation file, which is exactly why it is the
 * marker: any other sentinel could collide with real text.
 */
export const CONSENT_LABEL_MARKER = '\u0000'

/**
 * Splits an interpolated consent label into the text before and after the
 * document link.
 *
 * A translation that lost the `{{documents}}` placeholder still renders
 * usably: the whole sentence becomes the leading text and the link follows
 * it. Dropping the link instead would leave a checkbox demanding agreement
 * to a document with no way to open it.
 */
export function splitConsentLabel(label: string): { before: string; after: string } {
  const index = label.indexOf(CONSENT_LABEL_MARKER)
  if (index === -1) {
    return { before: label ? `${label} ` : '', after: '' }
  }
  return {
    before: label.slice(0, index),
    after: label.slice(index + CONSENT_LABEL_MARKER.length)
  }
}
