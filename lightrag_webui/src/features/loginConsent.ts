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
  /** True while the verdict for the TARGETED locale is unknown — before the
   * first response, and across a language switch (see `consentPending`). */
  pending: boolean
  /** Server's verdict; meaningless while `pending`. */
  consentRequired: boolean
  /** Whether the visitor has ticked the box FOR THE DOCUMENT ON SCREEN —
   * see `isAgreedTo`, never the raw checkbox state. */
  agreed: boolean
}

/**
 * A tick, and the agreement document it was given for.
 *
 * The pairing is the point: the checkbox lives as long as the login page,
 * but the document under it does not — a language switch replaces it. A bare
 * boolean would carry a tick from one locale's agreement onto another's, so
 * consent is stored against the text it was consent TO.
 */
export interface LoginConsentTick {
  /** The agreement document shown when the box was ticked, or null. */
  document: string | null
  agreed: boolean
}

/**
 * Whether `tick` counts as agreement to the document currently displayed.
 *
 * Identity is the document TEXT, not the locale: returning to a locale whose
 * agreement is byte-for-byte what the visitor already agreed to is the same
 * agreement, while a locale whose text differs is a new one and must be
 * ticked again. A null document is never agreed to — there is nothing to
 * have read.
 */
export function isAgreedTo(tick: LoginConsentTick, document: string | null): boolean {
  return tick.agreed && document !== null && tick.document === document
}

/**
 * Whether the consent control should be on screen.
 *
 * Only once the verdict for the targeted locale has settled: rendering it
 * earlier would mean guessing, and both guesses are wrong — a checkbox that
 * vanishes on a deployment that has no gate, or a form that briefly looks
 * ungated on one that does.
 */
export function shouldShowLoginConsent(state: LoginConsentState): boolean {
  return !state.pending && state.consentRequired
}

/**
 * Whether the gate currently BLOCKS a sign-in attempt.
 *
 * Blocking while `pending` is the load-bearing case, not a nicety. Two
 * windows open it: the first load, where the customization request runs in
 * parallel with the auth-status probe; and a language switch, where the
 * store keeps the PREVIOUS locale's snapshot on screen while the new one is
 * in flight. In both, a visitor who submits inside the window would sign in
 * on a gated deployment having agreed to nothing. Unknown therefore blocks,
 * and the flag is believed only once it is known.
 *
 * A hard customization failure is NOT unknown — the store settles to the
 * default representation with `consentRequired: false`, and the gate is off.
 * That is deliberate: the agreement text lives in the bundle, so a deployment
 * whose branding endpoint is unreachable has nothing to show a user and must
 * not become unloggable-into.
 */
export function isLoginBlockedByConsent(state: LoginConsentState): boolean {
  if (state.pending) return true
  return state.consentRequired && !state.agreed
}

/**
 * How the consent checkbox names the documents it links to.
 *
 * The BUNDLE owns this text (`locales.<locale>.consent_documents` in the
 * manifest), because only the deployment knows what its own document is
 * called — the WebUI's translation can only guess, and its guess is wrong
 * for every deployment that does not ship exactly a privacy policy plus a
 * model service agreement. The translated string stays as the fallback for a
 * bundle that declares no name, so an existing bundle keeps its label.
 *
 * A blank bundle value falls back too: the loader rejects one, but a label
 * that renders as an empty link would leave the checkbox demanding agreement
 * to something unnamed, and there is a correct string right here to use.
 */
export function resolveConsentDocuments(
  fromBundle: string | null | undefined,
  translated: string
): string {
  const declared = fromBundle?.trim()
  return declared ? declared : translated
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
