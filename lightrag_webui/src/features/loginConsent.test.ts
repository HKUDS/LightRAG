import { describe, expect, test } from 'bun:test'
import {
  CONSENT_LABEL_MARKER,
  isAgreedTo,
  isLoginBlockedByConsent,
  resolveConsentDocuments,
  shouldShowLoginConsent,
  splitConsentLabel
} from './loginConsent'

const state = (over: Partial<Parameters<typeof isLoginBlockedByConsent>[0]> = {}) => ({
  pending: false,
  consentRequired: false,
  agreed: false,
  ...over
})

describe('shouldShowLoginConsent', () => {
  test('shown only once the server said the gate exists', () => {
    expect(shouldShowLoginConsent(state({ consentRequired: true }))).toBe(true)
    expect(shouldShowLoginConsent(state())).toBe(false)
  })

  test('never while the customization response is still in flight', () => {
    // Both guesses are wrong before the answer arrives: a checkbox that
    // disappears, or a form that briefly looks ungated.
    expect(
      shouldShowLoginConsent(state({ pending: true, consentRequired: true }))
    ).toBe(false)
    expect(shouldShowLoginConsent(state({ pending: true }))).toBe(false)
  })
})

describe('isLoginBlockedByConsent', () => {
  test('gated deployment blocks until the box is ticked', () => {
    expect(isLoginBlockedByConsent(state({ consentRequired: true }))).toBe(true)
    expect(
      isLoginBlockedByConsent(state({ consentRequired: true, agreed: true }))
    ).toBe(false)
  })

  test('ungated deployment never blocks', () => {
    expect(isLoginBlockedByConsent(state())).toBe(false)
    expect(isLoginBlockedByConsent(state({ agreed: true }))).toBe(false)
  })

  test('an unsettled response blocks, whatever the last known flag was', () => {
    // Two windows this closes: the first load, where the customization
    // request runs beside the auth-status probe; and a language switch,
    // where the previous locale's snapshot stays on screen while the new
    // one is in flight. A submit landing in either would otherwise sign in
    // on a gated deployment having agreed to nothing.
    expect(isLoginBlockedByConsent(state({ pending: true }))).toBe(true)
    expect(
      isLoginBlockedByConsent(state({ pending: true, consentRequired: true, agreed: true }))
    ).toBe(true)
  })

  test('a stale tick does not survive the gate turning on', () => {
    // agreed=true carried over from a previous render is not consent to a
    // document the visitor never saw; only consentRequired + agreed passes,
    // and both are re-evaluated together on every render.
    expect(
      isLoginBlockedByConsent(state({ consentRequired: true, agreed: false }))
    ).toBe(true)
  })
})

describe('splitConsentLabel', () => {
  test('splits around the marker, keeping both sides verbatim', () => {
    expect(splitConsentLabel(`同意${CONSENT_LABEL_MARKER}`)).toEqual({
      before: '同意',
      after: ''
    })
    expect(splitConsentLabel(`I agree to the ${CONSENT_LABEL_MARKER}`)).toEqual({
      before: 'I agree to the ',
      after: ''
    })
  })

  test('handles a trailing verb (the link is not always last)', () => {
    // Japanese puts the verb after the document name; concatenating a prefix
    // and a link would render this sentence backwards.
    expect(splitConsentLabel(`${CONSENT_LABEL_MARKER}に同意します`)).toEqual({
      before: '',
      after: 'に同意します'
    })
  })

  test('splits on the FIRST marker only', () => {
    expect(
      splitConsentLabel(`a${CONSENT_LABEL_MARKER}b${CONSENT_LABEL_MARKER}c`)
    ).toEqual({ before: 'a', after: `b${CONSENT_LABEL_MARKER}c` })
  })

  test('a translation missing the placeholder still gets a link', () => {
    // Degrading to text-only would leave a checkbox demanding agreement to a
    // document with no way to open it.
    expect(splitConsentLabel('I agree')).toEqual({ before: 'I agree ', after: '' })
    expect(splitConsentLabel('')).toEqual({ before: '', after: '' })
  })
})

describe('isAgreedTo', () => {
  const tick = (document: string | null, agreed: boolean) => ({ document, agreed })

  test('a tick counts only for the document it was given for', () => {
    expect(isAgreedTo(tick('# 协议', true), '# 协议')).toBe(true)
    expect(isAgreedTo(tick('# 协议', false), '# 协议')).toBe(false)
  })

  test('a language switch does not carry consent onto the new agreement', () => {
    // The regression: the checkbox outlives the document under it, so a bare
    // boolean would let a visitor who agreed to the Chinese text submit
    // against the English one the instant it replaced it.
    expect(isAgreedTo(tick('# 协议', true), '# Agreement')).toBe(false)
  })

  test('returning to the very same document keeps the tick', () => {
    // Identity is the text: zh → en → zh puts back the document already
    // read and agreed to, so re-ticking it would be busywork, not consent.
    const agreed = tick('# 协议', true)
    expect(isAgreedTo(agreed, null)).toBe(false) // ungated locale in between
    expect(isAgreedTo(agreed, '# 协议')).toBe(true)
  })

  test('there is no agreeing to a document that does not exist', () => {
    expect(isAgreedTo(tick(null, true), null)).toBe(false)
    expect(isAgreedTo(tick('# 协议', true), null)).toBe(false)
  })
})

describe('resolveConsentDocuments', () => {
  const translated = 'Privacy Policy Agreement'

  test('the bundle wording wins over the WebUI translation', () => {
    // The deployment names its document; the translation cannot know it.
    expect(resolveConsentDocuments('Example Corp Terms of Service', translated)).toBe(
      'Example Corp Terms of Service'
    )
  })

  test('a bundle that declares no name keeps the translated default', () => {
    // Bundles written before the manifest field existed must keep working.
    expect(resolveConsentDocuments(null, translated)).toBe(translated)
    expect(resolveConsentDocuments(undefined, translated)).toBe(translated)
  })

  test('a blank declaration falls back instead of naming nothing', () => {
    expect(resolveConsentDocuments('', translated)).toBe(translated)
    expect(resolveConsentDocuments('   \n ', translated)).toBe(translated)
  })

  test('surrounding whitespace never reaches the label', () => {
    expect(resolveConsentDocuments('  Terms  ', translated)).toBe('Terms')
  })
})
