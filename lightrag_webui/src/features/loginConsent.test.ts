import { describe, expect, test } from 'bun:test'
import {
  CONSENT_LABEL_MARKER,
  isLoginBlockedByConsent,
  shouldShowLoginConsent,
  splitConsentLabel
} from './loginConsent'

const state = (over: Partial<Parameters<typeof isLoginBlockedByConsent>[0]> = {}) => ({
  loading: false,
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
      shouldShowLoginConsent(state({ loading: true, consentRequired: true }))
    ).toBe(false)
    expect(shouldShowLoginConsent(state({ loading: true }))).toBe(false)
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
    // The window this closes: the customization request runs beside the
    // auth-status probe, so a submit landing inside it would otherwise sign
    // in on a gated deployment having agreed to nothing.
    expect(isLoginBlockedByConsent(state({ loading: true }))).toBe(true)
    expect(
      isLoginBlockedByConsent(state({ loading: true, consentRequired: true, agreed: true }))
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
