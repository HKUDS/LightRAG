const BASIC_HTML_ENTITIES: Record<string, string> = {
  amp: '&',
  apos: '\'',
  gt: '>',
  lt: '<',
  nbsp: '\u00a0',
  quot: '"'
}

const stripInvalidUnicodeAndControls = (value: string): string =>
  Array.from(value, (character) => {
    const codePoint = character.codePointAt(0) ?? 0
    const isInvalidUnicode =
      (codePoint >= 0xd800 && codePoint <= 0xdfff) ||
      codePoint === 0xfffe ||
      codePoint === 0xffff
    const isControl =
      codePoint <= 0x08 ||
      (codePoint >= 0x0b && codePoint <= 0x0c) ||
      (codePoint >= 0x0e && codePoint <= 0x1f) ||
      codePoint === 0x7f
    return isInvalidUnicode || isControl ? '' : character
  }).join('')

const decodeHtmlEntities = (value: string): string => {
  if (typeof document !== 'undefined') {
    const textarea = document.createElement('textarea')
    textarea.innerHTML = value
    return textarea.value
  }

  return value.replace(/&(#(?:x[0-9a-f]+|\d+)|[a-z]+);/gi, (match, entity: string) => {
    if (entity.startsWith('#')) {
      const isHex = entity[1]?.toLowerCase() === 'x'
      const codePoint = Number.parseInt(entity.slice(isHex ? 2 : 1), isHex ? 16 : 10)
      if (Number.isNaN(codePoint) || codePoint > 0x10ffff) return match

      try {
        return String.fromCodePoint(codePoint)
      } catch {
        return match
      }
    }

    return BASIC_HTML_ENTITIES[entity.toLowerCase()] ?? match
  })
}

const toHalfWidthEntityCharacters = (value: string): string =>
  Array.from(value, (character) => {
    const codePoint = character.codePointAt(0)
    if (codePoint !== undefined && codePoint >= 0xff10 && codePoint <= 0xff19) {
      return String.fromCodePoint(codePoint - 0xfee0)
    }
    if (codePoint !== undefined && codePoint >= 0xff21 && codePoint <= 0xff3a) {
      return String.fromCodePoint(codePoint - 0xfee0)
    }
    if (codePoint !== undefined && codePoint >= 0xff41 && codePoint <= 0xff5a) {
      return String.fromCodePoint(codePoint - 0xfee0)
    }

    return character
  }).join('')

const stripMatchingOuterQuotes = (value: string): string => {
  const pairs: ReadonlyArray<readonly [string, string]> = [
    ['"', '"'],
    ['\'', '\''],
    ['“', '”'],
    ['‘', '’'],
    ['《', '》']
  ]

  let result = value
  for (const [opening, closing] of pairs) {
    if (!result.startsWith(opening) || !result.endsWith(closing)) continue

    const inner = result.slice(opening.length, -closing.length)
    if (!inner.includes(opening) && !inner.includes(closing)) result = inner
  }

  return result
}

/**
 * Mirror the backend extraction naming contract for entity names entered in the WebUI.
 * The backend remains authoritative; this helper provides immediate validation and
 * makes the duplicate check use the same canonical candidate that will be submitted.
 */
export const normalizeEntityName = (input: string): string => {
  let name = stripInvalidUnicodeAndControls(decodeHtmlEntities(input.trim()))
  if (!name) return ''

  name = name.replace(/<\/?p\s*>|<p\/>/gi, '')
  name = name.replace(/<\/?br\s*>|<br\/>/gi, '')
  name = toHalfWidthEntityCharacters(name)
    .replaceAll('－', '-')
    .replaceAll('＋', '+')
    .replaceAll('／', '/')
    .replaceAll('＊', '*')
    .replaceAll('（', '(')
    .replaceAll('）', ')')
    .replaceAll('—', '-')
    .replaceAll('　', ' ')

  name = name.replace(/(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])/g, '')
  name = name.replace(
    /(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9()[\]@#$%!&*\-=+_])/g,
    ''
  )
  name = name.replace(
    /(?<=[a-zA-Z0-9()[\]@#$%!&*\-=+_])\s+(?=[\u4e00-\u9fa5])/g,
    ''
  )

  name = stripMatchingOuterQuotes(name)
  name = name.replace(/[“”‘’]/g, '')
  name = name.replace(/['"]+(?=[\u4e00-\u9fa5])/g, '')
  name = name.replace(/(?<=[\u4e00-\u9fa5])['"]+/g, '')
  name = name.replaceAll('\u00a0', ' ')
  name = name.replace(/(?<=[^\d])\u202f/g, ' ').trim()

  if (name.length < 3 && /^[0-9]+$/.test(name)) return ''
  if (name.length < 6 && name.includes('.') && /^[0-9.]+$/.test(name)) return ''

  return name
}
