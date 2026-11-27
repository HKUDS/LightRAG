import { readFileSync } from 'fs'
import { join } from 'path'
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import type { Window as HappyDomWindow } from 'happy-dom'

/**
 * Global theme contracts of index.css:
 * - color-scheme flips native form controls/scrollbars with the theme class
 *   (asserted as COMPUTED style: the real declaration blocks are injected
 *   into happy-dom, which resolves color-scheme).
 * - the prefers-reduced-motion fallback stops looping/spatial animation
 *   while keeping non-motion transitions. happy-dom 21.0.4 resolves the
 *   prefers-reduced-motion media feature from
 *   window.happyDOM.settings.device.prefersReducedMotion (for matchMedia AND
 *   getComputedStyle), so these rules are asserted as COMPUTED style on real
 *   elements with that setting toggled — not pinned at source level.
 * - the destructive tokens reach WCAG AA (4.5:1) in every place the UI uses
 *   them: Badge/Button text on the token (static and hover tint) and Alert
 *   text on the page background, in BOTH themes. Badge hovers at /80 and
 *   Button at /90; /80 mixes more page background into the tint, so it is the
 *   worse contrast in both themes and bounds Button's /90 too. The ratios are
 *   computed from the actual token values, so a token edit that breaks
 *   contrast fails here rather than in a manual audit.
 */

const css = readFileSync(join(import.meta.dir, 'index.css'), 'utf8')

type Rgb = [number, number, number]

/** The top-level `selector { ... }` block that declares the theme tokens. */
const tokenBlock = (selector: string): { selector: string; body: string; index: number } => {
  const pattern = new RegExp(`^${selector.replace('.', '\\.')}\\s*\\{([^}]+)\\}`, 'gm')
  for (const match of css.matchAll(pattern)) {
    const body = match[1]
    if (body.includes('--background')) {
      return {
        selector,
        body: body.replace(/\/\*[\s\S]*?\*\//g, ''),
        index: match.index
      }
    }
  }
  throw new Error(`no ${selector} token block in index.css`)
}

const declarations = (body: string): Map<string, string> => {
  const result = new Map<string, string>()
  for (const part of body.split(';')) {
    const colon = part.indexOf(':')
    if (colon === -1) continue
    result.set(part.slice(0, colon).trim(), part.slice(colon + 1).trim())
  }
  return result
}

const hslToRgb = (value: string): Rgb => {
  const match = value.match(/hsl\(\s*([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s*\)/)
  if (!match) throw new Error(`not an hsl token: ${value}`)
  const h = Number(match[1])
  const s = Number(match[2]) / 100
  const l = Number(match[3]) / 100
  const c = (1 - Math.abs(2 * l - 1)) * s
  const hp = (((h % 360) + 360) % 360) / 60
  const x = c * (1 - Math.abs((hp % 2) - 1))
  const [r1, g1, b1]: Rgb =
    hp < 1
      ? [c, x, 0]
      : hp < 2
        ? [x, c, 0]
        : hp < 3
          ? [0, c, x]
          : hp < 4
            ? [0, x, c]
            : hp < 5
              ? [x, 0, c]
              : [c, 0, x]
  const m = l - c / 2
  return [(r1 + m) * 255, (g1 + m) * 255, (b1 + m) * 255]
}

const luminance = ([r, g, b]: Rgb): number => {
  const channel = (v: number): number => {
    const s = v / 255
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

const contrast = (a: Rgb, b: Rgb): number => {
  const la = luminance(a)
  const lb = luminance(b)
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)
}

/** sRGB alpha composite: `fg` at `alpha` over `bg` (Tailwind's /80 tint). */
const over = (fg: Rgb, alpha: number, bg: Rgb): Rgb => [
  fg[0] * alpha + bg[0] * (1 - alpha),
  fg[1] * alpha + bg[1] * (1 - alpha),
  fg[2] * alpha + bg[2] * (1 - alpha)
]

describe('color-scheme', () => {
  afterEach(() => {
    document.getElementById('global-theme-test')?.remove()
    document.documentElement.classList.remove('dark')
  })

  test('native controls follow the theme class, in source cascade order', () => {
    // Injected in the order the blocks appear in the file: :root and .dark
    // have equal specificity, so a reordered file would let light win.
    const blocks = [tokenBlock(':root'), tokenBlock('.dark')].sort((a, b) => a.index - b.index)
    const style = document.createElement('style')
    style.id = 'global-theme-test'
    style.textContent = blocks.map((b) => `${b.selector} { ${b.body} }`).join('\n')
    document.head.appendChild(style)

    expect(getComputedStyle(document.documentElement).colorScheme).toBe('light')
    document.documentElement.classList.add('dark')
    expect(getComputedStyle(document.documentElement).colorScheme).toBe('dark')
  })
})

describe('prefers-reduced-motion fallback', () => {
  // The preload registers happy-dom's Window as the global; the DOM lib's
  // Window type does not carry its happyDOM API, so reach it through the type.
  const device = (window as unknown as HappyDomWindow).happyDOM.settings.device
  const initialMotion = device.prefersReducedMotion

  // The real @media block from index.css, kept in its source cascade context:
  // any @layer wrapping it is preserved. happy-dom drops @layer contents, so a
  // block moved into a layer computes to nothing and these tests fail — the
  // alarm for "the override must stay unlayered to outrank the utilities layer".
  const reducedMotionCss = (): string => {
    const source = css.replace(/\/\*[\s\S]*?\*\//g, '')
    const start = source.indexOf('@media (prefers-reduced-motion: reduce)')
    if (start === -1) throw new Error('missing prefers-reduced-motion block in index.css')
    let depth = 0
    let end = -1
    for (let i = source.indexOf('{', start); i < source.length; i++) {
      if (source[i] === '{') depth += 1
      else if (source[i] === '}') {
        depth -= 1
        if (depth === 0) {
          end = i + 1
          break
        }
      }
    }
    if (end === -1) throw new Error('unterminated prefers-reduced-motion block in index.css')
    const block = source.slice(start, end)
    let wrapper: string | undefined
    for (const match of source.matchAll(/@layer\b([^{;]*)\{/g)) {
      if (match.index === undefined || match.index >= start) continue
      let layerDepth = 1
      let i = match.index + match[0].length
      while (i < source.length && layerDepth > 0) {
        if (source[i] === '{') layerDepth += 1
        else if (source[i] === '}') layerDepth -= 1
        i += 1
      }
      if (start < i - 1) wrapper = match[1].trim()
    }
    return wrapper === undefined ? block : `@layer ${wrapper} { ${block} }`
  }

  // Stand-ins for Tailwind's animation/transition utilities. UNLAYERED because
  // happy-dom drops @layer contents, so a layered base would compute to nothing.
  // Each base uses the declaration form its override writes (happy-dom does not
  // expand shorthands into longhands): iteration-count longhand for the looping
  // indicators, the animation shorthand for enter/exit.
  const base = `
    .animate-spin { animation-name: spin; animation-duration: 1s; animation-iteration-count: infinite; }
    .animate-pulse { animation-name: pulse; animation-duration: 2s; animation-iteration-count: infinite; }
    .animate-in { animation: enter 150ms ease-out; }
    .animate-out { animation: exit 150ms ease-in; }
    .transition-all { transition-property: all; transition-duration: 150ms; }
    .third-party { transition-property: transform; transition-duration: 200ms; }
  `

  const style = document.createElement('style')
  style.id = 'reduced-motion-test'
  const host = document.createElement('div')

  const element = (className: string): HTMLElement => {
    const el = document.createElement('div')
    el.className = className
    host.appendChild(el)
    return el
  }

  // happy-dom caches computed style per element and invalidates it only on a DOM
  // mutation, so toggling the setting alone would return stale values. Re-attach
  // the probes to force the media query to be re-evaluated.
  const setMotion = (value: string): void => {
    device.prefersReducedMotion = value
    for (const el of Array.from(host.children)) host.appendChild(el)
  }

  const computed = (el: HTMLElement, property: string): string =>
    getComputedStyle(el).getPropertyValue(property)

  const transitionList = (el: HTMLElement): string[] =>
    computed(el, 'transition-property')
      .split(',')
      .map((property) => property.trim())

  beforeEach(() => {
    style.textContent = base + reducedMotionCss()
    document.head.appendChild(style)
    document.body.appendChild(host)
  })

  afterEach(() => {
    device.prefersReducedMotion = initialMotion
    style.remove()
    host.remove()
    host.replaceChildren()
  })

  test('looping indicators run a single cycle', () => {
    const spin = element('animate-spin')
    const pulse = element('animate-pulse')
    expect(computed(spin, 'animation-iteration-count')).toBe('infinite')
    expect(computed(pulse, 'animation-iteration-count')).toBe('infinite')
    setMotion('reduce')
    expect(computed(spin, 'animation-iteration-count')).toBe('1')
    expect(computed(pulse, 'animation-iteration-count')).toBe('1')
  })

  test('spatial enter/exit keyframes resolve to the final state', () => {
    const enter = element('animate-in')
    const exit = element('animate-out')
    expect(computed(enter, 'animation')).toBe('enter 150ms ease-out')
    expect(computed(exit, 'animation')).toBe('exit 150ms ease-in')
    setMotion('reduce')
    expect(computed(enter, 'animation')).toBe('none')
    expect(computed(exit, 'animation')).toBe('none')
  })

  test('transitions keep color/opacity feedback and drop movement', () => {
    const utility = element('transition-all')
    // A higher-specificity unlayered rule: only the override's !important lets
    // the universal selector beat it (the rendered form of the old !important pin).
    const thirdParty = element('third-party')
    expect(computed(utility, 'transition-property')).toBe('all')
    expect(computed(thirdParty, 'transition-property')).toBe('transform')
    setMotion('reduce')
    for (const el of [utility, thirdParty]) {
      const list = transitionList(el)
      expect(list).toContain('color')
      expect(list).toContain('opacity')
      expect(list).not.toContain('transform')
      expect(list).not.toContain('all')
    }
  })
})

describe('destructive contrast (WCAG AA)', () => {
  const root = declarations(tokenBlock(':root').body)
  const dark = declarations(tokenBlock('.dark').body)
  const token = (block: Map<string, string>, name: string): Rgb => {
    const value = block.get(name)
    if (!value) throw new Error(`missing ${name} token`)
    return hslToRgb(value)
  }

  const lightDestructive = token(root, '--destructive')
  const lightForeground = token(root, '--destructive-foreground')
  const lightBackground = token(root, '--background')
  const darkDestructive = token(dark, '--destructive')
  const darkForeground = token(dark, '--destructive-foreground')
  const darkBackground = token(dark, '--background')

  const cases: Array<[string, Rgb, Rgb]> = [
    ['light Badge/Button', lightForeground, lightDestructive],
    ['light Badge hover /80', lightForeground, over(lightDestructive, 0.8, lightBackground)],
    ['light Alert', lightDestructive, lightBackground],
    ['dark Badge/Button', darkForeground, darkDestructive],
    ['dark Badge hover /80', darkForeground, over(darkDestructive, 0.8, darkBackground)],
    ['dark Alert', darkDestructive, darkBackground]
  ]

  test.each(cases)('%s reaches at least 4.5:1', (_name, foreground, background) => {
    expect(contrast(foreground, background)).toBeGreaterThanOrEqual(4.5)
  })
})
