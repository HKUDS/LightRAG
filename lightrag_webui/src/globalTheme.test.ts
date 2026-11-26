import { readFileSync } from 'fs'
import { join } from 'path'
import { afterEach, describe, expect, test } from 'bun:test'

/**
 * Global theme contracts of index.css:
 * - color-scheme flips native form controls/scrollbars with the theme class
 *   (asserted as COMPUTED style: the real declaration blocks are injected
 *   into happy-dom, which resolves color-scheme).
 * - the prefers-reduced-motion fallback stops looping/spatial animation
 *   while keeping non-motion transitions. happy-dom cannot toggle the
 *   prefers-reduced-motion media feature (matchMedia is always false and
 *   exposes no override), so these rules are pinned at source level — they
 *   are genuinely source-level CSS properties, not rendered component output.
 * - the destructive tokens reach WCAG AA (4.5:1) in every place the UI uses
 *   them: Badge/Button text on the token (static and /80 hover tint) and
 *   Alert text on the page background, in BOTH themes. The ratios are
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
  const block = (): string => {
    const start = css.indexOf('@media (prefers-reduced-motion: reduce) {')
    if (start === -1) throw new Error('missing prefers-reduced-motion block in index.css')
    let depth = 0
    for (let i = css.indexOf('{', start); i < css.length; i++) {
      if (css[i] === '{') depth += 1
      else if (css[i] === '}') {
        depth -= 1
        if (depth === 0) return css.slice(start, i + 1)
      }
    }
    throw new Error('unterminated prefers-reduced-motion block in index.css')
  }

  const ruleFor = (selector: string): string | undefined => {
    const inner = block()
    const body = inner
      .slice(inner.indexOf('{') + 1, inner.lastIndexOf('}'))
      .replace(/\/\*[\s\S]*?\*\//g, '')
    for (const match of body.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      const selectors = match[1].replace(/\s+/g, ' ').trim()
      if (selectors.split(',').some((s) => s.trim() === selector)) return match[2]
    }
    return undefined
  }

  test('looping indicators run a single cycle', () => {
    expect(ruleFor('.animate-spin')).toMatch(/animation-iteration-count:\s*1/)
    expect(ruleFor('.animate-pulse')).toMatch(/animation-iteration-count:\s*1/)
  })

  test('spatial enter/exit keyframes resolve to the final state', () => {
    expect(ruleFor('.animate-in')).toMatch(/animation:\s*none/)
    expect(ruleFor('.animate-out')).toMatch(/animation:\s*none/)
  })

  test('transitions keep color/opacity feedback and drop movement', () => {
    const rule = ruleFor('*')
    expect(rule).toMatch(/transition-property:[^;]*\bcolor\b/)
    expect(rule).toMatch(/transition-property:[^;]*\bopacity\b/)
    expect(rule).not.toMatch(/transition-property:[^;]*\btransform\b/)
    // Must also reach third-party unlayered styles (sonner).
    expect(rule).toMatch(/!important/)
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