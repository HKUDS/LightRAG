import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const source = readFileSync(join(import.meta.dir, 'LoginPage.tsx'), 'utf8')

describe('login logo customization wiring', () => {
  test('uses the resolved bundle logo instead of importing the built-in asset', () => {
    expect(source).not.toContain('import logoUrl from \'@/assets/logo.svg\'')
    expect(source).toContain('content.logoUrl && failedLogoUrl !== content.logoUrl')
    expect(source).toContain('src={content.logoUrl}')
    expect(source).toContain('alt={content.logoAlt}')
    expect(source).not.toContain('import { ZapIcon } from \'lucide-react\'')
    expect(source).not.toContain('<ZapIcon')
  })

  test('latches a failed URL without blocking a later locale logo', () => {
    expect(source).toContain(
      'const [failedLogoUrl, setFailedLogoUrl] = useState<string | null>(null)'
    )
    expect(source).toContain('onError={() => setFailedLogoUrl(content.logoUrl)}')
  })
})
