import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

const source = readFileSync(join(import.meta.dir, 'LoginPage.tsx'), 'utf8')
const welcomeSource = readFileSync(
  join(import.meta.dir, 'workspace', 'WorkspaceWelcome.tsx'),
  'utf8'
)

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

  test('uses the same resolved deployment title as the welcome page', () => {
    expect(source).toContain('{content.brandTitle}</h1>')
    expect(welcomeSource).toContain('{content.brandTitle}</h1>')
    expect(source).not.toContain('>LightRAG</h1>')
  })

  test('passes the fresh auth-status title into the customization fallback', () => {
    expect(source).toContain(
      'const [authStatusTitle, setAuthStatusTitle] = useState<string | null>(null)'
    )
    expect(source).toContain('setAuthStatusTitle(status.webui_title || null)')
    expect(source).toContain('useCustomizedContent(authStatusTitle)')
  })
})
