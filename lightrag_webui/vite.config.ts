import { defineConfig, loadEnv, type Plugin } from 'vite'
import path from 'path'
import fs from 'fs'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Use relative imports here. The '@' alias is configured in resolve.alias
// below and only takes effect during bundling — Node cannot resolve it when
// loading vite.config.ts. Bun resolves tsconfig paths natively, masking the
// issue, but Node does not.
import { normalizeApiPrefix, normalizeWebuiPrefix } from './src/lib/pathPrefix.ts'

/**
 * Inject `<script>window.__LIGHTRAG_CONFIG__ = ...</script>` into index.html.
 *
 * This mirrors what the FastAPI server does at request time in production
 * (see `SmartStaticFiles._inject_runtime_config` in
 * `lightrag/api/lightrag_server.py`). Doing it in dev too means the SPA
 * always reads its prefix the same way, so behaviour matches between
 * `bun run dev` and a production deploy.
 *
 * Only `VITE_DEV_API_PREFIX` is read; the WebUI mount path is fixed at
 * `/webui` (matching the backend's hardcoded `WEBUI_PATH`), so the
 * injected `webuiPrefix` follows the production formula
 * `apiPrefix + "/webui/"` automatically.
 */
function lightragRuntimeConfigPlugin(env: Record<string, string>): Plugin {
  const apiPrefix = normalizeApiPrefix(env.VITE_DEV_API_PREFIX)
  const webuiPrefix = normalizeWebuiPrefix(apiPrefix ? `${apiPrefix}/webui/` : '')
  const payload = JSON.stringify({ apiPrefix, webuiPrefix }).replace(
    /<\//g,
    '<\\/'
  )
  const snippet = `<script>window.__LIGHTRAG_CONFIG__ = ${payload};</script>`

  return {
    name: 'lightrag-dev-runtime-config',
    apply: 'serve',
    transformIndexHtml(html: string) {
      return html.replace('<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->', snippet)
    }
  }
}

/**
 * Workspace first-load dependency audit.
 *
 * The workspace entry's FIRST-LOAD STATIC DEPENDENCY CLOSURE must not contain
 * the admin-only heavyweights. Two complementary pieces of evidence are
 * required, because either alone is insufficient:
 *
 * 1. Chunk graph (equivalent to the manifest's imports/dynamicImports): the
 *    static-import closure of the workspace entry chunk proves mermaid & co.
 *    are reachable only through dynamic edges — but says nothing about which
 *    SOURCE MODULES were merged into each chunk.
 * 2. Source-module audit over `OutputChunk.modules`: proves no forbidden
 *    source module (GraphViewer, DocumentManager, RetrievalView,
 *    stores/graph, graphology, cytoscape, sigma, mermaid) was folded into a
 *    chunk the workspace entry loads statically. The Vite manifest CANNOT
 *    provide this — ManifestChunk has no field listing a chunk's internal
 *    modules, so a regression that merges graphology into a shared chunk
 *    would pass any manifest-only assertion.
 *
 * The plugin also enforces the first-load byte budget: the sum of the
 * closure's uncompressed JS + CSS must stay within the committed baseline
 * +10% (workspace-first-load-baseline.json). A budget that is only
 * "recorded" prevents no regression, so exceeding it FAILS the build.
 */
const FORBIDDEN_WORKSPACE_MODULES: RegExp[] = [
  /src\/features\/GraphViewer/,
  /src\/features\/DocumentManager/,
  /src\/features\/RetrievalView/,
  /src\/stores\/graph/,
  /node_modules\/graphology/,
  /node_modules\/cytoscape/,
  /node_modules\/sigma/,
  /node_modules\/mermaid/
]

const BASELINE_FILE = path.resolve(import.meta.dirname, 'workspace-first-load-baseline.json')
const BASELINE_TOLERANCE = 1.1

// Structural view of the Rollup/Rolldown OutputChunk fields the audit uses
// (the bundler's own type package is not a direct dependency here).
interface AuditOutputChunk {
  type: 'chunk'
  fileName: string
  name?: string
  isEntry: boolean
  imports: string[]
  code: string
  modules: Record<string, unknown>
  viteMetadata?: { importedCss?: Iterable<string> }
}

function workspaceFirstLoadAuditPlugin(): Plugin {
  return {
    name: 'lightrag-workspace-first-load-audit',
    apply: 'build',
    generateBundle(_options, bundle) {
      const chunks = Object.values(bundle).filter(
        (item) => item.type === 'chunk'
      ) as unknown as AuditOutputChunk[]
      const byFileName = new Map(chunks.map((chunk) => [chunk.fileName, chunk]))
      const entry = chunks.find((chunk) => chunk.isEntry && chunk.name === 'workspace')
      if (!entry) {
        this.error('workspace entry chunk not found — did the workspace.html input disappear?')
      }

      // Static-import closure (chunk.imports only; dynamicImports excluded).
      const closure = new Set<string>()
      const walk = (fileName: string) => {
        if (closure.has(fileName)) return
        closure.add(fileName)
        const chunk = byFileName.get(fileName)
        if (!chunk) return
        for (const imported of chunk.imports) walk(imported)
      }
      walk(entry.fileName)

      // Evidence 2: source-module audit over OutputChunk.modules.
      const violations: string[] = []
      let totalBytes = 0
      const cssSeen = new Set<string>()
      for (const fileName of closure) {
        const chunk = byFileName.get(fileName)
        if (!chunk) continue
        totalBytes += Buffer.byteLength(chunk.code, 'utf8')
        for (const cssFile of chunk.viteMetadata?.importedCss ?? []) {
          if (cssSeen.has(cssFile)) continue
          cssSeen.add(cssFile)
          const css = bundle[cssFile]
          if (css && css.type === 'asset' && typeof css.source === 'string') {
            totalBytes += Buffer.byteLength(css.source, 'utf8')
          }
        }
        for (const moduleId of Object.keys(chunk.modules)) {
          const normalized = moduleId.replace(/\\/g, '/')
          for (const pattern of FORBIDDEN_WORKSPACE_MODULES) {
            if (pattern.test(normalized)) {
              violations.push(`${fileName}: ${normalized} (matches ${pattern})`)
            }
          }
        }
      }

      if (violations.length > 0) {
        this.error(
          'Workspace first-load closure contains forbidden modules:\n' +
            violations.join('\n')
        )
      }

      // Byte budget against the committed baseline (+10%).
      if (fs.existsSync(BASELINE_FILE)) {
        const baseline = JSON.parse(fs.readFileSync(BASELINE_FILE, 'utf8'))
        const limit = Math.floor(baseline.firstLoadBytes * BASELINE_TOLERANCE)
        if (totalBytes > limit) {
          this.error(
            `Workspace first-load size ${totalBytes} bytes exceeds baseline ` +
              `${baseline.firstLoadBytes} +10% (${limit}). Either undo the ` +
              'regression or consciously update workspace-first-load-baseline.json.'
          )
        }
        console.log(
          `\n[workspace-audit] first-load ${totalBytes} bytes ` +
            `(baseline ${baseline.firstLoadBytes}, limit ${limit}) — OK`
        )
      } else {
        // Bootstrap: write the baseline on the first build so it can be
        // committed. Subsequent builds enforce it.
        fs.writeFileSync(
          BASELINE_FILE,
          JSON.stringify({ firstLoadBytes: totalBytes }, null, 2) + '\n'
        )
        console.log(
          `\n[workspace-audit] wrote new baseline: ${totalBytes} bytes — commit ` +
            'workspace-first-load-baseline.json'
        )
      }
    }
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  // Dev-only: prefix every proxied endpoint with the simulated site
  // prefix so e.g. `/site01/documents/...` is forwarded to the backend
  // running with LIGHTRAG_API_PREFIX=/site01.
  const devApiPrefix = normalizeApiPrefix(env.VITE_DEV_API_PREFIX)

  return {
    plugins: [
      react(),
      tailwindcss(),
      lightragRuntimeConfigPlugin(env),
      workspaceFirstLoadAuditPlugin()
    ],
    resolve: {
      alias: {
        '@': path.resolve(import.meta.dirname, './src')
      },
      // Force all modules to use the same katex instance
      // This ensures mhchem extension registered in main.tsx is available to rehype-katex
      dedupe: ['katex']
    },
    // Relative base: asset URLs in index.html become `./assets/...` so the
    // built bundle works under any reverse-proxy mount point. The browser
    // resolves them against the current document URL — which means the
    // server MUST serve index.html at a URL ending in '/' (the existing
    // /webui → /webui/ redirect already handles this).
    base: './',
    build: {
      outDir: path.resolve(import.meta.dirname, '../lightrag/api/webui'),
      emptyOutDir: true,
      chunkSizeWarningLimit: 3800,
      // NOTE: `manifest: true` is deliberately NOT set. The first-load audit
      // reads the chunk graph straight off Rollup's in-memory `bundle` in
      // generateBundle, so a manifest file adds no evidence — it would only
      // drop a ~63 KB .vite/manifest.json INSIDE the served static directory,
      // which both UI mounts would then hand out publicly
      // (/webui/.vite/manifest.json), exposing the whole chunk graph and the
      // source module paths for nothing.
      rollupOptions: {
        // One build, two HTML entries — both land at the output directory
        // root so `base: './'` resolves ./assets/... correctly under either
        // server mount (/webui and /workspace).
        input: {
          index: path.resolve(import.meta.dirname, 'index.html'),
          workspace: path.resolve(import.meta.dirname, 'workspace.html')
        },
        // Let Vite handle chunking automatically to avoid circular dependency issues
        output: {
          // Ensure consistent chunk naming format
          chunkFileNames: 'assets/[name]-[hash].js',
          // Entry file naming format
          entryFileNames: 'assets/[name]-[hash].js',
          // Asset file naming format
          assetFileNames: 'assets/[name]-[hash].[ext]'
        }
      }
    },
    server: {
      proxy: env.VITE_API_PROXY === 'true' && env.VITE_API_ENDPOINTS ?
        Object.fromEntries(
          env.VITE_API_ENDPOINTS.split(',').map(endpoint => [
            devApiPrefix + endpoint,
            {
              target: env.VITE_BACKEND_URL || 'http://localhost:9621',
              changeOrigin: true
              // No rewrite: the backend already understands its own prefix
              // via FastAPI's root_path, so forward the path verbatim.
            }
          ])
        ) : {}
    }
  }
})
