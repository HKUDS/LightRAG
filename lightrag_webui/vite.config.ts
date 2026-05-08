import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Use relative imports here. The '@' alias is configured in resolve.alias
// below and only takes effect during bundling — Node cannot resolve it when
// loading vite.config.ts. Bun resolves tsconfig paths natively, masking the
// issue, but Node does not.
//
// Likewise, do NOT pull `webuiPrefix` from './src/lib/constants': that value
// is read from `import.meta.env.VITE_*`, which Vite only populates inside
// source files (statically replaced at build). In the config file itself
// `import.meta.env` lacks user `VITE_*` vars, so the constant would silently
// collapse to its fallback. Use `loadEnv()` to get the real value.
import { normalizeWebuiPrefix } from './src/lib/pathPrefix'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src')
      },
      // Force all modules to use the same katex instance
      // This ensures mhchem extension registered in main.tsx is available to rehype-katex
      dedupe: ['katex']
    },
    base: normalizeWebuiPrefix(env.VITE_WEBUI_PREFIX),
    build: {
      outDir: path.resolve(__dirname, '../lightrag/api/webui'),
      emptyOutDir: true,
      chunkSizeWarningLimit: 3800,
      rollupOptions: {
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
            endpoint,
            {
              target: env.VITE_BACKEND_URL || 'http://localhost:9621',
              changeOrigin: true,
              rewrite: endpoint === '/api' ?
                (p: string) => p.replace(/^\/api/, '') :
                endpoint === '/docs' || endpoint === '/redoc' || endpoint === '/openapi.json' || endpoint === '/static' ?
                  (p: string) => p : undefined
            }
          ])
        ) : {}
    }
  }
})
