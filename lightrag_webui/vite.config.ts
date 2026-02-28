import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// Use relative import instead of '@/lib/constants' path alias.
// The '@' alias is configured in this file's resolve.alias and only takes effect
// during bundling — Node.js cannot resolve it when loading vite.config.ts itself.
// Bun resolves tsconfig paths natively, masking the issue, but Node.js does not.
import { webuiPrefix } from './src/lib/constants'

// https://vite.dev/config/
// Use functional config form so we can call loadEnv(). import.meta.env is only
// available inside Bun's runtime; Node.js leaves it undefined, crashing the build.
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
    // base: env.VITE_BASE_URL || '/webui/',
    base: webuiPrefix,
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
