import { defineConfig } from 'vite'
import path from 'path'
import { existsSync, mkdirSync } from 'fs'
import { webuiPrefix } from '@/lib/constants'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

const ensureWebuiDir = () => ({
  name: 'ensure-webui-dir',
  apply: 'build' as const,
  buildStart() {
    const outDir = path.resolve(__dirname, '../lightrag/api/webui')
    if (!existsSync(outDir)) {
      mkdirSync(outDir, { recursive: true })
    }
  },
  closeBundle() {
    const outDir = path.resolve(__dirname, '../lightrag/api/webui')
    if (!existsSync(outDir)) {
      mkdirSync(outDir, { recursive: true })
    }
  }
})

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), ensureWebuiDir()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    },
    // Force all modules to use the same katex instance
    // This ensures mhchem extension registered in main.tsx is available to rehype-katex
    dedupe: ['katex']
  },
  // base: import.meta.env.VITE_BASE_URL || '/webui/',
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
    proxy: import.meta.env.VITE_API_PROXY === 'true' && import.meta.env.VITE_API_ENDPOINTS ?
      Object.fromEntries(
        import.meta.env.VITE_API_ENDPOINTS.split(',').map(endpoint => [
          endpoint,
          {
            target: import.meta.env.VITE_BACKEND_URL || 'http://localhost:9621',
            changeOrigin: true,
            rewrite: endpoint === '/api' ?
              (path) => path.replace(/^\/api/, '') :
              endpoint === '/docs' || endpoint === '/redoc' || endpoint === '/openapi.json' || endpoint === '/static' ?
                (path) => path : undefined
          }
        ])
      ) : {}
  }
})
