import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react-swc'
import path from 'path'
import { defineConfig } from 'vite'
import { webuiPrefix } from '@/lib/constants'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
    // Force all modules to use the same React and katex instances
    // This prevents "Invalid hook call" errors from duplicate React copies
    dedupe: ['react', 'react-dom', 'katex'],
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
        assetFileNames: 'assets/[name]-[hash].[ext]',
      },
    },
  },
  server: {
    proxy:
      import.meta.env.VITE_API_PROXY === 'true' && import.meta.env.VITE_API_ENDPOINTS
        ? Object.fromEntries(
            import.meta.env.VITE_API_ENDPOINTS.split(',').map((endpoint) => [
              endpoint,
              {
                target: import.meta.env.VITE_BACKEND_URL || 'http://localhost:9621',
                changeOrigin: true,
                rewrite:
                  endpoint === '/api'
                    ? (path) => path.replace(/^\/api/, '')
                    : endpoint === '/docs' ||
                        endpoint === '/redoc' ||
                        endpoint === '/openapi.json' ||
                        endpoint === '/static'
                      ? (path) => path
                      : undefined,
              },
            ])
          )
        : {},
  },
})
