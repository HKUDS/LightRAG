import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// WebUI base path - must match the value in src/lib/constants.ts
const webuiPrefix = '/webui/'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '')

  // Backend URL for API proxy (default to local dev server)
  // Use 127.0.0.1 instead of localhost to avoid IPv6 resolution issues
  const backendUrl = env.VITE_BACKEND_URL || 'http://127.0.0.1:9621'

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src')
      }
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
      // Proxy all API routes to the backend during development
      proxy: {
        // API v1 routes (tenant management, knowledge bases, etc.)
        '/api/v1': {
          target: backendUrl,
          changeOrigin: true,
        },
        // Legacy API routes (chat, generate, tags, etc.)
        '/api': {
          target: backendUrl,
          changeOrigin: true,
        },
        // Document operations
        '/documents': {
          target: backendUrl,
          changeOrigin: true,
        },
        // Query operations
        '/query': {
          target: backendUrl,
          changeOrigin: true,
        },
        // Graph operations
        '/graph': {
          target: backendUrl,
          changeOrigin: true,
        },
        // Retrieval operations
        '/retrieval': {
          target: backendUrl,
          changeOrigin: true,
        },
        // Health check
        '/health': {
          target: backendUrl,
          changeOrigin: true,
        },
        // Authentication status
        '/auth-status': {
          target: backendUrl,
          changeOrigin: true,
        },
        // OpenAPI docs
        '/docs': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/redoc': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/openapi.json': {
          target: backendUrl,
          changeOrigin: true,
        },
        // Static assets for Swagger UI
        '/static': {
          target: backendUrl,
          changeOrigin: true,
        },
      }
    }
  }
})
