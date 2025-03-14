import { defineConfig } from 'vite'
import path from 'path'
import { webuiPrefix } from '@/lib/constants'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
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
    emptyOutDir: true
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
              endpoint === '/docs' || endpoint === '/openapi.json' ?
                (path) => path : undefined
          }
        ])
      ) : {}
  }
})
