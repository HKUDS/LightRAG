import { defineConfig } from 'vite'
import baseConfig from './vite.config'
import { mergeConfig } from 'vite'

export default mergeConfig(
  baseConfig,
  defineConfig({
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:9621',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '')
        },
        '/documents': {
          target: 'http://localhost:9621',
          changeOrigin: true
        },
        '/graphs': {
          target: 'http://localhost:9621',
          changeOrigin: true
        },
        '/graph': {
          target: 'http://localhost:9621',
          changeOrigin: true
        },
        '/health': {
          target: 'http://localhost:9621',
          changeOrigin: true
        },
        '/query': {
          target: 'http://localhost:9621',
          changeOrigin: true
        }
      }
    }
  })
)
