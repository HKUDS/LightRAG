/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_PROXY: string
  readonly VITE_API_ENDPOINTS: string
  readonly VITE_BACKEND_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
