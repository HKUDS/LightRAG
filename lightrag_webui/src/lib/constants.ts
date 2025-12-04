import { ButtonVariantType } from '@/components/ui/Button'

// Get backend URL from environment variable, or construct it from window location
const getBackendUrl = (): string => {
  // First try environment variable
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL
  }
  
  // Try to construct from current location if we're behind a reverse proxy
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol
    const host = window.location.host
    // Assume API is at :9621 on same host (standard dev port), or check if specified in environment
    return `${protocol}//${host.split(':')[0]}:9621`
  }
  
  // Fallback to localhost on port 9621 (standard development port)
  return 'http://localhost:9621'
}

export const backendBaseUrl = getBackendUrl()
export const webuiPrefix = '/'

export const controlButtonVariant: ButtonVariantType = 'ghost'

export const labelColorDarkTheme = '#B2EBF2'
export const labelColorLightTheme = '#000'
export const LabelColorHighlightedDarkTheme = '#000'

export const nodeColorDisabled = '#E2E2E2'
export const nodeBorderColor = '#EEEEEE'
export const nodeBorderColorSelected = '#F57F17'

export const edgeColorDarkTheme = '#969696'
export const edgeColorSelected = '#F57F17'
export const edgeColorHighlighted = '#B2EBF2'

export const searchResultLimit = 50
export const labelListLimit = 100

// Search History Configuration
export const searchHistoryMaxItems = 500
export const searchHistoryVersion = '1.0'

// API Request Limits
export const popularLabelsDefaultLimit = 300
export const searchLabelsDefaultLimit = 50

// UI Display Limits
export const dropdownDisplayLimit = 300

export const minNodeSize = 4
export const maxNodeSize = 20

export const healthCheckInterval = 15 // seconds

export const defaultQueryLabel = '*'

// reference: https://developer.mozilla.org/en-US/docs/Web/HTTP/MIME_types/Common_types
export const supportedFileTypes = {
  'text/plain': [
    '.txt',
    '.md',
    '.rtf',  //# Rich Text Format
    '.odt', // # OpenDocument Text
    '.tex', // # LaTeX
    '.epub', // # Electronic Publication
    '.html', // # HyperText Markup Language
    '.htm', // # HyperText Markup Language
    '.csv', // # Comma-Separated Values
    '.json', // # JavaScript Object Notation
    '.xml', // # eXtensible Markup Language
    '.yaml', // # YAML Ain't Markup Language
    '.yml', // # YAML
    '.log', // # Log files
    '.conf', // # Configuration files
    '.ini', // # Initialization files
    '.properties', // # Java properties files
    '.sql', // # SQL scripts
    '.bat', // # Batch files
    '.sh', // # Shell scripts
    '.c', // # C source code
    '.cpp', // # C++ source code
    '.py', // # Python source code
    '.java', // # Java source code
    '.js', // # JavaScript source code
    '.ts', // # TypeScript source code
    '.swift', // # Swift source code
    '.go', // # Go source code
    '.rb', // # Ruby source code
    '.php', // # PHP source code
    '.css', // # Cascading Style Sheets
    '.scss',  //# Sassy CSS
    '.less'
  ],
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
}

export const SiteInfo = {
  name: 'LightRAG',
  home: '/',
  github: 'https://github.com/HKUDS/LightRAG'
}
