import type { ButtonVariantType } from '@/components/ui/Button'

export const backendBaseUrl = ''
export const webuiPrefix = '/webui/'

export const controlButtonVariant: ButtonVariantType = 'ghost'

// Dark theme graph palette tuned for contrast on charcoal backgrounds
export const labelColorDarkTheme = '#E5ECFF'
export const LabelColorHighlightedDarkTheme = '#0F172A'
export const labelColorLightTheme = '#000'

export const nodeColorDisabled = '#9CA3AF'
export const nodeBorderColor = '#CBD5E1'
export const nodeBorderColorSelected = '#F97316'
export const nodeBorderColorHiddenConnections = '#F59E0B' // Amber color for nodes with hidden connections

export const edgeColorDarkTheme = '#4B5563'
export const edgeColorSelected = '#F97316'
export const edgeColorHighlightedDarkTheme = '#F59E0B'
export const edgeColorHighlightedLightTheme = '#F57F17'

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
    '.rtf', //# Rich Text Format
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
    '.scss', //# Sassy CSS
    '.less',
  ],
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
}

export const SiteInfo = {
  name: 'LightRAG',
  home: '/',
  github: 'https://github.com/HKUDS/LightRAG',
}
