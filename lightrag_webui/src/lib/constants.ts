import { ButtonVariantType } from '@/components/ui/Button'
import { normalizeApiPrefix, normalizeWebuiPrefix } from '@/lib/pathPrefix'
import { getRuntimeApiPrefix, getRuntimeWebuiPrefix } from '@/lib/runtimeConfig'

export const backendBaseUrl = normalizeApiPrefix(getRuntimeApiPrefix())
export const webuiPrefix = normalizeWebuiPrefix(getRuntimeWebuiPrefix())

export const controlButtonVariant: ButtonVariantType = 'ghost'

export const labelColorDarkTheme = '#FFFFFF'
export const LabelColorHighlightedDarkTheme = '#000000'
export const labelColorLightTheme = '#000'

export const nodeColorDisabled = '#E2E2E2'
export const nodeBorderColor = '#EEEEEE'
export const nodeBorderColorSelected = '#F57F17'

export const edgeColorDarkTheme = '#888888'
export const edgeColorSelected = '#F57F17'
export const edgeColorHighlightedDarkTheme = '#F57F17'
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
    '.textpack', // # Markdown Bundle(zip)
    '.mdx', // # MDX (Markdown + JSX)
    '.rtf', // # Rich Text Format
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
    '.h', // # C header
    '.cpp', // # C++ source code
    '.hpp', // # C++ header
    '.py', // # Python source code
    '.java', // # Java source code
    '.js', // # JavaScript source code
    '.ts', // # TypeScript source code
    '.swift', // # Swift source code
    '.go', // # Go source code
    '.rb', // # Ruby source code
    '.php', // # PHP source code
    '.css', // # Cascading Style Sheets
    '.scss', // # Sassy CSS
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

// --- Graph layout performance thresholds ------------------------------------
// Shared by the initial FA2 layout (GraphControl) and the manual worker
// layouts (LayoutsControl) so the two cannot drift.

// Above this node count, node labels are forced off regardless of the
// showNodeLabel setting (the hovered node's label is still drawn by sigma's
// hover layer). Rendering thousands of labels is a major large-graph slowdown.
export const LABEL_RENDER_LIMIT = 2000

// Above this node count, layout switches assign positions directly instead of
// animating: animateNodes interpolates every node per frame on the main thread.
export const ANIMATE_NODE_LIMIT = 5000

// Edge-count threshold that switches the graph between "small-graph experience"
// and "large-graph performance". At or below it edges render as curves and edge
// events (hover/click picking) follow the user setting; above it edges render
// straight and edge events are fully disabled (no picking buffer allocated).
// Shared by GraphControl (defaultEdgeType), GraphViewer (enableEdgeEvents
// gating) and Settings (greying the Edge Events menu item) so they cannot drift.
export const EDGE_PERF_LIMIT = 5000

// Time budget (ms) a relaxing worker layout runs before it is stopped. Scales
// with graph size, capped so huge graphs don't run unbounded.
export const workerBudgetMs = (order: number): number => Math.min(1500 + order / 10, 10000)

// One-time system-suggested user prompts, injected once into userPromptHistory
// (for both fresh installs and upgrades). See settings store version 20 migration.
export const suggestedUserPrompts: string[] = [
  'Ignore the `References Section Format` instruction in the system prompt, and do not include a `References` section in the response.',
  'For inline citations, use the footnote marker syntax `[^1]`, where the `^` preceding the identifier indicates a footnote reference. When multiple citations are required at a single location, each ID should be enclosed in separate footnote markers (e.g., `[^1][^2][^3]`).'
]
