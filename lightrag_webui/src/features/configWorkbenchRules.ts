export type ChunkKey = 'F' | 'R' | 'V' | 'P'
export type ParserPresetKey = 'native' | 'legacy' | 'raganything' | 'mineru' | 'docling' | 'custom'

export type ParserRuleHint = {
  applicableFileRange: string
  compatibleChunkRange: string
  fallbackChunkStrategy: string
}

const CUSTOM_RULE_HINT = '自定义规则'

const LEGACY_FILE_EXTENSIONS = [
  'txt',
  'md',
  'mdx',
  'pdf',
  'docx',
  'pptx',
  'xlsx',
  'rtf',
  'odt',
  'tex',
  'epub',
  'html',
  'htm',
  'csv',
  'json',
  'xml',
  'yaml',
  'yml',
  'log',
  'conf',
  'ini',
  'properties',
  'sql',
  'bat',
  'sh',
  'c',
  'h',
  'cpp',
  'hpp',
  'py',
  'java',
  'js',
  'ts',
  'swift',
  'go',
  'rb',
  'php',
  'css',
  'scss',
  'less'
]

const NATIVE_FILE_EXTENSIONS = ['docx']

const MINERU_FILE_EXTENSIONS = [
  'pdf',
  'doc',
  'docx',
  'ppt',
  'pptx',
  'xls',
  'xlsx',
  'png',
  'jpg',
  'jpeg',
  'jp2',
  'webp',
  'gif',
  'bmp'
]

const DOCLING_FILE_EXTENSIONS = [
  'pdf',
  'docx',
  'pptx',
  'xlsx',
  'md',
  'html',
  'xhtml',
  'png',
  'jpg',
  'jpeg',
  'tiff',
  'webp',
  'bmp'
]

const RAGANYTHING_EXTENSIONS = [
  'pdf',
  'doc',
  'docx',
  'ppt',
  'pptx',
  'xls',
  'xlsx',
  'png',
  'jpg',
  'jpeg',
  'bmp',
  'tiff',
  'tif',
  'gif',
  'webp',
  'md',
  'txt',
  'html',
  'xhtml'
]

const SIDECAR_CHUNK_RANGE = 'F、R、V、P（P 依赖 .blocks.jsonl sidecar）'
const LEGACY_CHUNK_RANGE = 'F、R、V；P 无 sidecar 时降级为 R'

const formatExtensions = (extensions: string[]): string => extensions.join('、')

const buildExtensionRules = (
  extensions: string[],
  engine: string,
  option: string,
  fallbackEngine = 'legacy',
  fallbackOption = '-R'
): string => {
  return [
    ...extensions.map((extension) => `${extension}:${engine}${option}`),
    `*:${fallbackEngine}${fallbackOption}`
  ].join(',')
}

const legacyOptionForStrategy = (strategy: ChunkKey): string => {
  return `-${strategy === 'P' ? 'R' : strategy}`
}

const fallbackHintFor = (preset: ParserPresetKey, strategy: ChunkKey): string => {
  if (preset === 'custom') return CUSTOM_RULE_HINT
  if (preset !== 'legacy') return 'R 递归字符（未匹配文件走 legacy-R）'
  if (strategy === 'P') return 'R 递归字符（P 无 sidecar 时自动降级）'
  if (strategy === 'V') return 'R 递归字符（V 无 Embedding 时降级）'
  return '无额外规则兜底'
}

export const getParserRuleHint = (
  preset: ParserPresetKey,
  strategy: ChunkKey
): ParserRuleHint => {
  switch (preset) {
    case 'native':
      return {
        applicableFileRange: formatExtensions(NATIVE_FILE_EXTENSIONS),
        compatibleChunkRange: SIDECAR_CHUNK_RANGE,
        fallbackChunkStrategy: fallbackHintFor(preset, strategy)
      }
    case 'legacy':
      return {
        applicableFileRange: formatExtensions(LEGACY_FILE_EXTENSIONS),
        compatibleChunkRange: LEGACY_CHUNK_RANGE,
        fallbackChunkStrategy: fallbackHintFor(preset, strategy)
      }
    case 'raganything':
      return {
        applicableFileRange: formatExtensions(RAGANYTHING_EXTENSIONS),
        compatibleChunkRange: SIDECAR_CHUNK_RANGE,
        fallbackChunkStrategy: fallbackHintFor(preset, strategy)
      }
    case 'mineru':
      return {
        applicableFileRange: formatExtensions(MINERU_FILE_EXTENSIONS),
        compatibleChunkRange: SIDECAR_CHUNK_RANGE,
        fallbackChunkStrategy: fallbackHintFor(preset, strategy)
      }
    case 'docling':
      return {
        applicableFileRange: formatExtensions(DOCLING_FILE_EXTENSIONS),
        compatibleChunkRange: SIDECAR_CHUNK_RANGE,
        fallbackChunkStrategy: fallbackHintFor(preset, strategy)
      }
    case 'custom':
      return {
        applicableFileRange: CUSTOM_RULE_HINT,
        compatibleChunkRange: CUSTOM_RULE_HINT,
        fallbackChunkStrategy: fallbackHintFor(preset, strategy)
      }
  }
}

export const buildParserRuleForPreset = (
  preset: ParserPresetKey,
  strategy: ChunkKey
): string | null => {
  const option = `-${strategy}`
  switch (preset) {
    case 'native':
      return `*:native-te${strategy},*:legacy-R`
    case 'legacy':
      return `*:legacy${legacyOptionForStrategy(strategy)}`
    case 'raganything':
      return buildExtensionRules(RAGANYTHING_EXTENSIONS, 'raganything', option)
    case 'mineru':
      return buildExtensionRules(MINERU_FILE_EXTENSIONS, 'mineru', option)
    case 'docling':
      return buildExtensionRules(DOCLING_FILE_EXTENSIONS, 'docling', option)
    case 'custom':
      return null
  }
}
