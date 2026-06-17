import { describe, expect, test } from 'bun:test'

import { buildParserRuleForPreset, getParserRuleHint } from '@/features/configWorkbenchRules'

describe('ConfigWorkbench parser rules', () => {
  test('keeps selected chunking for supported parser files and uses documented legacy-R fallback', () => {
    const rule = buildParserRuleForPreset('raganything', 'P')

    expect(rule).toContain('pdf:raganything-P')
    expect(rule).toContain('docx:raganything-P')
    expect(rule).toContain('html:raganything-P')
    expect(rule).not.toContain('*:raganything-P')
    expect(rule?.endsWith('*:legacy-R')).toBe(true)
    expect(buildParserRuleForPreset('native', 'P')).toBe('*:native-teP,*:legacy-R')
    expect(buildParserRuleForPreset('legacy', 'P')).toBe('*:legacy-R')
  })

  test('describes parser rules with documented chunking compatibility and fallback', () => {
    expect(getParserRuleHint('raganything', 'P')).toEqual({
      applicableFileRange:
        'pdf、doc、docx、ppt、pptx、xls、xlsx、png、jpg、jpeg、bmp、tiff、tif、gif、webp、md、txt、html、xhtml',
      compatibleChunkRange: 'F、R、V、P（P 依赖 .blocks.jsonl sidecar）',
      fallbackChunkStrategy: 'R 递归字符（未匹配文件走 legacy-R）'
    })

    expect(getParserRuleHint('legacy', 'P')).toEqual({
      applicableFileRange:
        'txt、md、mdx、pdf、docx、pptx、xlsx、rtf、odt、tex、epub、html、htm、csv、json、xml、yaml、yml、log、conf、ini、properties、sql、bat、sh、c、h、cpp、hpp、py、java、js、ts、swift、go、rb、php、css、scss、less',
      compatibleChunkRange: 'F、R、V；P 无 sidecar 时降级为 R',
      fallbackChunkStrategy: 'R 递归字符（P 无 sidecar 时自动降级）'
    })
  })
})
