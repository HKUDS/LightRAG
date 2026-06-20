import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { extractQualityBefore } from './kgMaintenanceDisplay'
import { ExecutionPanel, ValidationPanel } from './ExecutionAndValidationPanels'

describe('execution and validation panels', () => {
  test('execution panel renders focused accepted-change action and apply result', () => {
    const markup = renderToStaticMarkup(
      <ExecutionPanel
        acceptedChanges={`# Accepted Changes

## proposal-1

- 写入证据字段`}
        applyResult={`Applied: 2
hierarchy_missing_branch_count: 4 -> 0`}
        executing={false}
        onExecute={() => undefined}
      />
    )

    expect(markup).toContain('执行已接受变更')
    expect(markup).toContain('执行变更')
    expect(markup).toContain('Applied: 2')
    expect(markup).not.toContain('improvement_backlog.md')
  })

  test('execution panel summarizes real bullet-form apply results', () => {
    const markup = renderToStaticMarkup(
      <ExecutionPanel
        acceptedChanges={`# Accepted Changes

## proposal-1

- 写入证据字段`}
        applyResult={`- Applied: 2
- Blocked: 0
- hierarchy_missing_branch_count: 4 -> 0`}
        executing={false}
        onExecute={() => undefined}
      />
    )

    expect(markup).toContain('Applied: 2')
    expect(markup).not.toContain('尚未执行写入')
  })

  test('execution panel disables action when accepted changes have no headings', () => {
    const markup = renderToStaticMarkup(
      <ExecutionPanel
        acceptedChanges="- accepted text without proposal heading"
        applyResult=""
        executing={false}
        onExecute={() => undefined}
      />
    )

    expect(markup).toContain('暂无可执行的已接受变更')
    expect(markup).toContain('disabled=""')
  })

  test('execution panel ignores accepted headings with extra title text', () => {
    const invalidMarkup = renderToStaticMarkup(
      <ExecutionPanel
        acceptedChanges={`# Accepted Changes

## p1 accepted

- 这不是后端可执行的 proposal ID 标题`}
        applyResult=""
        executing={false}
        onExecute={() => undefined}
      />
    )

    const validMarkup = renderToStaticMarkup(
      <ExecutionPanel
        acceptedChanges={`# Accepted Changes

## p1

## prop-a

## prop.a-1

## _p1

## .p1

## -p1`}
        applyResult=""
        executing={false}
        onExecute={() => undefined}
      />
    )

    expect(invalidMarkup).toContain('暂无可执行的已接受变更')
    expect(invalidMarkup).toContain('disabled=""')
    expect(validMarkup).toContain('6 条已接受变更等待写入')
    expect(validMarkup).not.toContain('disabled=""')
  })

  test('validation panel renders quality deltas and already achieved result', () => {
    const applyResult = `Applied: 0
overall: 88 -> 97
hierarchy_missing_branch_count: 4 -> 0`

    const markup = renderToStaticMarkup(
      <ValidationPanel
        qualityBefore={extractQualityBefore(applyResult)}
        qualityAfter={{
          overall: 97,
          metrics: {
            hierarchy_missing_branch_count: 0
          }
        }}
        applyResult={applyResult}
      />
    )

    expect(markup).toContain('验证结果')
    expect(markup).toContain('88 → 97')
    expect(markup).toContain('4 → 0')
    expect(markup).toContain('没有新增写入，但当前质量已达标')
  })

  test('validation panel handles real bullet-form zero-apply results', () => {
    const applyResult = `- Applied: 0
- Blocked: 0
- hierarchy_missing_branch_count: 0 -> 0`

    const markup = renderToStaticMarkup(
      <ValidationPanel
        qualityBefore={extractQualityBefore(applyResult)}
        qualityAfter={{
          metrics: {
            hierarchy_missing_branch_count: 0
          }
        }}
        applyResult={applyResult}
      />
    )

    expect(markup).toContain('0 → 0')
    expect(markup).toContain('没有新增写入，但当前质量已达标')
  })

  test('validation panel renders total placeholder when backend result has no overall delta', () => {
    const applyResult = `- Applied: 2
- Blocked: 0
- hierarchy_missing_branch_count: 4 -> 0`

    const markup = renderToStaticMarkup(
      <ValidationPanel
        qualityBefore={extractQualityBefore(applyResult)}
        qualityAfter={{
          overall: 97,
          metrics: {
            hierarchy_missing_branch_count: 0
          }
        }}
        applyResult={applyResult}
      />
    )

    expect(markup).toContain('— → 97')
    expect(markup).toContain('4 → 0')
  })

  test('extractQualityBefore parses metric before-values from apply result deltas', () => {
    expect(
      extractQualityBefore(`Applied: 2
hierarchy_missing_branch_count: 4 -> 0`)
    ).toEqual({
      metrics: {
        hierarchy_missing_branch_count: 4
      }
    })
  })

  test('extractQualityBefore parses real bullet-form metric before-values', () => {
    expect(
      extractQualityBefore(`- Applied: 2
- Blocked: 0
- hierarchy_missing_branch_count: 4 -> 0`)
    ).toEqual({
      metrics: {
        hierarchy_missing_branch_count: 4
      }
    })
  })
})
