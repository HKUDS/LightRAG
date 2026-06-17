# Medical KG Browse Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a source-grounded, layered medical browsing graph for the influenza scenario while preserving raw KG data and the generic medical KG framework.

**Architecture:** Build on the existing medical KG profile. Backend code adds controlled category normalization, top-level medical hierarchy spine, and additive `metadata.medical_browse` projection. WebUI code consumes that metadata to show relation-aware rows, collapsed groups, and stable medical browse layout without changing the raw graph API contract.

**Tech Stack:** Python 3, FastAPI, Pydantic graph payloads, pytest, React 19, TypeScript, Graphology/Sigma, Bun test, Tailwind.

---

## File Structure

- Modify `D:\LightRAG\lightrag\medical_kg\ontology.py`
  - Owns first-level category definitions, extension category normalization, relation display labels, and helper functions shared by hierarchy/projection.
- Modify `D:\LightRAG\lightrag\medical_kg\hierarchy.py`
  - Adds top-level category spine edges and stable `MedicalGroup` metadata for category/subgroup nodes.
- Modify `D:\LightRAG\lightrag\medical_kg\graph_projection.py`
  - Keeps `medical_groups` and adds `metadata.medical_browse` with root, categories, subgroups, collapsed groups, representative examples, and relation display records.
- Modify `D:\LightRAG\tests\extraction\test_medical_kg_hierarchy.py`
  - Covers influenza first-level category spine and controlled extension category normalization.
- Modify `D:\LightRAG\tests\api\routes\test_medical_graph_projection.py`
  - Covers browse metadata, collapsed group examples, relation direction, and non-mutation.
- Modify `D:\LightRAG\lightrag_webui\src\api\lightrag.ts`
  - Adds TypeScript types for `medical_browse` and includes `medical_browse=true` in graph requests.
- Modify `D:\LightRAG\lightrag_webui\src\stores\graph.ts`
  - Merges `medical_browse` metadata during node expansion.
- Create `D:\LightRAG\lightrag_webui\src\components\graph\medicalBrowseGraph.ts`
  - Pure functions for applying browse metadata to raw nodes/edges, collapsed group labels, and deterministic medical layout positions.
- Create `D:\LightRAG\lightrag_webui\src\components\graph\medicalBrowseGraph.test.ts`
  - Tests collapsed group labels and medical layout roles.
- Modify `D:\LightRAG\lightrag_webui\src\hooks\useLightragGraph.tsx`
  - Applies medical browse projection before Sigma graph creation and uses stable role-based sizing/positions.
- Modify `D:\LightRAG\lightrag_webui\src\components\graph\medicalRelationGroups.ts`
  - Adds relation display names and selected-node direction details.
- Modify `D:\LightRAG\lightrag_webui\src\components\graph\medicalRelationGroups.test.ts`
  - Covers row labels such as `临床表现：高热不退` and full triple text.
- Modify `D:\LightRAG\lightrag_webui\src\components\graph\PropertiesView.tsx`
  - Renders relation-aware rows instead of generic `邻接`.
- Modify `D:\LightRAG\lightrag_webui\src\locales\zh.json`
  - Adds labels for medical browse groups and relation detail fields.
- Modify `D:\LightRAG\lightrag_webui\src\locales\en.json`
  - Adds English fallbacks for the same UI keys.
- Modify `D:\LightRAG\docs\MedicalKGProfile-zh.md`
  - Documents clean-workspace rebuild and medical browse behavior.
- Create `D:\LightRAG\docs\MedicalKGCleanWorkspace-zh.md`
  - Provides conservative steps for deleting unused old workspace data and rebuilding from imported documents.

---

## Task 1: Backend Taxonomy and Hierarchy Spine

**Files:**
- Modify: `D:\LightRAG\lightrag\medical_kg\ontology.py`
- Modify: `D:\LightRAG\lightrag\medical_kg\hierarchy.py`
- Test: `D:\LightRAG\tests\extraction\test_medical_kg_hierarchy.py`

- [ ] **Step 1: Write the failing hierarchy tests**

Add these tests to `test_medical_kg_hierarchy.py`:

```python
def test_build_medical_hierarchy_edges_adds_influenza_top_level_category_spine() -> None:
    nodes = {
        "流感病毒": [_node("流感病毒", "MedicalGroup")],
        "全身症状": [_node("全身症状", "MedicalGroup")],
        "呼吸道症状": [_node("呼吸道症状", "MedicalGroup")],
        "呼吸系统并发症": [_node("呼吸系统并发症", "MedicalGroup")],
    }

    next_nodes, hierarchy_edges = build_medical_hierarchy_edges(nodes)

    assert ("流感病毒", "病原体") in hierarchy_edges
    assert ("全身症状", "临床表现") in hierarchy_edges
    assert ("呼吸道症状", "临床表现") in hierarchy_edges
    assert ("呼吸系统并发症", "并发症/重症") in hierarchy_edges
    assert next_nodes["病原体"][0]["medical_group"] == "pathogen"
    assert next_nodes["临床表现"][0]["medical_group"] == "clinical_manifestation"
    assert next_nodes["并发症/重症"][0]["medical_group"] == "complication_severity"


def test_normalize_medical_category_key_maps_controlled_extensions() -> None:
    from lightrag.medical_kg.ontology import normalize_medical_category_key

    assert normalize_medical_category_key("复诊/随访") == "follow_up"
    assert normalize_medical_category_key("照护") == "nursing_care"
    assert normalize_medical_category_key("用药禁忌") == "contraindication"
    assert normalize_medical_category_key("未收录结构") == "other_medical"
```

- [ ] **Step 2: Run the tests and confirm red**

Run:

```powershell
Set-Location D:\LightRAG
.\scripts\test.sh tests\extraction\test_medical_kg_hierarchy.py
```

Expected: the new tests fail because top-level category parents and `normalize_medical_category_key()` are not implemented yet.

- [ ] **Step 3: Implement category definitions**

In `ontology.py`, add `MedicalCategory`, `TOP_LEVEL_MEDICAL_CATEGORIES`, `EXTENSION_MEDICAL_CATEGORIES`, `CATEGORY_ALIAS_MAP`, `normalize_medical_category_key()`, and `medical_category_label()`:

```python
@dataclass(frozen=True)
class MedicalCategory:
    key: str
    label: str
    aliases: tuple[str, ...] = ()


TOP_LEVEL_MEDICAL_CATEGORIES: tuple[MedicalCategory, ...] = (
    MedicalCategory("pathogen", "病原体", ("病原", "病原学")),
    MedicalCategory("transmission_epidemiology", "传播/流行病学", ("传播", "流行病学")),
    MedicalCategory("clinical_manifestation", "临床表现", ("症状", "体征", "症状表现")),
    MedicalCategory("complication_severity", "并发症/重症", ("并发症", "重症", "危重症")),
    MedicalCategory("diagnosis_testing", "诊断/检查", ("诊断", "检查", "检测", "检验")),
    MedicalCategory("treatment", "治疗", ("用药", "治疗方案", "抗病毒治疗")),
    MedicalCategory("prevention", "预防", ("疫苗", "预防措施", "隔离")),
    MedicalCategory("high_risk_population", "高危人群", ("高危", "风险人群", "特殊人群")),
    MedicalCategory("guideline_evidence", "指南/证据来源", ("指南", "证据来源", "诊疗方案")),
)

EXTENSION_MEDICAL_CATEGORIES: tuple[MedicalCategory, ...] = (
    MedicalCategory("differential_diagnosis", "鉴别诊断", ("鉴别", "相似疾病")),
    MedicalCategory("nursing_care", "护理", ("照护", "居家护理")),
    MedicalCategory("follow_up", "随访", ("复诊", "复诊/随访", "随访观察")),
    MedicalCategory("rehabilitation", "康复", ("恢复期管理",)),
    MedicalCategory("contraindication", "禁忌证", ("用药禁忌", "禁忌", "不宜使用")),
    MedicalCategory("adverse_reaction", "不良反应", ("副作用", "药物不良反应")),
    MedicalCategory("public_health", "公共卫生处置", ("报告", "隔离管理", "学校防控")),
)
```

- [ ] **Step 4: Implement hierarchy spine**

In `hierarchy.py`, add parent mappings:

```python
TOP_LEVEL_PARENT_BY_CHILD: dict[str, str] = {
    "流感病毒": "病原体",
    "甲型流感病毒": "病原体",
    "乙型流感病毒": "病原体",
    "全身症状": "临床表现",
    "呼吸道症状": "临床表现",
    "消化道症状": "临床表现",
    "呼吸系统并发症": "并发症/重症",
    "心脏并发症": "并发症/重症",
    "神经系统并发症": "并发症/重症",
}
```

Then combine it with existing `PARENT_BY_CHILD` in the builder so transitive completion can create `发热 -> 全身症状 -> 临床表现`.

- [ ] **Step 5: Run the focused backend tests**

Run:

```powershell
Set-Location D:\LightRAG
.\scripts\test.sh tests\extraction\test_medical_kg_hierarchy.py
```

Expected: all hierarchy tests pass.

---

## Task 2: Backend Medical Browse Projection

**Files:**
- Modify: `D:\LightRAG\lightrag\medical_kg\graph_projection.py`
- Modify: `D:\LightRAG\tests\api\routes\test_medical_graph_projection.py`
- Modify: `D:\LightRAG\lightrag\api\routers\graph_routes.py`

- [ ] **Step 1: Write browse metadata tests**

Add tests to `test_medical_graph_projection.py`:

```python
def test_project_medical_graph_adds_medium_expanded_browse_metadata():
    from lightrag.medical_kg.graph_projection import project_medical_graph

    payload = {
        "nodes": [
            {"id": "流行性感冒", "labels": ["Disease"], "properties": {"entity_type": "Disease", "entity_name": "流行性感冒"}},
            {"id": "临床表现", "labels": ["MedicalGroup"], "properties": {"entity_type": "MedicalGroup", "medical_group": "clinical_manifestation"}},
            {"id": "全身症状", "labels": ["MedicalGroup"], "properties": {"entity_type": "MedicalGroup", "medical_group": "clinical_manifestation"}},
            {"id": "呼吸道症状", "labels": ["MedicalGroup"], "properties": {"entity_type": "MedicalGroup", "medical_group": "clinical_manifestation"}},
            {"id": "发热", "labels": ["Symptom"], "properties": {"entity_type": "Symptom", "entity_name": "发热"}},
            {"id": "咳嗽", "labels": ["Symptom"], "properties": {"entity_type": "Symptom", "entity_name": "咳嗽"}},
            {"id": "咽痛", "labels": ["Symptom"], "properties": {"entity_type": "Symptom", "entity_name": "咽痛"}},
            {"id": "流涕", "labels": ["Symptom"], "properties": {"entity_type": "Symptom", "entity_name": "流涕"}},
        ],
        "edges": [
            {"id": "e0", "source": "流行性感冒", "target": "临床表现", "type": "related", "properties": {"keywords": "临床表现"}},
            {"id": "e1", "source": "临床表现", "target": "全身症状", "type": "related", "properties": {"keywords": "属于"}},
            {"id": "e2", "source": "临床表现", "target": "呼吸道症状", "type": "related", "properties": {"keywords": "属于"}},
            {"id": "e3", "source": "发热", "target": "全身症状", "type": "related", "properties": {"keywords": "属于"}},
            {"id": "e4", "source": "咳嗽", "target": "呼吸道症状", "type": "related", "properties": {"keywords": "属于"}},
            {"id": "e5", "source": "咽痛", "target": "呼吸道症状", "type": "related", "properties": {"keywords": "属于"}},
            {"id": "e6", "source": "流涕", "target": "呼吸道症状", "type": "related", "properties": {"keywords": "属于"}},
        ],
        "is_truncated": False,
    }

    projected = project_medical_graph(payload, include_browse=True)

    browse = projected["metadata"]["medical_browse"]
    assert browse["root_id"] == "流行性感冒"
    assert browse["default_depth"] == "medium"
    assert browse["node_roles"]["流行性感冒"] == "root"
    assert browse["node_roles"]["临床表现"] == "category"
    assert browse["node_roles"]["呼吸道症状"] == "subgroup"
    assert browse["collapsed_groups"][0]["label"] == "呼吸道症状 (3): 咳嗽、咽痛、流涕"
    assert browse["collapsed_groups"][0]["child_ids"] == ["咳嗽", "咽痛", "流涕"]
    assert browse["relation_details"]["e0"]["display"] == "临床表现：临床表现"
    assert browse["relation_details"]["e0"]["triple"] == "流行性感冒 - 临床表现 -> 临床表现"
```

- [ ] **Step 2: Run the tests and confirm red**

Run:

```powershell
Set-Location D:\LightRAG
.\scripts\test.sh tests\api\routes\test_medical_graph_projection.py
```

Expected: the new browse metadata test fails because `include_browse` and `metadata.medical_browse` do not exist.

- [ ] **Step 3: Implement `medical_browse` metadata**

Update `project_medical_graph(graph_payload, *, include_browse: bool = False)` so existing callers keep current behavior. When `include_browse=True`, add:

```python
payload["metadata"]["medical_browse"] = {
    "root_id": root_id,
    "default_depth": "medium",
    "category_order": [category.key for category in TOP_LEVEL_MEDICAL_CATEGORIES],
    "node_roles": node_roles,
    "collapsed_groups": collapsed_groups,
    "relation_details": relation_details,
}
```

Use source order from `payload["nodes"]` for representative examples. Use relation keywords through `normalize_relation_keyword()` for display labels. Keep direct graph nodes and edges unchanged.

- [ ] **Step 4: Add API query flag**

In `graph_routes.py`, add:

```python
medical_browse: bool = Query(
    False, description="Return display-oriented medical browsing metadata"
)
```

Then call:

```python
if medical_view or medical_browse:
    return project_medical_graph(graph, include_browse=medical_browse)
```

- [ ] **Step 5: Add route test for browse flag**

Extend the route test to call:

```python
response = client.get(
    "/graphs?label=流行性感冒&max_depth=2&max_nodes=20&medical_view=true&medical_browse=true",
    headers=_HEADERS,
)
assert "medical_browse" in response.json()["metadata"]
```

- [ ] **Step 6: Run backend projection tests**

Run:

```powershell
Set-Location D:\LightRAG
.\scripts\test.sh tests\api\routes\test_medical_graph_projection.py tests\extraction\test_medical_kg_hierarchy.py
```

Expected: all tests pass.

---

## Task 3: WebUI API Types and Metadata Merge

**Files:**
- Modify: `D:\LightRAG\lightrag_webui\src\api\lightrag.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\api\lightrag.test.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\stores\graph.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\stores\graph.test.ts`

- [ ] **Step 1: Write API path test**

In `lightrag.test.ts`, update the graph API test:

```ts
expect(apiModule.buildGraphQueryPath('COVID-19 & fever', 2, 150)).toBe(
  '/graphs?label=COVID-19%20%26%20fever&max_depth=2&max_nodes=150&medical_view=true&medical_browse=true'
)
```

- [ ] **Step 2: Write metadata merge test**

In `graph.test.ts`, add:

```ts
test('merges medical browse metadata from expanded graph metadata', () => {
  const merged = mergeGraphMetadata(
    {
      medical_browse: {
        root_id: '流行性感冒',
        default_depth: 'medium',
        category_order: ['clinical_manifestation'],
        node_roles: { 流行性感冒: 'root', 临床表现: 'category' },
        collapsed_groups: [],
        relation_details: {}
      }
    },
    {
      medical_browse: {
        root_id: '流行性感冒',
        default_depth: 'medium',
        category_order: ['clinical_manifestation'],
        node_roles: { 呼吸道症状: 'subgroup' },
        collapsed_groups: [{ id: 'cg:resp', label: '呼吸道症状 (2): 咳嗽、咽痛', child_ids: ['咳嗽', '咽痛'], count: 2, examples: ['咳嗽', '咽痛'], parent_id: '呼吸道症状' }],
        relation_details: { e1: { display: '临床表现：咳嗽', triple: '流行性感冒 - 临床表现 -> 咳嗽', relation: '临床表现', source: '流行性感冒', target: '咳嗽' } }
      }
    }
  )

  expect(merged?.medical_browse?.node_roles).toEqual({
    流行性感冒: 'root',
    临床表现: 'category',
    呼吸道症状: 'subgroup'
  })
  expect(merged?.medical_browse?.collapsed_groups).toHaveLength(1)
  expect(merged?.medical_browse?.relation_details.e1.display).toBe('临床表现：咳嗽')
})
```

- [ ] **Step 3: Run tests and confirm red**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\api\lightrag.test.ts src\stores\graph.test.ts
```

Expected: graph path and metadata merge tests fail before implementation.

- [ ] **Step 4: Implement TypeScript types and path**

In `lightrag.ts`, add:

```ts
export type LightragMedicalBrowseCollapsedGroup = {
  id: string
  label: string
  parent_id: string
  child_ids: string[]
  count: number
  examples: string[]
}

export type LightragMedicalBrowseRelationDetail = {
  source: string
  target: string
  relation: string
  display: string
  triple: string
}

export type LightragMedicalBrowseMetadata = {
  root_id?: string
  default_depth?: 'medium' | string
  category_order?: string[]
  node_roles?: Record<string, 'root' | 'category' | 'subgroup' | 'leaf' | string>
  collapsed_groups?: LightragMedicalBrowseCollapsedGroup[]
  relation_details?: Record<string, LightragMedicalBrowseRelationDetail>
}
```

Add `medical_browse?: LightragMedicalBrowseMetadata` to `LightragGraphMetadata`. Update `buildGraphQueryPath()` to append `&medical_browse=true` when `medicalView` is true.

- [ ] **Step 5: Implement metadata merge**

In `mergeGraphMetadata()`, merge `medical_browse` with:

```ts
merged.medical_browse = {
  ...current.medical_browse,
  ...incoming.medical_browse,
  category_order: Array.from(new Set([
    ...(current.medical_browse?.category_order ?? []),
    ...(incoming.medical_browse?.category_order ?? [])
  ])),
  node_roles: {
    ...(current.medical_browse?.node_roles ?? {}),
    ...(incoming.medical_browse?.node_roles ?? {})
  },
  collapsed_groups: mergeCollapsedGroups(
    current.medical_browse?.collapsed_groups,
    incoming.medical_browse?.collapsed_groups
  ),
  relation_details: {
    ...(current.medical_browse?.relation_details ?? {}),
    ...(incoming.medical_browse?.relation_details ?? {})
  }
}
```

Use a helper keyed by collapsed group `id` to dedupe.

- [ ] **Step 6: Run focused WebUI tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\api\lightrag.test.ts src\stores\graph.test.ts
```

Expected: tests pass.

---

## Task 4: Relation-Aware Properties Panel Rows

**Files:**
- Modify: `D:\LightRAG\lightrag_webui\src\components\graph\medicalRelationGroups.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\components\graph\medicalRelationGroups.test.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\components\graph\PropertiesView.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\locales\zh.json`
- Modify: `D:\LightRAG\lightrag_webui\src\locales\en.json`

- [ ] **Step 1: Write relation display tests**

Add to `medicalRelationGroups.test.ts`:

```ts
test('builds concise relation row text and full directed triple', () => {
  const groups = groupMedicalRelations([
    {
      id: '高热不退',
      label: '高热不退',
      edgeId: 'e1',
      selectedNodeId: '流行性感冒',
      sourceId: '流行性感冒',
      targetId: '高热不退',
      edgeKeywords: '临床表现',
      neighborEntityType: 'Symptom',
      neighborLabels: ['Symptom']
    }
  ])

  expect(groups[0].relations[0].displayName).toBe('临床表现')
  expect(groups[0].relations[0].displayValue).toBe('高热不退')
  expect(groups[0].relations[0].triple).toBe('流行性感冒 - 临床表现 -> 高热不退')
})
```

- [ ] **Step 2: Run test and confirm red**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\graph\medicalRelationGroups.test.ts
```

Expected: fails because `MedicalRelation` lacks direction/display fields.

- [ ] **Step 3: Extend relation grouping model**

In `medicalRelationGroups.ts`, extend `MedicalRelation`:

```ts
edgeId?: string
selectedNodeId?: string
sourceId?: string
targetId?: string
displayName?: string
displayValue?: string
triple?: string
```

Add `buildRelationDisplay(relation)` that normalizes missing keywords to `相关` and sets:

```ts
displayName: relation.edgeKeywords?.trim() || '相关'
displayValue: relation.label
triple: `${source} - ${displayName} -> ${target}`
```

Return relations after applying this helper.

- [ ] **Step 4: Update `PropertiesView` data collection**

In `refineNodeProperties()`, push these fields:

```ts
edgeId: edge.id,
selectedNodeId: node.id,
sourceId: edge.source,
targetId: edge.target,
edgeKeywords: edge.properties?.keywords
```

In the render loop, replace:

```tsx
name={t('graphPanel.propertiesView.node.neighbor', '邻接')}
value={relation.label}
```

with:

```tsx
name={relation.displayName || t('graphPanel.propertiesView.node.related', '相关')}
value={relation.displayValue || relation.label}
tooltip={relation.triple || relation.label}
```

- [ ] **Step 5: Run focused WebUI test and lint**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\graph\medicalRelationGroups.test.ts
bun run lint
```

Expected: tests and lint pass.

---

## Task 5: WebUI Medical Browse Projection and Layout

**Files:**
- Create: `D:\LightRAG\lightrag_webui\src\components\graph\medicalBrowseGraph.ts`
- Create: `D:\LightRAG\lightrag_webui\src\components\graph\medicalBrowseGraph.test.ts`
- Modify: `D:\LightRAG\lightrag_webui\src\hooks\useLightragGraph.tsx`

- [ ] **Step 1: Write pure function tests**

Create `medicalBrowseGraph.test.ts`:

```ts
import { describe, expect, test } from 'bun:test'
import { applyMedicalBrowseProjection } from './medicalBrowseGraph'
import type { LightragGraphType } from '@/api/lightrag'

describe('applyMedicalBrowseProjection', () => {
  test('adds collapsed group display nodes and hides collapsed leaf nodes', () => {
    const graph: LightragGraphType = {
      nodes: [
        { id: '流行性感冒', labels: ['Disease'], properties: { entity_type: 'Disease' } },
        { id: '呼吸道症状', labels: ['MedicalGroup'], properties: { entity_type: 'MedicalGroup' } },
        { id: '咳嗽', labels: ['Symptom'], properties: { entity_type: 'Symptom' } },
        { id: '咽痛', labels: ['Symptom'], properties: { entity_type: 'Symptom' } },
        { id: '流涕', labels: ['Symptom'], properties: { entity_type: 'Symptom' } }
      ],
      edges: [],
      metadata: {
        medical_browse: {
          root_id: '流行性感冒',
          default_depth: 'medium',
          node_roles: { 流行性感冒: 'root', 呼吸道症状: 'subgroup' },
          collapsed_groups: [
            { id: 'collapse:resp', parent_id: '呼吸道症状', label: '呼吸道症状 (3): 咳嗽、咽痛、流涕', child_ids: ['咳嗽', '咽痛', '流涕'], count: 3, examples: ['咳嗽', '咽痛', '流涕'] }
          ],
          relation_details: {}
        }
      }
    }

    const projected = applyMedicalBrowseProjection(graph)

    expect(projected.nodes.map((node) => node.id)).toContain('collapse:resp')
    expect(projected.nodes.map((node) => node.id)).not.toContain('咳嗽')
    expect(projected.nodes.find((node) => node.id === 'collapse:resp')?.properties.entity_type).toBe('MedicalCollapsedGroup')
    expect(projected.edges.some((edge) => edge.source === '呼吸道症状' && edge.target === 'collapse:resp')).toBe(true)
  })
})
```

- [ ] **Step 2: Run test and confirm red**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\graph\medicalBrowseGraph.test.ts
```

Expected: fails because the module does not exist.

- [ ] **Step 3: Implement browse projection**

Create `medicalBrowseGraph.ts` with:

```ts
export const applyMedicalBrowseProjection = (graph: LightragGraphType): LightragGraphType => {
  const browse = graph.metadata?.medical_browse
  if (!browse?.collapsed_groups?.length) {
    return graph
  }

  const collapsedChildren = new Set(browse.collapsed_groups.flatMap((group) => group.child_ids))
  const nodes = graph.nodes.filter((node) => !collapsedChildren.has(node.id))
  const edges = graph.edges.filter(
    (edge) => !collapsedChildren.has(edge.source) && !collapsedChildren.has(edge.target)
  )

  for (const group of browse.collapsed_groups) {
    nodes.push({
      id: group.id,
      labels: ['MedicalCollapsedGroup'],
      properties: {
        entity_id: group.label,
        entity_type: 'MedicalCollapsedGroup',
        medical_group_parent: group.parent_id,
        child_ids: group.child_ids,
        count: group.count,
        examples: group.examples
      }
    })
    edges.push({
      id: `edge:${group.parent_id}:${group.id}`,
      source: group.parent_id,
      target: group.id,
      type: 'medicalCollapsedGroup',
      properties: { keywords: '包含', weight: 0.1 }
    })
  }

  return { ...graph, nodes, edges }
}
```

Also export `getMedicalBrowseNodeRole()` and `getMedicalBrowsePosition()` for deterministic layout.

- [ ] **Step 4: Apply projection in fetch path**

In `useLightragGraph.tsx`, before building `nodeIdMap`, do:

```ts
import { applyMedicalBrowseProjection, getMedicalBrowseNodeRole, getMedicalBrowsePosition } from '@/components/graph/medicalBrowseGraph'

const displayData = applyMedicalBrowseProjection(rawData)
rawData = displayData
```

When assigning initial positions, if `rawGraph.metadata?.medical_browse` exists, use `getMedicalBrowsePosition(node.id, rawGraph.metadata.medical_browse, index)` instead of random coordinates. Give stable sizes:

```ts
const role = getMedicalBrowseNodeRole(rawNode.id, rawGraph.metadata?.medical_browse)
if (role === 'root') rawNode.size = 22
if (role === 'category') rawNode.size = 16
if (role === 'subgroup') rawNode.size = 13
if (rawNode.properties.entity_type === 'MedicalCollapsedGroup') rawNode.size = 11
```

- [ ] **Step 5: Run focused WebUI tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\graph\medicalBrowseGraph.test.ts src\api\lightrag.test.ts src\stores\graph.test.ts
```

Expected: tests pass.

---

## Task 6: Clean Workspace Rebuild Docs

**Files:**
- Modify: `D:\LightRAG\docs\MedicalKGProfile-zh.md`
- Create: `D:\LightRAG\docs\MedicalKGCleanWorkspace-zh.md`
- Modify: `D:\LightRAG\tests\kg\test_medical_kg_quality_snapshot.py`

- [ ] **Step 1: Write docs test**

Add to `test_medical_kg_quality_snapshot.py`:

```python
def test_clean_workspace_documentation_states_delete_and_rebuild_policy():
    doc = Path("docs/MedicalKGCleanWorkspace-zh.md").read_text(encoding="utf-8")

    required = [
        "旧 workspace 可以删除",
        "新 workspace",
        "MEDICAL_KG_PROFILE=clinical_guideline_zh",
        "ENTITY_EXTRACTION_USE_JSON=true",
        "不要删除配置目录之外的路径",
        "重新导入原始医学文档",
    ]
    for fragment in required:
        assert fragment in doc
```

- [ ] **Step 2: Run test and confirm red**

Run:

```powershell
Set-Location D:\LightRAG
.\scripts\test.sh tests\kg\test_medical_kg_quality_snapshot.py
```

Expected: fails because the clean workspace doc does not exist.

- [ ] **Step 3: Write clean workspace guide**

Create `MedicalKGCleanWorkspace-zh.md` with:

```markdown
# 医学 KG 干净 Workspace 重建指南

本项目当前采用新 workspace 重建策略。旧 workspace 可以删除，因为当前旧数据不再作为验收依据。

## 推荐配置

```env
WORKSPACE=influenza_medical_v1
MEDICAL_KG_PROFILE=clinical_guideline_zh
ENTITY_EXTRACTION_USE_JSON=true
```

## 删除原则

只删除 LightRAG 配置中的 working directory 和 input directory 下对应 workspace 的数据。不要删除配置目录之外的路径。

## 重建步骤

1. 停止正在运行的文档处理。
2. 备份或删除旧 workspace 数据。
3. 使用新 workspace 名称启动服务。
4. 重新导入原始医学文档。
5. 打开 WebUI 图谱，搜索 `流行性感冒` 验证层级浏览视图。
```

- [ ] **Step 4: Update profile docs**

In `MedicalKGProfile-zh.md`, add a link to `MedicalKGCleanWorkspace-zh.md` and mention `medical_browse=true`.

- [ ] **Step 5: Run docs tests**

Run:

```powershell
Set-Location D:\LightRAG
.\scripts\test.sh tests\kg\test_medical_kg_quality_snapshot.py
```

Expected: tests pass.

---

## Task 7: Integrated Verification

**Files:**
- No source file ownership. This task runs verification and records results in `D:\LightRAG\progress.md`.

- [ ] **Step 1: Run backend tests**

Run:

```powershell
Set-Location D:\LightRAG
.\scripts\test.sh tests\extraction\test_medical_kg_hierarchy.py tests\api\routes\test_medical_graph_projection.py tests\kg\test_medical_kg_quality_snapshot.py
```

Expected: all selected backend tests pass.

- [ ] **Step 2: Run WebUI tests**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun test src\components\graph\medicalRelationGroups.test.ts src\components\graph\medicalBrowseGraph.test.ts src\api\lightrag.test.ts src\stores\graph.test.ts
```

Expected: all selected WebUI tests pass.

- [ ] **Step 3: Run WebUI lint**

Run:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun run lint
```

Expected: lint passes.

- [ ] **Step 4: Browser verification**

Start the WebUI dev server if it is not already running:

```powershell
Set-Location D:\LightRAG\lightrag_webui
bun run dev
```

Open the local URL shown by Vite. Search `流行性感冒` and verify:

- The graph has disease, category, subgroup, and collapsed group layers.
- Leaf symptoms are collapsed behind subgroup nodes by default.
- Right properties panel rows show `临床表现：高热不退` style labels when relation metadata exists.
- Tooltip/detail text shows the directed triple.

Record the result in `progress.md`.

