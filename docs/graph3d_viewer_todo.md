# 3D Graph Viewer — Feature To-Do

Planned enhancements for the 3D knowledge graph viewer
(`lightrag/api/static/graph3d_viewer.html`, backend in
`lightrag/api/routers/graph3d_routes.py` + `query_routes.py`).

## 1. Brightness adjustment bar ✅

Add a slider control (footer bar or settings panel) that adjusts the
brightness/intensity of all graph nodes.

- Range: e.g. 0.2× – 2.0× of current brightness
- Apply via point material color multiplier / bloom strength
- Persist the setting in `localStorage` so it survives reloads

**Implemented**: footer slider next to the GK button; scales cluster base
colors via `getBaseColor()` and `UnrealBloomPass.strength`; persisted in
`localStorage` (`graph3d_brightness`).

## 2. Daily query cache with idle replay ✅

Cache all queries made during the day. When no new query has been queued
for 30 seconds, randomly replay activations from the cached queries.

- Backend: extend the activation recording in `query_routes.py` with a
  per-day cache (reset at midnight)
- Idle detection: 30s without new queued queries triggers replay mode
- Replay picks random cached queries and re-runs their activations
- **Replayed activations render in a dimmer color** to distinguish them
  from live queries (e.g. 40–50% opacity of the normal activation color)
- Replay stops immediately when a real query arrives
- Show a subtle "replay" indicator in the Query Log

**Implemented**: `_daily_query_cache` + `GET /query/activations/daily_replay`
in `query_routes.py`; frontend `checkIdleReplay()` polls every 5s after 30s
idle, replays with `REPLAY_DIM_FACTOR = 0.45`, logs `🔁 Replay` entries, and
re-checks idle state mid-fetch so a real query always wins.

## 3. Light-up effect for clicked node ✅

When a node is clicked, give it a distinct light-up/pulse effect.

- Bright highlight + glow on the clicked node itself
- Brief pulse animation (scale/brightness) on click
- Effect clears when another node is clicked or the details panel closes

**Implemented**: persistent-highlight layer (survives `pulse()`'s per-frame
reset); clicked node turns white with a 2.5× → 1.4× pulse-then-settle over
1.5s; cleared on next click or panel close.

## 4. Hover-to-highlight cluster from entity panel ✅

When the cursor hovers over the cluster text/label in the entity
information panel, light up all nodes belonging to that cluster (cluster N)
in the 3D graph.

- Hover on cluster name in Entity Details panel → highlight whole cluster
- Mouse leave → restore previous state
- Should compose cleanly with any active query activations

**Implemented**: `mouseenter`/`mouseleave` on `#entity-cluster` highlights
all cluster members (brightened cluster color, 1.8× size) via source-tagged
highlights that compose with live activations; `mouseleave` clears only the
hover-sourced entries.

## 5. Clickable connection nodes with relationship view ✅

Make the connected nodes listed in the entity panel clickable.

- Clicking a connected node shows the **relationships** between the two
  entities in the entity panel
- Simultaneously **light up the involved nodes and edges** in the 3D graph
- The relationship view lists nodes and edges with their information
- Items inside the relationship view are themselves clickable → navigate to
  that entity's information panel (drill-down navigation)
- Consider a back-history stack so users can retrace their navigation path

**Implemented**: `GET /graph3d/entity_edges` (weight-sorted neighbors) +
`GET /graph3d/edge_info` (description/keywords/weight, both lookup orders);
connection chips under the entity meta; relationship view with clickable
entity links; both nodes + connecting edge light up white in the 3D graph;
`← Back` button walks the navigation history.

## 6. Settings sidebar (collapsed footer) ✅

Collapse all footer navigation buttons into a single gear (⚙) button in the
bottom-right corner. Clicking opens a sliding sidebar containing all
controls (language, brightness, view toggles, panel toggles).

**Implemented**: deleted `.bottom-bar`; added `.settings-gear` fixed button +
`.settings-panel` sliding sidebar (mirrors `.side-panel` slide CSS); all 8
controls moved inside with full-width stacked layout.

## 7. Brightness slider ceiling ×2 ✅

Double the maximum brightness from 2.0× to 4.0×.

**Implemented**: slider `max` changed from `2.0` to `4.0`. Linear scale
unchanged.

## 8. Chinese i18n with language toggle ✅

Translate the entire viewer UI into Chinese. Provide an external
`graph3d_i18n.json` (en/zh mapping) served via `GET /graph3d/i18n.json`.
Add a language toggle in the settings sidebar. Default language: Chinese.

**Implemented**: `lightrag/api/static/graph3d_i18n.json` (78 keys, en/zh
parity enforced); `t()` runtime with embedded English fallback;
`data-i18n` / `data-i18n-title` / `data-i18n-alt` attribute wiring on all
static text + dynamic `t()` calls in `addLog()`, `displayInsights()`,
`showClusterInfo()`, `formatEntityList()`, `showRelationship()`;
`setLanguage()` persists to `localStorage` (`graph3d_lang`).

## 9. Title translation ✅

Translate "GK Brain Knowledge Graph" → "GK大脑类神经元3D视图" (default zh).

**Implemented**: `app.title` key in i18n JSON; `data-i18n` on title `<span>`
and `<title>` element.

## 10. Remove query input ✅

Remove the manual query input box and its submit path (the external query
polling, Query Log, and idle replay remain — only the input UI goes).

**Implemented**: deleted `#query-panel` markup, `.query-panel` CSS,
`#query-submit` / `#query-input` event listeners, and the
`queryAndActivate()` function. Removed dead `isManual` branch from
`processActivationQueue()`.

## 11. Title QR badge ✅

Pin a small (36×36) always-visible GK QR code next to the title in the
top-left.

**Implemented**: `<img class="title-qr" src="…gkcharger_qrcode.jpg">` inside
`.top-bar .logo` after the title span.

## 12. Light/dark theme toggle ✅

Add a theme toggle (🌙 Dark / ☀️ Light) to the settings sidebar, next to the
language toggle. Persisted in `localStorage`, applied before first paint.

**Implemented**: `[data-theme="light"]` CSS token overrides (bg, surface,
border, text, accent, top-bar fade) on top of the existing `:root` tokens;
`setTheme()` / `applyTheme()` re-skin the Three.js scene live — scene
background, point/edge materials switch from `AdditiveBlending` (dark) to
`NormalBlending` with higher opacity (light), bloom pass disabled in light
mode (reads as fog on a light background), and pulse/highlight colors lerp
toward dark navy instead of white so activations stay visible. Choice
persisted in `localStorage` (`graph3d_theme`); new i18n keys
`settings.theme` / `theme.dark` / `theme.light` (en/zh).

## 13. Title & QR visibility toggle ✅

Add a settings-sidebar toggle to show/hide the top-bar title and the GK QR
badge (`.top-bar .logo`). Persisted in `localStorage`.

**Implemented**: `btn-toggle-title` in the View section; hides/shows the
`.logo` block (title + QR); button dims to 50% opacity while hidden, same
pattern as the other panel toggles; state persisted in `localStorage`
(`graph3d_show_title`, default visible); new i18n key `btn.toggleTitle`
(en/zh).
