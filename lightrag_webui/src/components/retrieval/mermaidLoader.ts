/**
 * On-demand loader for the Mermaid renderer.
 *
 * Both query entries share `ChatMessage`, so mermaid MUST NOT be a static
 * import — it would land in the workspace entry's first-load closure. It is
 * fetched when a complete ```mermaid``` block is actually rendered, which
 * keeps full Mermaid support in both entries at no first-load cost.
 *
 * It lives in its own module so the load can be STUBBED without touching the
 * `mermaid` package itself. Bun's module-mock registry is process-wide and a
 * module mock cannot be undone, so mocking `mermaid` directly hands every
 * later test file whatever stub the last mocker left behind — and registering
 * a rejecting mock only works while nothing in the process has loaded mermaid
 * yet, which no test file can guarantee about the ones before it. Stubbing
 * this one-function module instead is order-independent and exactly
 * restorable.
 */
export const loadMermaid = () => import('mermaid').then((m) => m.default)
