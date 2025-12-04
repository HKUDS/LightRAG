# Task Log - Fix KB Switching Refresh Issue

**Date:** 2025-11-23
**Task:** Fix issue where switching KB does not refresh data in Document Manager and Graph Viewer.

## Actions

- Analyzed `lightrag_webui/src/features/DocumentManager.tsx` and found missing dependency in `useEffect`.
- Analyzed `lightrag_webui/src/hooks/useLightragGraph.tsx` (used by `GraphViewer`) and found missing dependency in `useEffect`.
- Updated `DocumentManager.tsx` to include `selectedKB?.kb_id` in the data fetching dependency array.
- Updated `useLightragGraph.tsx` to include `selectedKB?.kb_id` in the data fetching dependency array and added logic to reset fetch state on KB change.
- Verified `ApiSite.tsx` does not need updates.
- Verified `RetrievalTesting.tsx` uses new context for queries automatically via API interceptor.

## Decisions

- Decided to trigger re-fetch automatically when `selectedKB` changes in the store.
- Decided not to clear chat history in `RetrievalTesting` automatically, as users might want to reference previous queries, and the new queries will naturally use the new context.

## Next Steps

- Verify the fix in the running application (user to perform).

## Lessons/Insights

- When using global state stores (like Zustand) with `useEffect` for data fetching, it's crucial to include the relevant state selectors in the dependency array, even if the API client handles the context injection internally. The component needs to "know" when to re-trigger the fetch.
