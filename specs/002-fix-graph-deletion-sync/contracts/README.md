# API Contracts: Fix Graph Deletion Sync

**Feature**: 002-fix-graph-deletion-sync
**Status**: Not Applicable

## Overview

This bug fix does not introduce any new API endpoints or modify existing API contracts.

## Existing Endpoints (Unchanged)

The following endpoints are used but not modified:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/documents/{doc_id}` | DELETE | Delete document (existing) |
| `/graphs` | GET | Get graph data (existing) |

## Internal Changes Only

All changes are internal implementation fixes:
- `adelete_by_doc_id()` - Reorder cache deletion
- `NetworkXStorage._get_graph()` - Add deletion flag check
- `NetworkXStorage` - Add `_deletion_in_progress` flag

No API contract changes required.
