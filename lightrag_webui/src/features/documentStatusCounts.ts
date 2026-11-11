/**
 * Helpers over the backend's full-corpus status-count map
 * (`PaginatedDocsResponse.status_counts`, produced by `get_all_status_counts`).
 */

/**
 * Stable digest of a status-count map, used to detect that the corpus changed.
 *
 * Keys are lower-cased before sorting so that a backend reporting `PROCESSED`
 * rather than `processed` does not read as a count change: the key case varies
 * by storage backend, which is also why `getCountValue` accepts both spellings.
 *
 * The map covers every DocStatus the backend knows about, `pending` and the
 * deprecated `preprocessed` included. The per-page bucket structure this
 * replaced could not: its keys were filter buckets, which have no entry for
 * either status, so a page of only pending documents produced a constant
 * digest and never signalled a change.
 */
export const computeStatusCountsFingerprint = (
  counts: Record<string, number>
): string =>
  Object.entries(counts)
    .map(([key, value]) => `${key.toLowerCase()}:${value ?? 0}`)
    .sort()
    .join('|')
