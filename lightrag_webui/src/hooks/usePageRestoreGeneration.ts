import { useEffect, useState } from 'react'

/**
 * Subscribe to pages restored from the browser's back/forward cache.
 *
 * A BFCache restore resumes the existing JavaScript heap instead of mounting a
 * new React tree, so mount effects do not run again. Consumers use the returned
 * generation to restart work whose cadence or data may be stale after the page
 * was frozen.
 */
export const subscribeToPageRestore = (
  onRestore: () => void,
  target: EventTarget = window
): (() => void) => {
  const handlePageShow = (event: Event) => {
    if ((event as PageTransitionEvent).persisted) {
      onRestore()
    }
  }

  target.addEventListener('pageshow', handlePageShow)
  return () => target.removeEventListener('pageshow', handlePageShow)
}

/** Incremented whenever the current document is restored from BFCache. */
export default function usePageRestoreGeneration(): number {
  const [generation, setGeneration] = useState(0)

  useEffect(
    () => subscribeToPageRestore(() => setGeneration((current) => current + 1)),
    []
  )

  return generation
}
