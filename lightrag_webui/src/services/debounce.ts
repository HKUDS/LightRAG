/**
 * Debounce utility function
 *
 * Creates a debounced version of a function that delays invoking
 * the function until after `wait` milliseconds have elapsed since
 * the last time the debounced function was invoked.
 */

export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  return function(this: any, ...args: Parameters<T>) {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }

    timeoutId = setTimeout(() => {
      fn.apply(this, args)
      timeoutId = null
    }, wait)
  }
}

/**
 * Debounce with immediate option
 *
 * If `immediate` is true, the function is invoked on the leading edge
 * instead of the trailing edge of the wait interval.
 */
export function debounceImmediate<T extends (...args: any[]) => any>(
  fn: T,
  wait: number,
  immediate: boolean = false
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  return function(this: any, ...args: Parameters<T>) {
    const callNow = immediate && !timeoutId

    if (timeoutId) {
      clearTimeout(timeoutId)
    }

    timeoutId = setTimeout(() => {
      timeoutId = null
      if (!immediate) {
        fn.apply(this, args)
      }
    }, wait)

    if (callNow) {
      fn.apply(this, args)
    }
  }
}

export default debounce
