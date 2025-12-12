import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'
import type { StoreApi, UseBoundStore } from 'zustand'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function randomColor() {
  const digits = '0123456789abcdef'
  let code = '#'
  for (let i = 0; i < 6; i++) {
    code += digits.charAt(Math.floor(Math.random() * 16))
  }
  return code
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Creates a throttled function that limits how often the original function can be called
 * @param fn The function to throttle
 * @param delay The delay in milliseconds
 * @returns A throttled version of the function
 */
export function throttle<T extends (...args: never[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastCall = 0
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  return function (this: unknown, ...args: Parameters<T>) {
    const now = Date.now()
    const remaining = delay - (now - lastCall)

    if (remaining <= 0) {
      // If enough time has passed, execute the function immediately
      if (timeoutId) {
        clearTimeout(timeoutId)
        timeoutId = null
      }
      lastCall = now
      fn.apply(this, args)
    } else if (!timeoutId) {
      // If not enough time has passed, set a timeout to execute after the remaining time
      timeoutId = setTimeout(() => {
        lastCall = Date.now()
        timeoutId = null
        fn.apply(this, args)
      }, remaining)
    }
  }
}

type WithSelectors<S> = S extends { getState: () => infer T }
  ? S & { use: { [K in keyof T]: () => T[K] } }
  : never

export const createSelectors = <S extends UseBoundStore<StoreApi<object>>>(_store: S) => {
  type State = ReturnType<S['getState']>
  const store = _store as WithSelectors<typeof _store>
  store.use = {} as { [K in keyof State]: () => State[K] }
  for (const k of Object.keys(store.getState())) {
    const key = k as keyof State & string
    ;(store.use as Record<string, () => unknown>)[key] = () =>
      store((s) => (s as Record<string, unknown>)[key])
  }

  return store
}
