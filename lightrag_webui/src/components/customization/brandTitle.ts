import { getRuntimeWebuiTitle } from '@/lib/runtimeConfig'

const normalizedTitle = (title: string | null | undefined): string | null => {
  const trimmed = title?.trim()
  return trimmed ? trimmed : null
}

/**
 * Resolves the deployment title without consulting persisted auth state.
 *
 * A successful customization response is authoritative. If that optional
 * endpoint is unavailable, the current /auth-status response is the next
 * freshest source; the title injected into index.html covers deployments
 * where a reverse proxy cannot reach either API route. Local-storage auth
 * state is intentionally excluded because it may belong to an older server
 * configuration.
 */
export function resolveBrandTitle(
  snapshotTitle: string | null | undefined,
  authStatusTitle: string | null | undefined
): string {
  return (
    normalizedTitle(snapshotTitle) ??
    normalizedTitle(authStatusTitle) ??
    normalizedTitle(getRuntimeWebuiTitle()) ??
    'LightRAG'
  )
}
