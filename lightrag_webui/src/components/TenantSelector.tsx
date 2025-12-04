import { useEffect } from 'react'
import { useTenantState } from '@/stores/tenant'
import { useAuthStore } from '@/stores/state'
import { fetchTenantsPaginated, fetchKnowledgeBasesPaginated } from '@/api/tenant'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select'
import Button from '@/components/ui/Button'
import { PlusIcon, Building2, ArrowRightLeft, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react'

interface TenantSelectorProps {
  onTenantChange?: () => void
  onKBChange?: () => void
  hideTenantSelect?: boolean
  hideKBSelect?: boolean
}

export function TenantSelector({ onTenantChange, onKBChange, hideTenantSelect = false, hideKBSelect = false }: TenantSelectorProps) {
  const selectedTenant = useTenantState.use.selectedTenant()
  const selectedKB = useTenantState.use.selectedKB()
  const tenants = useTenantState.use.tenants()
  const knowledgeBases = useTenantState.use.knowledgeBases()
  const loading = useTenantState.use.loading()
  const error = useTenantState.use.error()
  const multiTenantEnabled = useAuthStore(state => state.multiTenantEnabled)

  const setSelectedTenant = useTenantState.use.setSelectedTenant()
  const setSelectedKB = useTenantState.use.setSelectedKB()
  const setTenants = useTenantState.use.setTenants()
  const setKnowledgeBases = useTenantState.use.setKnowledgeBases()
  const setLoading = useTenantState.use.setLoading()
  const setError = useTenantState.use.setError()
  const clearTenantSelection = useTenantState.use.clearTenantSelection()

  const tenantPageSize = 5
  const kbPageSize = 5

  useEffect(() => {
    // In single-tenant mode, skip tenant API calls
    if (!multiTenantEnabled) {
      console.log('[TenantSelector] Single-tenant mode, skipping tenant API calls')
      return
    }

    const loadTenants = async () => {
      setLoading(true)
      try {
        const data = await fetchTenantsPaginated(1, tenantPageSize)
        setTenants(data.items)
        // Auto-selection is handled by useTenantInitialization hook
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load tenants')
      } finally {
        setLoading(false)
      }
    }

    loadTenants()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [multiTenantEnabled])

  useEffect(() => {
    if (hideKBSelect) return

    // In single-tenant mode, skip KB API calls
    if (!multiTenantEnabled) {
      console.log('[TenantSelector] Single-tenant mode, skipping KB API calls')
      return
    }

    if (!selectedTenant) {
      setKnowledgeBases([])
      setSelectedKB(null)
      return
    }

    const loadKBs = async () => {
      setLoading(true)
      try {
        const data = await fetchKnowledgeBasesPaginated(selectedTenant.tenant_id, 1, kbPageSize)
        setKnowledgeBases(data.items)
        if (!selectedKB && data.items.length > 0) {
          setSelectedKB(data.items[0])
        } else if (selectedKB && !data.items.find(kb => kb.kb_id === selectedKB.kb_id)) {
          setSelectedKB(null)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load knowledge bases')
      } finally {
        setLoading(false)
      }
    }

    loadKBs()
    onTenantChange?.()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTenant?.tenant_id, hideKBSelect, multiTenantEnabled])

  useEffect(() => {
    onKBChange?.()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedKB?.kb_id])

  if (error && tenants.length === 0) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 bg-destructive/10 rounded-lg border border-destructive/20">
        <AlertCircle className="h-4 w-4 text-destructive" />
        <span className="text-sm text-destructive">{error}</span>
      </div>
    )
  }

  if (loading && tenants.length === 0) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg">
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Initializing tenants...</span>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-3 px-3 py-2 bg-muted/50 rounded-lg border border-border/50">
      {!hideTenantSelect ? (
        <div className="flex flex-col gap-1">
          <label className="text-xs font-semibold text-muted-foreground">Tenant</label>
          <div className="flex gap-2 items-center">
            <Select
              value={selectedTenant?.tenant_id || ''}
              onValueChange={(value) => {
                const tenant = tenants.find(t => t.tenant_id === value)
                if (tenant) setSelectedTenant(tenant)
              }}
              disabled={loading || tenants.length === 0}
            >
              <SelectTrigger className="h-8 text-xs w-40">
                <SelectValue placeholder="Select tenant..." />
              </SelectTrigger>
              <SelectContent>
                {tenants.map(tenant => (
                  <SelectItem key={tenant.tenant_id} value={tenant.tenant_id}>
                    <div className="flex items-center gap-2">
                      <Building2 className="h-3 w-3 text-muted-foreground" />
                      {tenant.name || tenant.tenant_name || tenant.tenant_id}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Button
              size="sm"
              variant="ghost"
              className="h-8 w-8 p-0 opacity-50 cursor-not-allowed"
              disabled
              title="Create new tenant (coming soon)"
            >
              <PlusIcon className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-1">
          <label className="text-xs font-semibold text-muted-foreground">Tenant</label>
          <div className="flex gap-2 items-center h-8">
            {loading ? (
              <Loader2 className="h-4 w-4 text-muted-foreground animate-spin" />
            ) : (
              <CheckCircle2 className="h-4 w-4 text-green-500" />
            )}
            <span className="text-sm font-medium truncate max-w-[120px]" title={selectedTenant?.name || selectedTenant?.tenant_name}>
              {selectedTenant?.name || selectedTenant?.tenant_name || 'No Tenant'}
            </span>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 w-6 ml-1 p-0 hover:bg-muted"
              onClick={() => clearTenantSelection()}
              title="Switch Tenant"
            >
              <ArrowRightLeft className="h-3 w-3" />
            </Button>
          </div>
        </div>
      )}

      {!hideKBSelect && selectedTenant && (
        <div className="w-px h-12 bg-border/50" />
      )}

      {!hideKBSelect && selectedTenant && (
        <div className="flex flex-col gap-1">
          <label className="text-xs font-semibold text-muted-foreground">Knowledge Base</label>
          <div className="flex gap-2 items-center">
            <Select
              value={selectedKB?.kb_id || ''}
              onValueChange={(value) => {
                const kb = knowledgeBases.find(k => k.kb_id === value)
                if (kb) setSelectedKB(kb)
              }}
              disabled={loading || knowledgeBases.length === 0}
            >
              <SelectTrigger className="h-8 text-xs w-40">
                <SelectValue placeholder="Select KB..." />
              </SelectTrigger>
              <SelectContent>
                {knowledgeBases.map(kb => (
                  <SelectItem key={kb.kb_id} value={kb.kb_id}>
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full bg-blue-500" />
                      {kb.name || kb.kb_name || kb.kb_id}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Button
              size="sm"
              variant="ghost"
              className="h-8 w-8 p-0 opacity-50 cursor-not-allowed"
              disabled
              title="Create new knowledge base (coming soon)"
            >
              <PlusIcon className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      {!hideKBSelect && selectedTenant && selectedKB && (
        <div className="flex items-center gap-2 ml-2 px-2 py-1 bg-background rounded border border-border/50">
          <span className="text-xs text-muted-foreground font-medium">
            {loading ? (
              <div className="flex items-center gap-1">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading...
              </div>
            ) : (
              <div className="flex items-center gap-1">
                <CheckCircle2 className="h-3 w-3 text-green-500" />
                {knowledgeBases.find(kb => kb.kb_id === selectedKB.kb_id)?.num_documents || 0} docs
              </div>
            )}
          </span>
        </div>
      )}
    </div>
  )
}

export default TenantSelector
