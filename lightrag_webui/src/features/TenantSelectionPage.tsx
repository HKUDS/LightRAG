import { useState, useEffect, useMemo } from 'react'
import { fetchTenantsPaginated } from '@/api/tenant'
import { Tenant, useTenantState } from '@/stores/tenant'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { Loader2, Search, Building2, Clock, Database, FileText, HardDrive } from 'lucide-react'

// Storage key for last selected tenant
const LAST_TENANT_KEY = 'lightrag:lastSelectedTenant'

interface TenantSelectionPageProps {
  onSelect: (tenant: Tenant) => void
}

export default function TenantSelectionPage({ onSelect }: TenantSelectionPageProps) {
  const [tenants, setTenants] = useState<Tenant[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [error, setError] = useState<string | null>(null)
  const setSelectedTenant = useTenantState.use.setSelectedTenant()
  
  // Track last selected tenant ID
  const [lastTenantId] = useState<string | null>(() => {
    try {
      return sessionStorage.getItem(LAST_TENANT_KEY)
    } catch {
      return null
    }
  })

  const loadTenants = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetchTenantsPaginated(page, 12, search)
      setTenants(response.items)
      setTotalPages(response.total_pages)
    } catch (err) {
      console.error('Failed to load tenants', err)
      setError('Failed to load tenants. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const timer = setTimeout(() => {
      loadTenants()
    }, 300)
    return () => clearTimeout(timer)
  }, [page, search])

  const handleSelect = (tenant: Tenant) => {
    // Set in store (which handles localStorage)
    setSelectedTenant(tenant)
    
    // Save as last selected tenant for next time
    try {
      sessionStorage.setItem(LAST_TENANT_KEY, tenant.tenant_id)
    } catch (e) {
      console.warn('Failed to save last tenant', e)
    }
    
    onSelect(tenant)
  }
  
  // Sort tenants to show last selected first
  const sortedTenants = useMemo(() => {
    if (!lastTenantId) return tenants
    
    return [...tenants].sort((a, b) => {
      if (a.tenant_id === lastTenantId) return -1
      if (b.tenant_id === lastTenantId) return 1
      return 0
    })
  }, [tenants, lastTenantId])

  // Check if a tenant was recently selected
  const isLastSelected = (tenantId: string) => tenantId === lastTenantId

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-4xl space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Welcome to LightRAG</h1>
          <p className="text-muted-foreground text-lg">Select a tenant to continue to the dashboard</p>
        </div>

        <div className="flex items-center space-x-2 max-w-md mx-auto">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search tenants..."
              className="pl-9"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </div>

        {error && (
          <div className="text-center text-destructive bg-destructive/10 p-4 rounded-md">
            {error}
            <Button variant="link" onClick={loadTenants} className="ml-2">Retry</Button>
          </div>
        )}

        {loading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sortedTenants.map((tenant) => (
              <Card 
                key={tenant.tenant_id} 
                className={`cursor-pointer hover:border-primary hover:shadow-lg transition-all duration-200 ${
                  isLastSelected(tenant.tenant_id) ? 'border-primary/50 bg-primary/5 ring-2 ring-primary/20' : ''
                }`}
                onClick={() => handleSelect(tenant)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <Building2 className="h-5 w-5 text-primary flex-shrink-0" />
                        <CardTitle className="text-lg font-semibold leading-tight">
                          {tenant.name || tenant.tenant_name || tenant.tenant_id}
                        </CardTitle>
                      </div>
                      {isLastSelected(tenant.tenant_id) && (
                        <span className="inline-flex items-center gap-1 text-xs text-primary bg-primary/10 px-2 py-0.5 rounded-full mb-2">
                          <Clock className="h-3 w-3" />
                          Recently used
                        </span>
                      )}
                    </div>
                  </div>
                  <CardDescription className="text-sm leading-relaxed mt-1">
                    {tenant.description || 'No description available'}
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="grid grid-cols-3 gap-3 pt-3 border-t border-border/50">
                    <div className="flex flex-col items-center p-2 rounded-lg bg-muted/50">
                      <Database className="h-4 w-4 text-blue-500 mb-1" />
                      <span className="text-lg font-bold">{tenant.num_knowledge_bases || 0}</span>
                      <span className="text-xs text-muted-foreground">KBs</span>
                    </div>
                    <div className="flex flex-col items-center p-2 rounded-lg bg-muted/50">
                      <FileText className="h-4 w-4 text-green-500 mb-1" />
                      <span className="text-lg font-bold">{tenant.num_documents || 0}</span>
                      <span className="text-xs text-muted-foreground">Docs</span>
                    </div>
                    <div className="flex flex-col items-center p-2 rounded-lg bg-muted/50">
                      <HardDrive className="h-4 w-4 text-orange-500 mb-1" />
                      <span className="text-lg font-bold">
                        {tenant.storage_used_gb ? `${tenant.storage_used_gb.toFixed(1)}` : '0'}
                      </span>
                      <span className="text-xs text-muted-foreground">GB</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
            
            {tenants.length === 0 && !loading && (
              <div className="col-span-full text-center py-12 text-muted-foreground">
                No tenants found matching your search.
              </div>
            )}
          </div>
        )}

        {totalPages > 1 && (
          <div className="flex justify-center space-x-2 mt-8">
            <Button 
              variant="outline" 
              disabled={page <= 1} 
              onClick={() => setPage(p => p - 1)}
            >
              Previous
            </Button>
            <div className="flex items-center px-4 text-sm">
              Page {page} of {totalPages}
            </div>
            <Button 
              variant="outline" 
              disabled={page >= totalPages} 
              onClick={() => setPage(p => p + 1)}
            >
              Next
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
