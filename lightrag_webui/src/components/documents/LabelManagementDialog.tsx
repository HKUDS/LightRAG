import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/Dialog'
import Input from '@/components/ui/Input'
import Badge from '@/components/ui/Badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/Card'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import {
  getAllLabels,
  createLabel,
  deleteLabel,
  getLabelStatistics,
  LabelType,
  LabelCreateType,
  LabelStatisticsType
} from '@/api/lightrag'
import { TagIcon, PlusIcon, TrashIcon, BarChart3Icon } from 'lucide-react'

interface LabelManagementDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function LabelManagementDialog({ open, onOpenChange }: LabelManagementDialogProps) {
  const { t } = useTranslation()
  const [labels, setLabels] = useState<Record<string, LabelType>>({})
  const [statistics, setStatistics] = useState<LabelStatisticsType | null>(null)
  const [loading, setLoading] = useState(false)
  const [creatingLabel, setCreatingLabel] = useState(false)
  const [showCreateForm, setShowCreateForm] = useState(false)
  
  // Form state for creating labels
  const [newLabel, setNewLabel] = useState<LabelCreateType>({
    name: '',
    description: '',
    color: '#0066cc'
  })

  const loadLabels = async () => {
    try {
      setLoading(true)
      const [labelsData, statsData] = await Promise.all([
        getAllLabels(),
        getLabelStatistics()
      ])
      setLabels(labelsData)
      setStatistics(statsData)
    } catch (error) {
      console.error('Error loading labels:', error)
      toast.error(`Failed to load labels: ${errorMessage(error)}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open) {
      loadLabels()
    }
  }, [open])

  const handleCreateLabel = async () => {
    if (!newLabel.name.trim()) {
      toast.error('Label name is required')
      return
    }

    try {
      setCreatingLabel(true)
      await createLabel(newLabel)
      toast.success('Label created successfully')
      setNewLabel({ name: '', description: '', color: '#0066cc' })
      setShowCreateForm(false)
      await loadLabels()
    } catch (error) {
      console.error('Error creating label:', error)
      toast.error(`Failed to create label: ${errorMessage(error)}`)
    } finally {
      setCreatingLabel(false)
    }
  }

  const handleDeleteLabel = async (labelName: string) => {
    if (!confirm(`Are you sure you want to delete the label "${labelName}"?`)) {
      return
    }

    try {
      await deleteLabel(labelName)
      toast.success('Label deleted successfully')
      await loadLabels()
    } catch (error) {
      console.error('Error deleting label:', error)
      toast.error(`Failed to delete label: ${errorMessage(error)}`)
    }
  }

  const labelList = Object.values(labels)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TagIcon className="h-5 w-5" />
            Label Management
          </DialogTitle>
          <DialogDescription>
            Manage document labels and view usage statistics
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4 overflow-hidden">
          {/* Statistics Card */}
          {statistics && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <BarChart3Icon className="h-5 w-5" />
                  Statistics
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-blue-600">
                      {statistics.total_labels}
                    </div>
                    <div className="text-sm text-muted-foreground">Total Labels</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-green-600">
                      {statistics.total_labeled_documents}
                    </div>
                    <div className="text-sm text-muted-foreground">Labeled Documents</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-purple-600">
                      {Object.keys(statistics.labels_with_counts).length}
                    </div>
                    <div className="text-sm text-muted-foreground">Active Labels</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Create Label Section */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center justify-between">
                <span>Create New Label</span>
                <Button
                  size="sm"
                  onClick={() => setShowCreateForm(!showCreateForm)}
                  variant={showCreateForm ? "secondary" : "default"}
                >
                  <PlusIcon className="h-4 w-4 mr-1" />
                  {showCreateForm ? 'Cancel' : 'Add Label'}
                </Button>
              </CardTitle>
            </CardHeader>
            
            {showCreateForm && (
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label htmlFor="label-name" className="block text-sm font-medium mb-1">Name</label>
                    <Input
                      id="label-name"
                      value={newLabel.name}
                      onChange={(e) => setNewLabel(prev => ({ ...prev, name: e.target.value }))}
                      placeholder="Label name"
                      maxLength={100}
                    />
                  </div>
                  <div>
                    <label htmlFor="label-color" className="block text-sm font-medium mb-1">Color</label>
                    <Input
                      id="label-color"
                      type="color"
                      value={newLabel.color}
                      onChange={(e) => setNewLabel(prev => ({ ...prev, color: e.target.value }))}
                      className="h-10"
                    />
                  </div>
                  <div className="md:col-span-1 flex items-end">
                    <Button
                      onClick={handleCreateLabel}
                      disabled={creatingLabel || !newLabel.name.trim()}
                      className="w-full"
                    >
                      {creatingLabel ? 'Creating...' : 'Create Label'}
                    </Button>
                  </div>
                </div>
                <div className="mt-4">
                  <label htmlFor="label-description" className="block text-sm font-medium mb-1">Description</label>
                  <textarea
                    id="label-description"
                    value={newLabel.description}
                    onChange={(e) => setNewLabel(prev => ({ ...prev, description: e.target.value }))}
                    placeholder="Label description (optional)"
                    maxLength={500}
                    rows={2}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </CardContent>
            )}
          </Card>

          {/* Labels Table */}
          <Card className="flex-1 overflow-hidden">
            <CardHeader className="pb-3">
              <CardTitle>Existing Labels ({labelList.length})</CardTitle>
            </CardHeader>
            <CardContent className="p-0 overflow-auto">
              {loading ? (
                <div className="p-4 text-center text-muted-foreground">
                  Loading labels...
                </div>
              ) : labelList.length === 0 ? (
                <div className="p-8 text-center text-muted-foreground">
                  <TagIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No labels found</p>
                  <p className="text-sm">Create your first label to get started</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Label</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead className="text-center">Documents</TableHead>
                      <TableHead className="text-center">Created</TableHead>
                      <TableHead className="text-center">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {labelList.map((label) => (
                      <TableRow key={label.name}>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div
                              className="w-4 h-4 rounded-full border"
                              style={{ backgroundColor: label.color }}
                            />
                            <Badge variant="secondary">{label.name}</Badge>
                          </div>
                        </TableCell>
                        <TableCell className="max-w-xs">
                          <div className="truncate" title={label.description}>
                            {label.description || 'No description'}
                          </div>
                        </TableCell>
                        <TableCell className="text-center">
                          {label.document_count}
                        </TableCell>
                        <TableCell className="text-center">
                          {new Date(label.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-center">
                          <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => handleDeleteLabel(label.name)}
                            disabled={label.document_count > 0}
                            title={label.document_count > 0 ? 'Cannot delete label with assigned documents' : 'Delete label'}
                          >
                            <TrashIcon className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </div>

        <DialogFooter>
          <Button variant="secondary" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}