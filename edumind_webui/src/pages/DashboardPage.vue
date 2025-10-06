<template>
  <div class="dashboard-page">
    <v-container fluid class="dashboard-page__content">
      <div class="dashboard-page__intro">
        <div>
          <h2 class="text-subtitle-1 font-weight-semibold mb-1">Curate your knowledge spaces</h2>
          <p class="text-body-2 text-medium-emphasis mb-0">
            Review the workspaces generated for user 101 and continue your creation flow.
          </p>
        </div>
      </div>

      <div v-if="loading" class="dashboard-page__state">
        <v-progress-circular indeterminate color="primary" size="40" />
        <p class="text-body-2 text-medium-emphasis mt-3 mb-0">Loading workspaces…</p>
      </div>

      <div v-else-if="error" class="dashboard-page__state">
        <v-icon size="36" color="primary">mdi-alert-circle-outline</v-icon>
        <p class="text-body-2 text-medium-emphasis mt-3 mb-2">Unable to load workspaces.</p>
        <v-btn color="primary" variant="flat" @click="refresh">Try Again</v-btn>
      </div>

      <v-row v-else class="dashboard-page__grid" align="stretch" justify="start">
        <v-col
          v-for="workspace in workspaces"
          :key="workspace.id"
          cols="12"
          sm="6"
          md="4"
          class="dashboard-page__grid-item"
        >
          <v-card class="dashboard-page__card" elevation="2">
            <v-card-item class="pb-2">
              <div class="dashboard-page__card-header">
                <v-avatar color="primary" size="40" variant="flat">
                  <v-icon size="20" color="white">mdi-folder-outline</v-icon>
                </v-avatar>
                <div class="dashboard-page__card-meta">
                  <div class="dashboard-page__card-meta-top">
                    <h3 class="text-subtitle-1 font-weight-semibold mb-1">{{ workspace.name }}</h3>
                    <v-menu location="bottom end">
                      <template #activator="{ props: menuProps }">
                        <v-btn
                          icon
                          variant="text"
                          color="primary"
                          v-bind="menuProps"
                        >
                          <v-icon>mdi-dots-vertical</v-icon>
                        </v-btn>
                      </template>
                      <v-list density="comfortable">
                        <v-list-item @click="openEditWorkspace(workspace)">
                          <v-list-item-title>Edit workspace</v-list-item-title>
                        </v-list-item>
                        <v-list-item @click="openDeleteWorkspace(workspace)">
                          <v-list-item-title>Delete workspace</v-list-item-title>
                        </v-list-item>
                      </v-list>
                    </v-menu>
                  </div>
                  <p class="text-caption text-medium-emphasis mb-0">
                    Created {{ formatDate(workspace.createdAt) }}
                  </p>
                </div>
              </div>
            </v-card-item>
            <v-card-text class="dashboard-page__card-text">
              <p class="text-body-2 text-medium-emphasis mb-0">
                {{ workspace.instructions }}
              </p>
            </v-card-text>
            <v-card-actions class="pt-0">
              <v-btn
                variant="text"
                color="primary"
                prepend-icon="mdi-launch"
              :to="{ name: 'Studio', query: { workspaceId: workspace.id } }"
              >
                Open Workspace
              </v-btn>
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>

      <div v-if="!loading && !hasWorkspaces && !error" class="dashboard-page__state">
        <v-icon size="36" color="primary">mdi-folder-remove-outline</v-icon>
        <p class="text-body-2 text-medium-emphasis mt-3 mb-0">
          No workspaces have been created for this user yet.
        </p>
      </div>
    </v-container>

    <v-dialog v-model="workspaceDialogOpen" max-width="480">
      <v-card>
        <v-card-title class="text-h6 font-weight-semibold">{{ dialogTitle }}</v-card-title>
        <v-card-text class="pt-4">
          <v-text-field
            v-model="newWorkspaceName"
            label="Workspace name"
            variant="outlined"
            density="comfortable"
            :disabled="savingWorkspace"
            required
          />
          <v-textarea
            v-model="newWorkspaceInstructions"
            label="Instructions (optional)"
            variant="outlined"
            density="comfortable"
            auto-grow
            rows="3"
            :disabled="savingWorkspace"
          />
          <v-alert
            v-if="dialogError"
            type="error"
            variant="tonal"
            density="comfortable"
            class="mt-3"
          >
            {{ dialogError }}
          </v-alert>
        </v-card-text>
        <v-card-actions class="justify-end">
          <v-btn
            variant="text"
            color="primary"
            @click="closeWorkspaceDialog"
            :disabled="savingWorkspace"
          >
            Cancel
          </v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :loading="savingWorkspace"
            @click="handleSaveWorkspace"
          >
            {{ isEditMode ? 'Save Changes' : 'Create' }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-dialog v-model="deleteDialogOpen" max-width="420">
      <v-card>
        <v-card-title class="text-h6 font-weight-semibold">Delete Workspace</v-card-title>
        <v-card-text>
          <p class="text-body-2 mb-0">
            Are you sure you want to delete
            <strong>{{ deleteTarget?.name || 'this workspace' }}</strong>? This action cannot be undone.
          </p>
          <v-alert
            v-if="dialogError"
            type="error"
            variant="tonal"
            density="comfortable"
            class="mt-3"
          >
            {{ dialogError }}
          </v-alert>
        </v-card-text>
        <v-card-actions class="justify-end">
          <v-btn
            variant="text"
            color="primary"
            @click="closeDeleteDialog"
            :disabled="deletingWorkspace"
          >
            Cancel
          </v-btn>
          <v-btn
            color="error"
            variant="flat"
            :loading="deletingWorkspace"
            @click="handleDeleteWorkspace"
          >
            Delete
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, onActivated, onDeactivated, watch } from 'vue'
import { useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useDashboardStore, useHeaderStore } from '@/stores'

const dashboardStore = useDashboardStore()
const router = useRouter()
const headerStore = useHeaderStore()
const { workspaces, loading, error, hasWorkspaces } = storeToRefs(dashboardStore)

const workspaceDialogOpen = ref(false)
const dialogMode = ref<'create' | 'edit'>('create')
const targetWorkspaceId = ref('')
const savingWorkspace = ref(false)
const newWorkspaceName = ref('')
const newWorkspaceInstructions = ref('')
const dialogError = ref('')

const deleteDialogOpen = ref(false)
const deletingWorkspace = ref(false)
const deleteTarget = ref<{ id: string; name: string } | null>(null)

const isEditMode = computed(() => dialogMode.value === 'edit')
const dialogTitle = computed(() => (isEditMode.value ? 'Edit Workspace' : 'Create Workspace'))

const refresh = () => {
  dashboardStore.initialise({ force: true })
}

const formatDate = (value?: string) => {
  if (!value) return 'recently'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'recently'
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

const resetWorkspaceForm = () => {
  newWorkspaceName.value = ''
  newWorkspaceInstructions.value = ''
  dialogError.value = ''
  targetWorkspaceId.value = ''
  dialogMode.value = 'create'
}

const openCreateWorkspace = () => {
  resetWorkspaceForm()
  dialogMode.value = 'create'
  workspaceDialogOpen.value = true
}

const openEditWorkspace = (workspace: { id: string; name: string; instructions: string }) => {
  resetWorkspaceForm()
  dialogMode.value = 'edit'
  targetWorkspaceId.value = workspace.id
  newWorkspaceName.value = workspace.name
  newWorkspaceInstructions.value = workspace.instructions === 'No instructions provided.' ? '' : workspace.instructions
  workspaceDialogOpen.value = true
}

const closeWorkspaceDialog = () => {
  workspaceDialogOpen.value = false
  resetWorkspaceForm()
}

const handleSaveWorkspace = async () => {
  if (!newWorkspaceName.value.trim()) {
    dialogError.value = 'Workspace name is required.'
    return
  }

  savingWorkspace.value = true
  dialogError.value = ''

  try {
    let workspace
    if (isEditMode.value) {
      workspace = await dashboardStore.updateWorkspace({
        id: targetWorkspaceId.value,
        name: newWorkspaceName.value,
        instructions: newWorkspaceInstructions.value,
      })
    } else {
      workspace = await dashboardStore.createWorkspace({
        name: newWorkspaceName.value,
        instructions: newWorkspaceInstructions.value,
      })
    }

    closeWorkspaceDialog()

    if (workspace?.id) {
      router.push({ name: 'Studio', query: { workspaceId: workspace.id } })
    }
  } catch (error) {
    dialogError.value = error?.message || `Failed to ${isEditMode.value ? 'update' : 'create'} workspace.`
  } finally {
    savingWorkspace.value = false
  }
}

const openDeleteWorkspace = (workspace: { id: string; name: string }) => {
  deleteTarget.value = { id: workspace.id, name: workspace.name }
  deleteDialogOpen.value = true
  dialogError.value = ''
}

const closeDeleteDialog = () => {
  deleteDialogOpen.value = false
  deleteTarget.value = null
  dialogError.value = ''
}

const handleDeleteWorkspace = async () => {
  if (!deleteTarget.value?.id) {
    return
  }

  deletingWorkspace.value = true
  dialogError.value = ''

  try {
    await dashboardStore.deleteWorkspace({ id: deleteTarget.value.id })
    closeDeleteDialog()
  } catch (error) {
    dialogError.value = error?.message || 'Failed to delete workspace.'
  } finally {
    deletingWorkspace.value = false
  }
}

const applyHeader = () => {
  headerStore.setHeader({
    title: 'Workspace Dashboard',
    description: 'Curate your knowledge spaces and continue building content.',
    actions: [
      {
        id: 'create-workspace',
        label: 'Create Workspace',
        icon: 'mdi-plus-circle-outline',
        variant: 'flat',
        onClick: openCreateWorkspace,
        disabled: loading.value || savingWorkspace.value,
      },
      {
        id: 'refresh-workspaces',
        label: 'Refresh',
        icon: 'mdi-refresh',
        variant: 'text',
        onClick: refresh,
        loading: loading.value,
      },
      {
        id: 'go-questions',
        label: 'Go to Questions Library',
        icon: 'mdi-file-document-multiple-outline',
        variant: 'text',
        to: { name: 'Questions' },
      },
    ],
  })
}

onMounted(() => {
  dashboardStore.initialise()
  // Make sure header is applied on first mount
  applyHeader()
})

// If this view is cached via <keep-alive>, ensure header reapplies when we return
onActivated(() => {
  applyHeader()
})

// Clean up header when leaving this view
onDeactivated(() => {
  headerStore.resetHeader()
})

onUnmounted(() => {
  headerStore.resetHeader()
})

// Keep header reactive to state changes, but don't rely on it for initial/return cases
watch([loading, savingWorkspace, deletingWorkspace], () => {
  applyHeader()
})

</script>

<style scoped>
.dashboard-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #ffffff;
  overflow: hidden;
}

.dashboard-page__content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 32px 40px;
  gap: 32px;
  overflow-y: auto;
}

.dashboard-page__intro {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.dashboard-page__state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 12px;
  padding: 48px 0;
  color: rgba(17, 24, 39, 0.72);
}

.dashboard-page__grid {
  margin: 0;
  gap: 16px;
}

.dashboard-page__grid-item {
  display: flex;
  min-width: 0;
}

.dashboard-page__card {
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 260px;
}

.dashboard-page__card-header {
  display: flex;
  align-items: flex-start;
  gap: 16px;
}

.dashboard-page__card-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
}

.dashboard-page__card-meta-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.dashboard-page__card-text {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.dashboard-page__card-text p {
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}

@media (max-width: 960px) {
  .dashboard-page__content {
    padding: 24px 16px;
  }
}
</style>
