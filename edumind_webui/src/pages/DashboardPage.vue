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
        <div class="dashboard-page__intro-actions">
          <v-btn
            variant="outlined"
            color="primary"
            prepend-icon="mdi-plus-circle-outline"
            :disabled="loading"
            @click="openCreateWorkspace"
          >
            Create Workspace
          </v-btn>
          <v-btn
            variant="text"
            color="primary"
            prepend-icon="mdi-refresh"
            :loading="loading"
            @click="refresh"
          >
            Refresh
          </v-btn>
          <v-btn
            variant="text"
            color="primary"
            prepend-icon="mdi-file-document-multiple-outline"
            :to="{ name: 'Questions' }"
          >
            Go to Questions Library
          </v-btn>
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
                  <h3 class="text-subtitle-1 font-weight-semibold mb-1">{{ workspace.name }}</h3>
                  <p class="text-caption text-medium-emphasis mb-0">
                    Created {{ formatDate(workspace.createdAt) }}
                  </p>
                </div>
              </div>
            </v-card-item>
            <v-card-text>
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

    <v-dialog v-model="createDialogOpen" max-width="480">
      <v-card>
        <v-card-title class="text-h6 font-weight-semibold">Create Workspace</v-card-title>
        <v-card-text class="pt-4">
          <v-text-field
            v-model="newWorkspaceName"
            label="Workspace name"
            variant="outlined"
            density="comfortable"
            :disabled="creatingWorkspace"
            required
          />
          <v-textarea
            v-model="newWorkspaceInstructions"
            label="Instructions (optional)"
            variant="outlined"
            density="comfortable"
            auto-grow
            rows="3"
            :disabled="creatingWorkspace"
          />
          <v-alert
            v-if="createError"
            type="error"
            variant="tonal"
            density="comfortable"
            class="mt-3"
          >
            {{ createError }}
          </v-alert>
        </v-card-text>
        <v-card-actions class="justify-end">
          <v-btn
            variant="text"
            color="primary"
            @click="closeCreateWorkspace"
            :disabled="creatingWorkspace"
          >
            Cancel
          </v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :loading="creatingWorkspace"
            @click="handleCreateWorkspace"
          >
            Create
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useDashboardStore } from '@/stores'

const dashboardStore = useDashboardStore()
const router = useRouter()
const { workspaces, loading, error, hasWorkspaces } = storeToRefs(dashboardStore)

const createDialogOpen = ref(false)
const creatingWorkspace = ref(false)
const newWorkspaceName = ref('')
const newWorkspaceInstructions = ref('')
const createError = ref('')

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

const openCreateWorkspace = () => {
  newWorkspaceName.value = ''
  newWorkspaceInstructions.value = ''
  createError.value = ''
  createDialogOpen.value = true
}

const closeCreateWorkspace = () => {
  createDialogOpen.value = false
}

const handleCreateWorkspace = async () => {
  if (!newWorkspaceName.value.trim()) {
    createError.value = 'Workspace name is required.'
    return
  }

  creatingWorkspace.value = true
  createError.value = ''

  try {
    const workspace = await dashboardStore.createWorkspace({
      name: newWorkspaceName.value,
      instructions: newWorkspaceInstructions.value,
    })

    closeCreateWorkspace()

    if (workspace?.id) {
      router.push({ name: 'Studio', query: { workspaceId: workspace.id } })
    } else {
      await dashboardStore.initialise({ force: true })
    }
  } catch (error) {
    createError.value = error?.message || 'Failed to create workspace.'
  } finally {
    creatingWorkspace.value = false
  }
}

onMounted(() => {
  dashboardStore.initialise()
})
onMounted(() => {
  dashboardStore.initialise()
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

.dashboard-page__intro-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: flex-end;
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
}

.dashboard-page__grid-item {
  display: flex;
}

.dashboard-page__card {
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  display: flex;
  flex-direction: column;
  width: 100%;
}

.dashboard-page__card-header {
  display: flex;
  align-items: center;
  gap: 16px;
}

.dashboard-page__card-meta {
  display: flex;
  flex-direction: column;
}

@media (max-width: 960px) {
  .dashboard-page__content {
    padding: 24px 16px;
  }

  .dashboard-page__bar {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }
}
</style>
