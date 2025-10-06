<template>
  <div class="dashboard-page">
    <v-app-bar flat color="white" class="dashboard-page__app-bar" height="76">
      <v-container class="py-0" fluid>
        <div class="dashboard-page__bar">
          <div>
            <p class="text-overline text-uppercase text-medium-emphasis mb-1">Your Spaces</p>
            <h1 class="text-h5 font-weight-semibold mb-0">Workspace Dashboard</h1>
          </div>
          <div class="dashboard-page__actions">
            <v-btn
              color="primary"
              variant="flat"
              prepend-icon="mdi-notebook-edit-outline"
              :to="{ name: 'Workspace' }"
            >
              Go to Canvas Studio
            </v-btn>
          </div>
        </div>
      </v-container>
    </v-app-bar>

    <v-container fluid class="dashboard-page__content">
      <div class="dashboard-page__intro">
        <div>
          <h2 class="text-subtitle-1 font-weight-semibold mb-1">Curate your knowledge spaces</h2>
          <p class="text-body-2 text-medium-emphasis mb-0">
            Review the workspaces generated for user 101 and continue your creation flow.
          </p>
        </div>
        <v-btn
          variant="outlined"
          color="primary"
          prepend-icon="mdi-refresh"
          :loading="loading"
          @click="refresh"
        >
          Refresh
        </v-btn>
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
                :to="{ name: 'Workspace', query: { workspaceId: workspace.id } }"
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
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useDashboardStore } from '@/stores'

const dashboardStore = useDashboardStore()
const { workspaces, loading, error, hasWorkspaces } = storeToRefs(dashboardStore)

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

.dashboard-page__app-bar {
  border-bottom: 1px solid rgba(22, 101, 52, 0.12);
}

.dashboard-page__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding-inline: 16px;
}

.dashboard-page__actions {
  display: flex;
  align-items: center;
  gap: 16px;
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
