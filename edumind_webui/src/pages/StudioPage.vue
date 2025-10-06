<template>
  <div class="studio-page">
    <AppPageHeader
      :title="appName"
      :description="tagline"
      action-label="Go to Questions Library"
      action-icon="mdi-file-document-multiple-outline"
      :action-to="{ name: 'Questions' }"
    />

    <v-container fluid class="studio-page__content">
      <div class="studio-page__workspace">
        <ToolsPanel class="studio-page__workspace-panel" />
        <CanvasPanel class="studio-page__workspace-canvas" />
      </div>
    </v-container>
  </div>
</template>

<script setup lang="ts">
import { onMounted, watch, computed } from 'vue'
import { useRoute } from 'vue-router'
import { storeToRefs } from 'pinia'
import ToolsPanel from '@/components/ToolsPanel.vue'
import CanvasPanel from '@/components/CanvasPanel.vue'
import AppPageHeader from '@/components/AppPageHeader.vue'
import {
  useAppStore,
  useHomeStore,
  useDashboardStore,
  useWorkspaceContextStore,
} from '@/stores'

const appStore = useAppStore()
const homeStore = useHomeStore()
const dashboardStore = useDashboardStore()
const workspaceContextStore = useWorkspaceContextStore()
const route = useRoute()

const { appName, tagline } = storeToRefs(appStore)
const { workspaces } = storeToRefs(dashboardStore)
const { workspaceId } = storeToRefs(workspaceContextStore)

const routeWorkspaceId = computed(() => {
  const value = route.query.workspaceId
  if (Array.isArray(value)) {
    return value[0]
  }
  return value || ''
})

const setWorkspaceContext = () => {
  let targetId = routeWorkspaceId.value
  let targetName = ''

  if (!targetId && workspaces.value.length > 0) {
    targetId = workspaces.value[0].id
    targetName = workspaces.value[0].name
  }

  if (targetId) {
    const matched = workspaces.value.find((workspace) => workspace.id === targetId)
    if (matched) {
      targetName = matched.name
    }
  }

  if (!targetId) {
    workspaceContextStore.resetWorkspace()
    return
  }

  workspaceContextStore.setWorkspace({ id: targetId, name: targetName })
}

onMounted(() => {
  homeStore.initialise()
  if (typeof dashboardStore.initialise === 'function') {
    dashboardStore.initialise()
  }
  setWorkspaceContext()
})

watch([routeWorkspaceId, workspaces], () => {
  setWorkspaceContext()
})

watch(
  workspaceId,
  (value) => {
    homeStore.loadProjectCanvases({ projectId: value, force: true }).catch(() => {})
  },
  { immediate: true }
)
</script>

<style scoped>
.studio-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  background-color: #ffffff;
}

.studio-page__content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 32px 40px;
  gap: 24px;
  overflow: hidden;
}

.studio-page__workspace {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(280px, 3fr) minmax(0, 7fr);
  gap: 24px;
  min-height: 0;
}

.studio-page__workspace-panel,
.studio-page__workspace-canvas {
  height: 100%;
  display: flex;
  min-height: 0;
}

@media (max-width: 1280px) {
  .studio-page__workspace {
    grid-template-columns: minmax(240px, 35%) minmax(0, 1fr);
  }
}

@media (max-width: 960px) {
  .studio-page__content {
    padding: 24px 16px;
    overflow: auto;
  }

  .studio-page__workspace {
    grid-template-columns: 1fr;
  }
}
</style>
