<template>
  <div class="studio-page">
    <v-container fluid class="studio-page__content">
      <div class="studio-page__workspace">
        <ToolsPanel class="studio-page__workspace-panel" />
        <CanvasPanel class="studio-page__workspace-canvas" />
      </div>
    </v-container>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, onActivated, onDeactivated, watch, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'
import ToolsPanel from '@/components/ToolsPanel.vue'
import CanvasPanel from '@/components/CanvasPanel.vue'
import {
  useAppStore,
  useHomeStore,
  useDashboardStore,
  useWorkspaceContextStore,
  useHeaderStore,
} from '@/stores'

const appStore = useAppStore()
const homeStore = useHomeStore()
const dashboardStore = useDashboardStore()
const workspaceContextStore = useWorkspaceContextStore()
const headerStore = useHeaderStore()
const route = useRoute()
const router = useRouter()

const { appName, tagline } = storeToRefs(appStore)
const { workspaces } = storeToRefs(dashboardStore)
const { workspaceId } = storeToRefs(workspaceContextStore)

const routeWorkspaceId = computed(() => {
  const value = route.query.workspaceId
  return Array.isArray(value) ? value[0] : (value || '')
})

const setWorkspaceContext = () => {
  let targetId = routeWorkspaceId.value
  let targetName = ''

  if (!targetId && workspaces.value.length > 0) {
    targetId = workspaces.value[0].id
    targetName = workspaces.value[0].name
  }

  if (targetId) {
    const matched = workspaces.value.find((w) => w.id === targetId)
    if (matched) targetName = matched.name
  }

  if (!targetId) {
    workspaceContextStore.resetWorkspace()
    return
  }
  workspaceContextStore.setWorkspace({ id: targetId, name: targetName })
}

const applyHeader = () => {
  headerStore.setHeader({
    title: appName.value,
    description: tagline.value,
    actions: [
      {
        id: 'studio-go-questions',
        label: 'Go to Questions Library',
        icon: 'mdi-file-document-multiple-outline',
        variant: 'text',
        to: { name: 'Questions' },
      },
    ],
  })
}

onMounted(() => {
  homeStore.initialise()
  if (typeof dashboardStore.initialise === 'function') {
    dashboardStore.initialise()
  }
  setWorkspaceContext()
  applyHeader()
})

onActivated(() => {
  // Re-apply header and ensure workspace context is up to date when navigating back
  setWorkspaceContext()
  applyHeader()
})

onDeactivated(() => {
  headerStore.resetHeader()
})

onUnmounted(() => {
  headerStore.resetHeader()
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

// Keep header reactive to name/tagline changes too
watch([appName, tagline], () => applyHeader())
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
