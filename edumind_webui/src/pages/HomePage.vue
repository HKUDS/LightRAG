<template>
  <div class="home-page">
    <v-app-bar flat color="white" class="home-page__app-bar" height="76">
      <v-container class="py-0" fluid>
        <div class="home-page__bar">
          <div class="home-page__brand">
            <v-avatar color="primary" size="48" variant="flat">
              <span class="home-page__brand-initials text-subtitle-1 font-weight-semibold">{{ organizationInitials }}</span>
            </v-avatar>
            <div>
              <p class="text-overline text-uppercase text-medium-emphasis mb-1">{{ organization }}</p>
              <h1 class="text-h5 font-weight-semibold mb-0">{{ appName }}</h1>
            </div>
          </div>
          <div class="home-page__actions">
            <p class="text-body-2 text-medium-emphasis mb-0">{{ tagline }}</p>
            <v-btn
              color="primary"
              variant="flat"
              class="px-6"
              prepend-icon="mdi-file-document-multiple-outline"
              :to="{ name: 'Questions' }"
            >
              View Question Library
            </v-btn>
          </div>
        </div>
      </v-container>
    </v-app-bar>

    <v-container fluid class="home-page__content">
      <div class="home-page__workspace">
        <ToolsPanel class="home-page__workspace-panel" />
        <CanvasPanel class="home-page__workspace-canvas" />
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

const { appName, tagline, organization, organizationInitials } = storeToRefs(appStore)
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
.home-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  background-color: #ffffff;
}

.home-page__app-bar {
  border-bottom: 1px solid rgba(22, 101, 52, 0.12);
}

.home-page__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 32px;
  padding-inline: 16px;
}

.home-page__brand {
  display: flex;
  align-items: center;
  gap: 16px;
}

.home-page__actions {
  display: flex;
  align-items: center;
  gap: 24px;
}

.home-page__content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 32px 40px;
  gap: 24px;
  overflow: hidden;
}

.home-page__workspace {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(280px, 3fr) minmax(0, 7fr);
  gap: 24px;
  min-height: 0;
}

.home-page__workspace-panel,
.home-page__workspace-canvas {
  height: 100%;
  display: flex;
}

.home-page__brand-initials {
  color: #ffffff;
}

@media (max-width: 1280px) {
  .home-page__workspace {
    grid-template-columns: minmax(240px, 35%) minmax(0, 1fr);
  }
}

@media (max-width: 960px) {
  .home-page__content {
    padding: 24px 16px;
    overflow: auto;
  }

  .home-page__bar {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }

  .home-page__actions {
    width: 100%;
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .home-page__workspace {
    grid-template-columns: 1fr;
  }
}
</style>
