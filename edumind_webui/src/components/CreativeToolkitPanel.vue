<template>
  <v-card class="tools-panel" elevation="2">
    <v-card-text class="pt-4">
      <v-expansion-panels v-model="panelModel" multiple elevation="0" class="tool-expansion">
        <v-expansion-panel v-for="tool in tools" :key="tool.id" :value="tool.id">
          <v-expansion-panel-title expand-icon="mdi-chevron-down">
            <div class="d-flex align-center justify-space-between w-100">
              <div class="tools-panel__tool-meta">
                <v-avatar color="primary" size="44" variant="flat">
                  <v-icon :icon="tool.icon" size="22" color="white" />
                </v-avatar>
                <div>
                  <p class="text-subtitle-1 font-weight-semibold mb-1">{{ tool.name }}</p>
                  <p class="text-body-2 text-medium-emphasis mb-0">{{ tool.description }}</p>
                </div>
              </div>
            </div>
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <MCQGeneratorForm v-if="tool.id === 'mcq-generator'" />
            <div v-else class="tools-panel__cta">
              <v-btn
                color="primary"
                variant="flat"
                block
                style="padding-block: 14px;"
                prepend-icon="mdi-play-circle-outline"
                @click="() => handleExecute(tool.id)"
              >
                {{ tool.quickAction }}
              </v-btn>
            </div>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { computed, type Ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useHomeStore } from '@/stores'
import MCQGeneratorForm from './MCQGeneratorForm.vue'

interface ToolConfig {
  id: string
  name: string
  description: string
  icon: string
  quickAction: string
}

const homeStore = useHomeStore()
const storeRefs = storeToRefs(homeStore)

const tools = computed<ToolConfig[]>(() => storeRefs.tools.value as ToolConfig[])
const expandedToolIds = storeRefs.expandedToolIds as Ref<string[]>

const panelModel = computed<string[]>({
  get: () => expandedToolIds.value,
  set: (value) => {
    const parsed = Array.isArray(value) ? value : value ? [value] : []
    homeStore.setExpandedToolIds(parsed)
  },
})

const handleExecute = (toolId: ToolConfig['id']) => {
  if (toolId === 'mcq-generator') {
    return
  }
  homeStore.executeTool(toolId)
}
</script>

<style scoped>
.tools-panel {
  height: 100%;
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  backdrop-filter: blur(10px);
}

.tools-panel__header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.tools-panel__tool-meta {
  display: flex;
  align-items: center;
  gap: 16px;
}

.tool-expansion :deep(.v-expansion-panel-title) {
  padding: 20px;
}

.tool-expansion :deep(.v-expansion-panel-text__wrapper) {
  padding: 0 20px 24px 20px;
}

.tools-panel__cta {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
</style>
