<template>
  <v-card class="canvas-panel" elevation="2">
    <v-card-item class="pb-0">
      <div class="canvas-panel__header">
        <div class="canvas-panel__actions">
          <v-btn
            variant="outlined"
            color="primary"
            prepend-icon="mdi-broom"
            :disabled="!hasOutputs"
            @click="handleDiscardOutputs"
          >
            Clear Canvas
          </v-btn>
          <v-btn
            color="primary"
            variant="flat"
            prepend-icon="mdi-plus-circle-outline"
            :disabled="!canAddTab || creatingCanvas"
            :loading="creatingCanvas"
            @click="handleAddTab"
          >
            New Canvas
          </v-btn>
        </div>
      </div>
    </v-card-item>

    <v-card-text class="canvas-panel__body">
      <v-tabs v-model="tabModel" class="canvas-panel__tabs" show-arrows color="primary">
        <v-tab
          v-for="tab in tabs"
          :key="tab.id"
          :value="tab.id"
          class="text-subtitle-2 font-weight-medium text-capitalize"
        >
          <div class="canvas-panel__tab">
            <v-icon size="18">mdi-notebook-outline</v-icon>
            <span>{{ tab.name }}</span>
            <v-btn
              size="x-small"
              icon="mdi-close"
              variant="text"
              color="primary"
              :loading="isDeletingCanvas(tab.id)"
              :disabled="isDeletingCanvas(tab.id)"
              @click.stop="handleCloseTab(tab.id)"
            />
          </div>
        </v-tab>
      </v-tabs>

      <v-window v-model="tabModel" class="canvas-panel__window">
        <v-window-item v-for="tab in tabs" :key="tab.id" :value="tab.id">
          <div v-if="tab.outputs.length === 0" class="canvas-panel__empty">
            <v-sheet class="canvas-panel__empty-sheet" color="white" variant="outlined">
              <v-avatar color="primary" size="56" variant="flat">
                <v-icon size="28" color="white">mdi-compass-outline</v-icon>
              </v-avatar>
              <h3 class="text-subtitle-1 font-weight-semibold mt-4 mb-2">Your canvas is ready</h3>
              <p class="text-body-2 text-medium-emphasis mb-0">
                Trigger a tool from the left panel to see fresh content appear here.
              </p>
            </v-sheet>
          </div>
          <div v-else class="canvas-panel__list">
            <template v-for="output in tab.outputs" :key="output.id">
              <MCQCard
                v-if="output.type === 'mcq'"
                :mcq="output"
                :show-actions="true"
                @delete="handleDeleteOutput"
              />
              <AssignmentCard
                v-else
                :assignment="output"
                :show-actions="true"
                @delete="handleDeleteOutput"
              />
            </template>
          </div>
        </v-window-item>
      </v-window>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { storeToRefs } from 'pinia';
import MCQCard from '@/components/MCQCard.vue';
import AssignmentCard from '@/components/AssignmentCard.vue';
import { useHomeStore } from '@/stores';

const homeStore = useHomeStore();
const { tabs, activeTabId, activeTab, hasActiveOutputs, canAddTab, creatingCanvas } = storeToRefs(homeStore);

const tabModel = computed<string>({
  get: () => activeTabId.value,
  set: (value) => {
    if (typeof value === 'string') {
      homeStore.setActiveTab(value);
    }
  },
});

const hasOutputs = computed(() => hasActiveOutputs.value);
const isDeletingCanvas = (tabId: string) => homeStore.isDeletingCanvas(tabId);

const handleAddTab = async () => {
  if (!canAddTab.value || creatingCanvas.value) {
    return;
  }
  try {
    await homeStore.addTab();
  } catch (error) {
    console.error('Failed to create canvas', error);
  }
};

const handleCloseTab = async (tabId: string) => {
  const targetTab = tabs.value.find((tab) => tab.id === tabId);
  if (!targetTab) {
    return;
  }

  const confirmed = window.confirm(`Close ${targetTab.name}? All outputs on this canvas will be removed.`);
  if (confirmed) {
    try {
      await homeStore.closeTab(tabId);
    } catch (error) {
      console.error('Failed to close canvas', error);
    }
  }
};

const handleDiscardOutputs = () => {
  if (!hasOutputs.value) {
    return;
  }

  const confirmed = window.confirm('Discard all outputs on this canvas?');
  if (confirmed) {
    homeStore.discardActiveTabOutputs();
  }
};

const handleDeleteOutput = (outputId: string) => {
  homeStore.deleteOutputById(outputId);
};
</script>

<style scoped>
.canvas-panel {
  border-radius: 28px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.canvas-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
}

.canvas-panel__actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.canvas-panel__body {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px;
  gap: 16px;
}

.canvas-panel__tabs :deep(.v-tab__slider) {
  border-radius: 999px 999px 0 0;
}

.canvas-panel__tabs :deep(.v-tab) {
  padding: 12px 20px;
  min-width: 140px;
}

.canvas-panel__tab {
  display: flex;
  align-items: center;
  gap: 8px;
}

.canvas-panel__window {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.canvas-panel__window :deep(.v-window-item) {
  height: 100%;
}

.canvas-panel__empty {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  padding: 24px;
}

.canvas-panel__empty-sheet {
  max-width: 420px;
  text-align: center;
  border-radius: 24px;
  padding: 40px 32px;
  border: 1px dashed rgba(22, 101, 52, 0.18);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.canvas-panel__list {
  display: flex;
  flex-direction: column;
  gap: 24px;
  height: 100%;
  overflow-y: auto;
  padding-right: 8px;
}

@media (max-width: 960px) {
  .canvas-panel__body {
    padding: 16px;
  }

  .canvas-panel__tabs :deep(.v-tab) {
    min-width: 120px;
  }
}
</style>
