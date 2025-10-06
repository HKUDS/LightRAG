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
      <v-row class="home-page__grid" align="stretch" justify="stretch">
        <v-col cols="12" md="4" lg="3" class="home-page__tools">
          <ToolsPanel class="w-100" />
        </v-col>
        <v-col cols="12" md="8" lg="9" class="home-page__canvas">
          <CanvasPanel class="w-100" />
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { storeToRefs } from 'pinia';
import ToolsPanel from '@/components/ToolsPanel.vue';
import CanvasPanel from '@/components/CanvasPanel.vue';
import { useAppStore, useHomeStore } from '@/stores';

const appStore = useAppStore();
const homeStore = useHomeStore();

const { appName, tagline, organization, organizationInitials } = storeToRefs(appStore);

onMounted(() => {
  homeStore.initialise();
});
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

.home-page__grid {
  flex: 1;
  min-height: 0;
  gap: 24px;
}

.home-page__tools,
.home-page__canvas {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.home-page__brand-initials {
  color: #ffffff;
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
}
</style>
