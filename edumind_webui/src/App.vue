<template>
  <v-app>
    <AppPageHeader
      :title="headerTitle"
      :description="headerDescription"
      :show-back="headerShowBack"
      :actions="headerActions"
    />
    <v-main class="app-main">
      <router-view />
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import { storeToRefs } from 'pinia'
import AppPageHeader from '@/components/AppPageHeader.vue'
import { useAppStore, useHeaderStore } from '@/stores'

const route = useRoute()
const appStore = useAppStore()
const headerStore = useHeaderStore()

const { title, description, showBack, actions } = storeToRefs(headerStore)

const headerTitle = computed(() => title.value || appStore.appName)
const headerDescription = computed(() => description.value || '')
const headerShowBack = computed(() => showBack.value)
const headerActions = computed(() => actions.value || [])

watch(
  () => route.name,
  () => {
    // If a route doesn't set header data, fallback to defaults.
    if (!title.value) {
      headerStore.setHeader({ title: appStore.appName })
    }
  },
  { immediate: true }
)
</script>

<style scoped>

.app-main {
  min-height: 100vh;
  background-color: #ffffff;
}
</style>
