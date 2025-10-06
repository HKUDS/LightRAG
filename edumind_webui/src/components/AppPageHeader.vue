<template>
  <v-app-bar flat color="white" class="app-page-header" height="76">
    <v-container class="py-0" fluid>
      <div class="app-page-header__bar">
        <div class="app-page-header__brand">
          <v-avatar color="primary" size="48" variant="flat">
            <span class="app-page-header__brand-initials text-subtitle-1 font-weight-semibold">{{ organizationInitials }}</span>
          </v-avatar>
          <div>
            <p class="text-overline text-uppercase text-medium-emphasis mb-1">{{ organization }}</p>
            <h1 class="text-h5 font-weight-semibold mb-0">{{ title }}</h1>
          </div>
        </div>
        <div class="app-page-header__actions" v-if="hasActions">
          <p v-if="description" class="text-body-2 text-medium-emphasis mb-0">{{ description }}</p>
          <div class="app-page-header__action-buttons">
            <v-btn
              v-if="showBack"
              variant="text"
              color="primary"
              prepend-icon="mdi-arrow-left"
              @click="handleBack"
            >
              Go Back
            </v-btn>
            <v-btn
              v-if="actionLabel"
              color="primary"
              variant="flat"
              class="px-6"
              :prepend-icon="actionIcon"
              v-bind="actionAttrs"
              @click="handleAction"
            >
              {{ actionLabel }}
            </v-btn>
          </div>
        </div>
      </div>
    </v-container>
  </v-app-bar>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useAppStore } from '@/stores';

interface RouteLike {
  name?: string;
  path?: string;
  params?: Record<string, unknown>;
  query?: Record<string, unknown>;
}

const props = withDefaults(
  defineProps<{
    title: string;
    description?: string;
    actionLabel?: string;
    actionIcon?: string;
    actionTo?: string | RouteLike;
    showBack?: boolean;
  }>(),
  {
    description: '',
    actionLabel: '',
    actionIcon: 'mdi-arrow-right',
    actionTo: undefined,
    showBack: false,
  }
);

const emit = defineEmits<{
  action: [];
}>();

const router = useRouter();
const appStore = useAppStore();

const organization = computed(() => appStore.organization);
const organizationInitials = computed(() => appStore.organizationInitials);

const hasActionLink = computed(() => props.actionTo !== undefined && props.actionTo !== null);
const actionAttrs = computed(() => (hasActionLink.value ? { to: props.actionTo } : {}));
const hasActions = computed(
  () => Boolean(props.description) || props.showBack || Boolean(props.actionLabel)
);

const handleAction = () => {
  if (!hasActionLink.value) {
    emit('action');
  }
};

const handleBack = () => {
  router.back();
};
</script>

<style scoped>
.app-page-header {
  border-bottom: 1px solid rgba(22, 101, 52, 0.12);
}

.app-page-header__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding-inline: 16px;
}

.app-page-header__brand {
  display: flex;
  align-items: center;
  gap: 16px;
}

.app-page-header__brand-initials {
  color: #ffffff;
}

.app-page-header__actions {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
}

.app-page-header__action-buttons {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

@media (max-width: 960px) {
  .app-page-header__bar {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }

  .app-page-header__actions {
    align-items: flex-start;
  }

  .app-page-header__action-buttons {
    justify-content: flex-start;
  }
}
</style>
