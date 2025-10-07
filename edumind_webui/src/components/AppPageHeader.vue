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
        <div class="app-page-header__actions" v-if="hasVisibleActions">
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
              v-for="action in actions"
              :key="action.id"
              :variant="action.variant || 'flat'"
              :color="action.color || 'primary'"
              class="px-6"
              :prepend-icon="action.icon"
              :to="action.to"
              :disabled="action.disabled"
              :loading="action.loading"
              @click="handleActionClick(action)"
            >
              {{ action.label }}
            </v-btn>
          </div>
        </div>
      </div>
    </v-container>
    <v-btn
      v-if="userStore.isAuthenticated"
      variant="flat"
      style="margin: 16px"
      prepend-icon="mdi-logout"
      color="red"
      @click="handleLogout"
    >
      Logout
    </v-btn>
  </v-app-bar>
</template>

<script setup lang="ts">
import { computed, nextTick } from 'vue';
import { useRouter, type RouteLocationRaw } from 'vue-router';
import { useAppStore } from '@/stores';
import { useUserStore } from '@/stores';
import { signOut } from '@/api/auth';

interface HeaderAction {
  id: string;
  label: string;
  icon?: string;
  variant?: 'flat' | 'text' | 'outlined';
  color?: string;
  to?: RouteLocationRaw;
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void | Promise<void>;
}

const props = withDefaults(
  defineProps<{
    title?: string;
    description?: string;
    showBack?: boolean;
    actions?: HeaderAction[];
  }>(),
  {
    title: '',
    description: '',
    showBack: false,
    actions: () => [],
  }
);

const router = useRouter();
const appStore = useAppStore();
const userStore = useUserStore();

const organization = computed(() => appStore.organization);
const organizationInitials = computed(() => appStore.organizationInitials);
const actions = computed(() => props.actions ?? []);

const hasVisibleActions = computed(
  () => Boolean(props.description) || props.showBack || actions.value.length > 0
);

const handleBack = () => {
  router.back();
};

const handleActionClick = (action: HeaderAction) => {
  if (typeof action.onClick === 'function') {
    action.onClick();
  }
};

const handleLogout = async () => {
  try {
    await signOut()
  } catch (e) {
    console.error(e)
  } finally {
    userStore.clearAuth()
    await nextTick()
    await router.push({ name: 'Login' })
  }
}
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
