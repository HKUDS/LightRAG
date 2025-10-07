<template>
  <div class="login-page">
    <v-card class="login-card" elevation="8">
      <div class="login-card__brand">
        <v-avatar size="56" color="primary" variant="flat">
          <span class="text-h6 font-weight-bold text-white">EM</span>
        </v-avatar>
        <div>
          <p class="login-card__eyebrow">EduMind Studio</p>
          <h1 class="login-card__title">Sign in to continue</h1>
          <p class="login-card__subtitle">Use the credentials shared with your team lead.</p>
        </div>
      </div>

      <v-form class="login-card__form" @submit.prevent="handleSubmit">
        <v-text-field
          v-model="username"
          label="Username"
          variant="outlined"
          density="comfortable"
          autocomplete="username"
          required
          :disabled="loading"
        />
        <v-text-field
          v-model="password"
          label="Password"
          variant="outlined"
          density="comfortable"
          type="password"
          autocomplete="current-password"
          required
          :disabled="loading"
        />

        <v-alert
          v-if="errorMessage"
          type="error"
          variant="tonal"
          density="comfortable"
        >
          {{ errorMessage }}
        </v-alert>

        <v-btn
          color="primary"
          variant="flat"
          block
          size="large"
          type="submit"
          :loading="loading"
          :disabled="loading"
        >
          Sign In
        </v-btn>
      </v-form>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useUserStore } from '@/stores'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()

const username = ref('')
const password = ref('')

const loading = computed(() => userStore.loading)
const errorMessage = computed(() => userStore.error?.message || '')

const handleSubmit = async () => {
  if (loading.value) {
    return
  }

  try {
    await userStore.signIn({ username: username.value, password: password.value })
    const redirectPath = typeof route.query.redirect === 'string' ? route.query.redirect : undefined
    if (redirectPath) {
      router.replace(redirectPath)
    } else {
      router.replace({ name: 'Dashboard' })
    }
  } catch (error) {
    // error already captured in store; no-op
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(circle at top, rgba(22, 101, 52, 0.12), transparent 55%),
    radial-gradient(circle at bottom, rgba(16, 185, 129, 0.08), transparent 50%),
    #f5f7fb;
  padding: 24px;
}

.login-card {
  width: min(420px, 100%);
  padding: 32px 28px;
  border-radius: 28px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  backdrop-filter: blur(8px);
}

.login-card__brand {
  display: flex;
  align-items: center;
  gap: 16px;
}

.login-card__eyebrow {
  margin: 0;
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  color: rgba(30, 41, 59, 0.6);
}

.login-card__title {
  margin: 4px 0;
  font-size: 1.45rem;
  font-weight: 600;
  color: rgba(15, 23, 42, 0.92);
}

.login-card__subtitle {
  margin: 0;
  color: rgba(30, 41, 59, 0.72);
  font-size: 0.95rem;
}

.login-card__form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.login-card :deep(.v-alert) {
  margin-top: 8px;
}

@media (max-width: 480px) {
  .login-card {
    padding: 24px 20px;
  }
}
</style>