import { createRouter, createWebHistory } from 'vue-router';
import DashboardPage from '@/pages/DashboardPage.vue';
import StudioPage from '@/pages/StudioPage.vue';
import QuestionsPage from '@/pages/QuestionsPage.vue';
import LoginPage from '@/pages/LoginPage.vue';
import { useUserStore } from '@/stores';

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: DashboardPage,
    meta: { requiresAuth: true }
  },
  {
    path: '/studio',
    name: 'Studio',
    component: StudioPage,
    meta: { requiresAuth: true }
  },
  {
    path: '/workspace',
    redirect: { name: 'Studio' }
  },
  {
    path: '/questions',
    name: 'Questions',
    component: QuestionsPage,
    meta: { requiresAuth: true }
  },
  {
    path: '/login',
    name: 'Login',
    component: LoginPage,
    meta: { requiresAuth: false }
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: { name: 'Dashboard' }
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

router.beforeEach(async (to, from, next) => {
  const userStore = useUserStore();

  if (!userStore.hydrated && !userStore.loading) {
    await userStore.hydrate();
  }

  if (to.meta?.requiresAuth === false) {
    if (userStore.isAuthenticated && to.name === 'Login') {
      return next({ name: 'Dashboard' });
    }
    return next();
  }

  if (!userStore.isAuthenticated) {
    return next({
      name: 'Login',
      query: to.fullPath ? { redirect: to.fullPath } : undefined,
    });
  }

  return next();
});

export default router;