import { createRouter, createWebHistory } from 'vue-router';
import DashboardPage from '@/pages/DashboardPage.vue';
import StudioPage from '@/pages/StudioPage.vue';
import QuestionsPage from '@/pages/QuestionsPage.vue';

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: DashboardPage
  },
  {
    path: '/studio',
    name: 'Studio',
    component: StudioPage
  },
  {
    path: '/workspace',
    redirect: { name: 'Studio' }
  },
  {
    path: '/questions',
    name: 'Questions',
    component: QuestionsPage
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

export default router;
