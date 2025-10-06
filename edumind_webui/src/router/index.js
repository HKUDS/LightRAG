import { createRouter, createWebHistory } from 'vue-router';
import DashboardPage from '@/pages/DashboardPage.vue';
import HomePage from '@/pages/HomePage.vue';
import QuestionsPage from '@/pages/QuestionsPage.vue';

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: DashboardPage
  },
  {
    path: '/workspace',
    name: 'Workspace',
    component: HomePage
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
