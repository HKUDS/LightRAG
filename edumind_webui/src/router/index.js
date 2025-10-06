import { createRouter, createWebHistory } from 'vue-router';
import HomePage from '@/pages/HomePage.vue';
import QuestionsPage from '@/pages/QuestionsPage.vue';

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomePage
  },
  {
    path: '/questions',
    name: 'Questions',
    component: QuestionsPage
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;