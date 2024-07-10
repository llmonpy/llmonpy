// Composables
import { createRouter, createWebHistory } from 'vue-router'
import { API } from "@/js/api";

const routes = [
  {
    path: '/',
    component: () => import('@/layouts/default/Default.vue'),
    children: [
      {
        path: '',
        name: 'LLMonPy Trace Viewer',
        component: () => import('@/views/TraceViewer.vue'),
      },
      {
        path: '/qbawa',
        name: 'Qbawa Viewer',
        component: () => import('@/views/QbawaViewer.vue'),
      },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
})

/*
router.beforeEach(async (to) => {
  if ( API.isLoggedIn() == false && to.name != "Login" ) {
    return "/login";
  }
});
*/

export default router
