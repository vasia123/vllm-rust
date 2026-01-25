import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import './style.css'
import App from './App.vue'
import Dashboard from './views/Dashboard.vue'
import Settings from './views/Settings.vue'

const routes = [
  { path: '/', redirect: '/admin' },
  { path: '/admin', component: Dashboard },
  { path: '/admin/settings', component: Settings },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

createApp(App).use(router).mount('#app')
