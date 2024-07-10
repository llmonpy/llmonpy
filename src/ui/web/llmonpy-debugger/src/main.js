/**
 * main.js
 *
 * Bootstraps Vuetify and other plugins then mounts the App`
 */

// Plugins
import { registerPlugins } from '@/plugins'

// Components
import App from './App.vue'

// Composables
import { createApp } from 'vue'
import {InitLLMonPyScopeAPI} from "@/js/api";

const app = createApp(App)

registerPlugins(app)
InitLLMonPyScopeAPI(import.meta.env.VITE_APP_API_URL);

app.mount('#app')
