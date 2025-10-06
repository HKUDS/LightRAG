import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import { registerPlugins } from './plugins'
import './style.css'

const app = createApp(App)
const pinia = createPinia()

registerPlugins(app)

app.use(pinia)
app.use(router)
app.mount('#app')
