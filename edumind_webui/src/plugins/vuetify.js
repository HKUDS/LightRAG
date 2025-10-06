// Styles
import 'vuetify/styles'
import '@mdi/font/css/materialdesignicons.css'

// Vuetify
import { createVuetify } from 'vuetify'
import { aliases, mdi } from 'vuetify/iconsets/mdi'

const greenPrimary = '#166534'

const defaultTheme = {
  dark: false,
  colors: {
    background: '#FFFFFF',
    surface: '#FFFFFF',
    primary: greenPrimary,
    secondary: greenPrimary,
    info: greenPrimary,
    warning: greenPrimary,
    error: greenPrimary,
    success: greenPrimary,
  },
  variables: {
    'font-family': 'Overpass, sans-serif',
    'border-radius-root': '16px',
  },
}

export default createVuetify({
  theme: {
    defaultTheme: 'light',
    variations: {
      colors: ['primary', 'secondary', 'accent'],
      lighten: 2,
      darken: 2,
    },
    themes: {
      light: defaultTheme,
    },
  },
  icons: {
    defaultSet: 'mdi',
    aliases,
    sets: {
      mdi,
    },
  },
  defaults: {
    global: {
      ripple: false,
    },
    VCard: {
      elevation: 1,
      rounded: 'xl',
    },
    VBtn: {
      rounded: 'lg',
      style: 'text-transform: none; letter-spacing: 0;',
    },
  },
})
