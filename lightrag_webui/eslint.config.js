import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import stylisticJs from '@stylistic/eslint-plugin-js'
import tseslint from 'typescript-eslint'
import prettier from 'eslint-config-prettier'
import react from 'eslint-plugin-react'

export default tseslint.config({ ignores: ['dist'] }, prettier, {
  extends: [js.configs.recommended, ...tseslint.configs.recommended],
  files: ['**/*.{ts,tsx,js,jsx}'],
  languageOptions: {
    ecmaVersion: 2020,
    globals: globals.browser
  },
  settings: { react: { version: '19.0' } },
  plugins: {
    'react-hooks': reactHooks,
    'react-refresh': reactRefresh,
    '@stylistic/js': stylisticJs,
    react
  },
  rules: {
    ...reactHooks.configs.recommended.rules,
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    ...react.configs.recommended.rules,
    ...react.configs['jsx-runtime'].rules,
    '@stylistic/js/indent': ['error', 2],
    '@stylistic/js/quotes': ['error', 'single'],
    '@typescript-eslint/no-explicit-any': ['off']
  }
})
