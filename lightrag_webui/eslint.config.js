import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import stylistic from '@stylistic/eslint-plugin'
import tseslint from 'typescript-eslint'
import prettier from 'eslint-config-prettier'
import react from 'eslint-plugin-react'

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended, prettier],
    files: ['**/*.{ts,tsx,js,jsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser
    },
    settings: { react: { version: '19.0' } },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
      '@stylistic': stylistic,
      react
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      ...react.configs.recommended.rules,
      ...react.configs['jsx-runtime'].rules,
      '@stylistic/indent': ['error', 2],
      '@stylistic/quotes': ['error', 'single'],
      '@typescript-eslint/no-explicit-any': ['off']
    }
  },
  {
    // Test files and the test harness under src/test/ are never part of the
    // production bundle or the HMR graph, so two rules that exist to protect
    // those do not apply here.
    files: ['src/test/**/*.{ts,tsx}', '**/*.test.{ts,tsx}'],
    rules: {
      // Merging jest-dom's matchers into bun:test's `Matchers` is by
      // definition an interface that declares no members of its own.
      '@typescript-eslint/no-empty-object-type': 'off',
      // Render helpers legitimately export both a wrapper component and the
      // helpers around it; Fast Refresh never sees this file.
      'react-refresh/only-export-components': 'off'
    }
  }
)
