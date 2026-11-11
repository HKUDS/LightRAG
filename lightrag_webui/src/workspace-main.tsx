// MUST stay the FIRST static import: runs the settings-storage split
// migration before any store module is evaluated (ESM evaluates same-level
// imports depth-first in source order). See migrations/splitSettingsStorage.ts.
import '@/migrations/runSettingsStorageSplit'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import WorkspaceAppRouter from './WorkspaceAppRouter'
import './i18n.ts';
import 'katex/dist/katex.min.css';
// Import KaTeX extensions at app startup to ensure they are registered before any rendering
import 'katex/contrib/mhchem'; // Chemistry formulas: \ce{} and \pu{}
import 'katex/contrib/copy-tex'; // Allow copying rendered formulas as LaTeX source

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <WorkspaceAppRouter />
  </StrictMode>
)
