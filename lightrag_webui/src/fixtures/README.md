# Little Bull Knowledge fixtures

Dados determinísticos para prototipar a nova experiência doméstica e simples da
WebUI sem depender de contratos backend ainda não implementados.

## O que cobre

- Áreas/workspaces: Casa, Família, Finanças, Trabalho, Estudos e Pequeno negócio.
- Documentos com status humano, tags, confidencialidade e ações sugeridas.
- Conversas com respostas, confiança e citações de fontes.
- Assistentes por área.
- Agentes internos, subagentes e skills/ações em linguagem de produto.
- Aprovações humanas para ações destrutivas.
- Planos resumidos, critic findings e trilha de auditoria sem chain-of-thought bruto.
- Estado de sincronização do catálogo OpenRouter.
- Atividades recentes.
- Perfis de modelo em linguagem leiga.
- Estado inicial da Home.

## Como usar

Importe de `@/fixtures/littleBullKnowledge` em telas novas ou histórias visuais:

```ts
import {
  areasFixture,
  getDocumentsByWorkspace,
  getWorkspaceSuggestedQuestions
} from '@/fixtures/littleBullKnowledge'
```

Os IDs de área já são compatíveis com o header `LIGHTRAG-WORKSPACE`.
