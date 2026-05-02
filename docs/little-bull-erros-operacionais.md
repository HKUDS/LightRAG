# Little Bull/TRAG - erros operacionais que nao podem se repetir

Data: 2026-05-02
Escopo: Docker local, autenticacao, UI Little Bull Premium e validacao visual.
Status: ativo. Este documento e bloqueante para qualquer declaracao de READY.

## Regra de ouro

Nao declarar "pronto", "READY", "no ar funcional" ou "entregue" so porque Docker subiu, build passou, endpoint respondeu 200 ou contrato backend existe.

Para declarar entregue, precisa haver evidencia minima:

- login visual feito como usuario real;
- smoke visual da pagina no navegador;
- console e network sem erro critico;
- fluxo principal executado, nao apenas tela renderizada;
- screenshot ou relatorio objetivo;
- riscos residuais tratados ou com plano claro de correcao.

## Erros cometidos

1. Declarei o sistema "no ar" antes de validar a experiencia humana no navegador.
2. Confundi Docker ativo e HTTP 200 com produto utilizavel.
3. Nao detectei cedo que a tela de login podia ficar branca por token antigo/stale session.
4. Demorei a entregar acesso local usavel e a ajustar autenticacao para modo enterprise.
5. Tratei backend, schema e contratos como se bastassem para validar produto premium.
6. Nao auditei cada pagina premium antes de sugerir que o sistema estava pronto.
7. Deixei varias paginas distintas renderizarem o mesmo componente generico `PremiumModulePage`.
8. Aceitei "Notas", "Inbox", "Daily Notes", "Canvas", "MOCs", "Trilhas" e "Agent Builder" como paginas quando eram shells genericos.
9. Nao exigi API client frontend para todos os endpoints backend ja existentes.
10. Nao capturei antes os erros HTTP 500 em rotas administrativas.
11. Nao diagnostiquei cedo o erro de excesso de conexoes PostgreSQL no Docker local.
12. Nao usei subagentes cedo o suficiente, apesar da solicitacao explicita.
13. Nao mantive um ledger simples de "erros a nao repetir" durante a execucao longa.
14. Nao tratei riscos residuais com energia suficiente antes de tentar avancar fase.
15. Confundi comunicacao de "ambiente local subiu" com "produto esta pronto para uso".

## Evidencias confirmadas por subagentes

Auditoria visual read-only:

- funcional de ponta a ponta: 0 paginas;
- parcial: 7 paginas;
- placeholder: 17 paginas;
- quebrada visual: 0 paginas, mas com erros 500;
- causa relevante dos 500: `asyncpg.exceptions.TooManyConnectionsError`.

Auditoria frontend/read-only:

- `grupos`, `subgrupos`, `notas`, `inbox`, `daily`, `canvas`, `mocs`, `trilhas`, `agent-builder`, `modelos`, `custos`, `jobs`, `juridico`, `relatorios`, `auditoria` e `aprovacoes` caem em `PremiumModulePage`;
- `PremiumModulePage` nao implementa editor Markdown, wikilinks, backlinks, inbox real, daily note, canvas visual, MOCs, trilhas ou builder por chat;
- `lightrag_webui/src/api/lightrag.ts` nao expoe clients para grande parte das rotas Obsidian-like ja existentes no backend.

Auditoria backend/contratos:

- backend tem endpoints reais para notas, tags, backlinks, provenance, inbox, daily notes, canvas, content maps, trilhas e agent builder;
- o problema principal desta frente e a ponte frontend/API client/UI, nao ausencia total de contrato backend.

## Status real das paginas

| Pagina | Status real | Problema principal |
|---|---|---|
| Dashboard | parcial | metricas e cards podem falhar por 500 |
| Workspaces | parcial | sem prova de criacao/edicao real |
| Grupos | placeholder | sem CRUD/workflow proprio |
| Subgrupos | placeholder | sem CRUD/workflow proprio |
| Documentos | parcial | upload/listagem nao provados ponta a ponta |
| Notas | placeholder | sem editor Markdown, wikilinks, tags ou backlinks |
| Inbox | placeholder | sem triagem, status, prioridade ou conversao |
| Daily Notes | placeholder | sem daily note automatica ou resumo do dia |
| Canvas | placeholder | sem board visual, nodes, edges ou drag/drop |
| MOCs | placeholder | sem mapa de conteudo navegavel |
| Trilhas | placeholder | sem construtor de trilha e passos |
| Grafo | parcial | dados invalidos podem quebrar confianca |
| Chat | parcial | query real e contexto operacional incompletos |
| Agent Builder | placeholder | sem builder por chat, sessoes ou publish review |
| Assistentes | parcial | sem validacao completa de criacao/uso |
| Modelos | placeholder | 500 em admin/models |
| Custos | placeholder | sem ledger navegavel enterprise |
| Jobs | placeholder | sem job manager real |
| Juridico | placeholder | sem revisao humana e fluxo processual real |
| Relatorios | placeholder | sem builder/export governado |
| Atividade | parcial | timeline simples |
| Auditoria | placeholder | sem filtros enterprise |
| Aprovacoes | placeholder | sem fluxo real de aprovacao |
| Admin | parcial | admin parcialmente usavel, com 500 em endpoints |

## Correcoes ja aplicadas

- Dockerfile passou a incluir `lightrag_enterprise` na imagem.
- Docker local foi colocado em modo de autenticacao enterprise.
- Usuario MASTER local foi bootstrapado sem registrar segredo no git.
- Login passou a limpar token stale ao abrir `/login`, evitando loop/tela branca.
- Build frontend, testes Bun, TypeScript e lint passaram apos a correcao de login.

## Bloqueios atuais

1. Paginas premium ainda nao sao enterprise interativas.
2. `PremiumModulePage` mascara modulos diferentes como se fossem entregas reais.
3. Faltam clients TypeScript para notas, inbox, daily notes, canvas, MOCs, trilhas e agent builder.
4. Docker local ainda precisa correcao de pool/conexoes para eliminar `TooManyConnectionsError`.
5. Faltam testes visuais que falhem quando paginas diferentes renderizam o mesmo shell generico.
6. Faltam fluxos de ponta a ponta com evidencia para cada modulo premium.

## Checklist obrigatorio antes de falar "pronto"

- [ ] `git status --short` revisado.
- [ ] Sem secrets, `.env`, tokens ou senhas em logs/docs/commits.
- [ ] Docker local sobe sem erro critico nos logs.
- [ ] Login testado visualmente no navegador.
- [ ] Cada pagina do menu foi aberta no navegador.
- [ ] Cada pagina tem componente proprio ou justificativa explicita para compartilhar componente.
- [ ] Nenhuma pagina premium critica e apenas placeholder.
- [ ] Console do navegador sem erro bloqueante.
- [ ] Network sem HTTP 500 em fluxo normal.
- [ ] API client frontend existe para os endpoints usados pela pagina.
- [ ] Fluxo principal da pagina foi executado, nao apenas renderizado.
- [ ] Teste automatizado cobre o bug corrigido.
- [ ] Teste visual ou Playwright cobre as paginas premium principais.
- [ ] Riscos residuais foram corrigidos ou viraram plano datado e pequeno.
- [ ] Nao declarar READY se algum gate acima falhar.

## Proxima menor fatia correta

1. Corrigir `TooManyConnectionsError` no backend/Postgres Docker local.
2. Criar API clients TypeScript para:
   - notas;
   - inbox;
   - daily notes;
   - canvas;
   - MOCs/content maps;
   - trilhas;
   - agent builder sessions.
3. Substituir `PremiumModulePage` por telas reais, nesta ordem:
   - `NotesPage`;
   - `InboxPage`;
   - `DailyNotesPage`;
   - `CanvasPage`;
   - `MocsPage`;
   - `TrailsPage`;
   - `AgentBuilderPage`.
4. Adicionar teste que falha se essas paginas renderizarem o mesmo shell generico.
5. Revalidar Docker local com smoke visual e network sem 500.

## Regra permanente de comunicacao

Quando algo estiver apenas parcial, dizer "parcial".
Quando for placeholder, dizer "placeholder".
Quando estiver no ar mas nao pronto, dizer "local subiu, produto ainda nao esta pronto".
Nunca maquiar contrato backend como experiencia premium entregue.
