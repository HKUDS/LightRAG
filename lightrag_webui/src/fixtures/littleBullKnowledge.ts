import type { DocStatus, QueryMode } from '@/api/lightrag'

export const LITTLE_BULL_DEFAULT_WORKSPACE_ID = 'casa'

export type AreaPrivacy = 'solo' | 'familia' | 'equipe'
export type AreaStatus = 'ready' | 'processing' | 'needs_review'
export type Confidentiality = 'normal' | 'sensivel' | 'privado'
export type ActivityKind = 'document' | 'question' | 'area' | 'assistant' | 'alert'
export type ModelProfileId = 'rapido' | 'equilibrado' | 'inteligente' | 'privado'

export type AreaFixture = {
  id: string
  label: string
  emoji: string
  subtitle: string
  description: string
  privacy: AreaPrivacy
  status: AreaStatus
  documentCount: number
  readyCount: number
  processingCount: number
  lastUpdatedLabel: string
  accent: string
  tags: string[]
  suggestedQuestions: string[]
}

export type KnowledgeDocumentFixture = {
  id: string
  workspaceId: string
  fileName: string
  title: string
  filePath: string
  typeLabel: string
  status: DocStatus
  humanStatus: string
  sizeLabel: string
  uploadedAt: string
  updatedAtLabel: string
  ownerLabel: string
  confidentiality: Confidentiality
  tags: string[]
  summary: string
  suggestedActions: string[]
}

export type SourceCitationFixture = {
  id: string
  documentId: string
  title: string
  fileName: string
  excerpt: string
  confidence: 'alta' | 'media' | 'baixa'
}

export type ChatMessageFixture = {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: SourceCitationFixture[]
}

export type ChatThreadFixture = {
  id: string
  workspaceId: string
  title: string
  queryMode: QueryMode
  responseStyle: 'simples' | 'com_fontes' | 'checklist' | 'resumo'
  confidenceLabel: 'alta' | 'media' | 'baixa'
  updatedAtLabel: string
  messages: ChatMessageFixture[]
}

export type AssistantFixture = {
  id: string
  name: string
  tagline: string
  description: string
  workspaceIds: string[]
  enabled: boolean
  responseRules: string[]
  examplePrompts: string[]
  defaultProfile: ModelProfileId
}

export type ActivityFixture = {
  id: string
  workspaceId: string
  kind: ActivityKind
  title: string
  description: string
  createdAtLabel: string
  status: 'success' | 'info' | 'warning'
}

export type ModelProfileFixture = {
  id: ModelProfileId
  label: string
  description: string
  audienceLabel: string
  visibleToHomeUsers: boolean
}

export type InternalAgentFixture = {
  id: string
  displayName: string
  mission: string
  allowedSkills: string[]
  visibleToHomeUsers: boolean
  homeLabel: string
  riskLevel: 'low' | 'medium' | 'high'
}

export type InternalSubagentFixture = {
  id: string
  displayName: string
  description: string
  usedByAssistantIds: string[]
  visibleToHomeUsers: boolean
}

export type SkillActionFixture = {
  id: string
  label: string
  description: string
  inputKind: 'question' | 'document' | 'workspace' | 'model' | 'admin'
  requiresWorkspace: boolean
  isDestructive: boolean
  requiresHumanApproval: boolean
  homeVisible: boolean
}

export type ApprovalQueueFixture = {
  id: string
  actionId: string
  workspaceId: string
  title: string
  reason: string
  requestedBy: string
  createdAtLabel: string
  status: 'pending' | 'approved' | 'rejected'
}

export type WorkflowPlanFixture = {
  id: string
  name: string
  summary: string
  score: number
  selected: boolean
  tradeoffs: string[]
  validationLabel: string
}

export type CriticFindingFixture = {
  id: string
  rule: 'preserve_core' | 'dynamic_catalog' | 'human_approval' | 'prompt_injection'
  severity: 'info' | 'warning' | 'critical'
  message: string
  workspaceId: string
  relatedActionId: string
}

export type ModelCatalogSyncFixture = {
  provider: 'OpenRouter'
  status: 'synced' | 'stale' | 'failed'
  lastSyncLabel: string
  modelCount: number
  outputPath: string
  errorLabel: string | null
}

export type AuditTrailFixture = {
  id: string
  actor: string
  action: string
  workspaceId: string
  tenantId: string
  createdAtLabel: string
  metadataSummary: string
}

export type AdminFixture = {
  selectedProfile: ModelProfileId
  providerStatus: {
    label: string
    connected: boolean
    lastSyncLabel: string
  }
  monthlyUsage: {
    capLabel: string
    usedLabel: string
    percentUsed: number
  }
  privacyControls: Array<{
    id: string
    label: string
    description: string
    enabled: boolean
  }>
}

export type HomeSnapshotFixture = {
  activeWorkspaceId: string
  greeting: string
  headline: string
  summaryCards: Array<{
    label: string
    value: string
    helper: string
  }>
  quickActions: Array<{
    label: string
    description: string
    target: 'ask' | 'knowledge' | 'areas' | 'assistants'
  }>
}

export const areasFixture: AreaFixture[] = [
  {
    id: 'casa',
    label: 'Casa',
    emoji: '🏠',
    subtitle: 'Manuais, garantias, contas e contratos da casa.',
    description: 'Use para guardar tudo que ajuda nas decisões e rotinas da casa.',
    privacy: 'familia',
    status: 'ready',
    documentCount: 18,
    readyCount: 16,
    processingCount: 2,
    lastUpdatedLabel: 'Atualizada hoje',
    accent: '#FACC15',
    tags: ['Manual', 'Garantia', 'Conta', 'Contrato'],
    suggestedQuestions: [
      'Quando vence a garantia da geladeira?',
      'Quais contas da casa aparecem nos documentos?',
      'Resuma o contrato de aluguel em linguagem simples.'
    ]
  },
  {
    id: 'familia',
    label: 'Família',
    emoji: '👨‍👩‍👧',
    subtitle: 'Escola, saúde, viagens e documentos importantes.',
    description: 'Um espaço seguro para informações recorrentes da família.',
    privacy: 'familia',
    status: 'ready',
    documentCount: 12,
    readyCount: 12,
    processingCount: 0,
    lastUpdatedLabel: 'Atualizada ontem',
    accent: '#F97316',
    tags: ['Escola', 'Saúde', 'Viagem', 'Identidade'],
    suggestedQuestions: [
      'Quais vacinas aparecem nos documentos?',
      'Qual é o checklist da próxima viagem?',
      'O que preciso lembrar sobre a escola?'
    ]
  },
  {
    id: 'financas',
    label: 'Finanças',
    emoji: '💳',
    subtitle: 'Recibos, impostos, investimentos e vencimentos.',
    description: 'Organize comprovantes e peça resumos sem abrir planilhas.',
    privacy: 'solo',
    status: 'needs_review',
    documentCount: 23,
    readyCount: 20,
    processingCount: 1,
    lastUpdatedLabel: 'Atualizada há 2 dias',
    accent: '#22C55E',
    tags: ['Recibo', 'Imposto', 'Investimento', 'Cartão'],
    suggestedQuestions: [
      'Quais recibos ainda preciso revisar?',
      'Liste despesas recorrentes encontradas.',
      'Monte um resumo de impostos deste ano.'
    ]
  },
  {
    id: 'trabalho',
    label: 'Trabalho',
    emoji: '💼',
    subtitle: 'Propostas, clientes, projetos e reuniões.',
    description: 'Memória leve para tarefas de trabalho e pequeno negócio.',
    privacy: 'equipe',
    status: 'ready',
    documentCount: 34,
    readyCount: 31,
    processingCount: 3,
    lastUpdatedLabel: 'Atualizada hoje',
    accent: '#2563EB',
    tags: ['Cliente', 'Proposta', 'Projeto', 'Reunião'],
    suggestedQuestions: [
      'Quais propostas estão esperando retorno?',
      'Resuma a última reunião com cliente.',
      'Crie um checklist do projeto em aberto.'
    ]
  },
  {
    id: 'estudos',
    label: 'Estudos',
    emoji: '📚',
    subtitle: 'PDFs, cursos, resumos e anotações.',
    description: 'Transforme conteúdo de estudo em perguntas e revisões rápidas.',
    privacy: 'solo',
    status: 'ready',
    documentCount: 15,
    readyCount: 15,
    processingCount: 0,
    lastUpdatedLabel: 'Atualizada semana passada',
    accent: '#7C3AED',
    tags: ['Curso', 'Resumo', 'Artigo', 'Aula'],
    suggestedQuestions: [
      'Explique este conceito de forma simples.',
      'Crie 10 perguntas para revisão.',
      'Monte um resumo do material da semana.'
    ]
  },
  {
    id: 'pequeno_negocio',
    label: 'Pequeno negócio',
    emoji: '🛒',
    subtitle: 'Produtos, atendimento, pedidos e perguntas frequentes.',
    description: 'Uma base simples para responder clientes com mais consistência.',
    privacy: 'equipe',
    status: 'processing',
    documentCount: 27,
    readyCount: 19,
    processingCount: 8,
    lastUpdatedLabel: 'Atualizando agora',
    accent: '#0EA5E9',
    tags: ['Produto', 'FAQ', 'Pedido', 'Cliente'],
    suggestedQuestions: [
      'Quais são as perguntas mais comuns dos clientes?',
      'Crie uma resposta simples sobre prazo de entrega.',
      'Liste produtos com política de troca especial.'
    ]
  }
]

export const documentsFixture: KnowledgeDocumentFixture[] = [
  {
    id: 'doc_manual_geladeira',
    workspaceId: 'casa',
    fileName: 'manual-geladeira-brastemp.pdf',
    title: 'Manual da geladeira Brastemp',
    filePath: '/Casa/Eletrodomésticos/manual-geladeira-brastemp.pdf',
    typeLabel: 'Manual',
    status: 'processed',
    humanStatus: 'Pronto',
    sizeLabel: '3,8 MB',
    uploadedAt: '2026-04-21',
    updatedAtLabel: 'hoje',
    ownerLabel: 'João',
    confidentiality: 'normal',
    tags: ['Manual', 'Garantia', 'Cozinha'],
    summary: 'Instruções de uso, garantia, limpeza e códigos de erro da geladeira.',
    suggestedActions: ['Perguntar', 'Criar checklist', 'Marcar favorito']
  },
  {
    id: 'doc_nota_geladeira',
    workspaceId: 'casa',
    fileName: 'nota-fiscal-geladeira.pdf',
    title: 'Nota fiscal da geladeira',
    filePath: '/Casa/Notas fiscais/nota-fiscal-geladeira.pdf',
    typeLabel: 'Nota fiscal',
    status: 'processed',
    humanStatus: 'Pronto',
    sizeLabel: '420 KB',
    uploadedAt: '2026-04-21',
    updatedAtLabel: 'hoje',
    ownerLabel: 'João',
    confidentiality: 'sensivel',
    tags: ['Nota fiscal', 'Garantia', 'Compra'],
    summary: 'Compra realizada em 12/03/2026 com garantia legal e garantia estendida.',
    suggestedActions: ['Perguntar', 'Extrair vencimento', 'Mover']
  },
  {
    id: 'doc_contrato_aluguel',
    workspaceId: 'casa',
    fileName: 'contrato-aluguel-2026.pdf',
    title: 'Contrato de aluguel 2026',
    filePath: '/Casa/Contratos/contrato-aluguel-2026.pdf',
    typeLabel: 'Contrato',
    status: 'processed',
    humanStatus: 'Pronto',
    sizeLabel: '1,2 MB',
    uploadedAt: '2026-04-19',
    updatedAtLabel: 'ontem',
    ownerLabel: 'João',
    confidentiality: 'privado',
    tags: ['Contrato', 'Aluguel', 'Moradia'],
    summary: 'Contrato residencial com cláusulas de reajuste, multa e vistoria.',
    suggestedActions: ['Resumir', 'Explicar multa', 'Gerar checklist']
  },
  {
    id: 'doc_recibos_2026',
    workspaceId: 'financas',
    fileName: 'recibos-medicos-2026.zip',
    title: 'Recibos médicos 2026',
    filePath: '/Finanças/Impostos/recibos-medicos-2026.zip',
    typeLabel: 'Recibos',
    status: 'processing',
    humanStatus: 'Lendo',
    sizeLabel: '9,4 MB',
    uploadedAt: '2026-04-24',
    updatedAtLabel: 'agora',
    ownerLabel: 'João',
    confidentiality: 'sensivel',
    tags: ['Imposto', 'Saúde', 'Recibo'],
    summary: 'Pacote de recibos para conferência de despesas dedutíveis.',
    suggestedActions: ['Aguardar leitura', 'Revisar depois']
  },
  {
    id: 'doc_escola_calendario',
    workspaceId: 'familia',
    fileName: 'calendario-escolar-2026.pdf',
    title: 'Calendário escolar 2026',
    filePath: '/Família/Escola/calendario-escolar-2026.pdf',
    typeLabel: 'Calendário',
    status: 'processed',
    humanStatus: 'Pronto',
    sizeLabel: '860 KB',
    uploadedAt: '2026-04-18',
    updatedAtLabel: 'há 3 dias',
    ownerLabel: 'Marina',
    confidentiality: 'normal',
    tags: ['Escola', 'Datas', 'Família'],
    summary: 'Datas de provas, reuniões, férias e eventos escolares.',
    suggestedActions: ['Criar lembretes', 'Perguntar', 'Resumir mês']
  },
  {
    id: 'doc_reuniao_cliente',
    workspaceId: 'trabalho',
    fileName: 'reuniao-cliente-aurora.md',
    title: 'Reunião com Cliente Aurora',
    filePath: '/Trabalho/Clientes/Aurora/reuniao-cliente-aurora.md',
    typeLabel: 'Anotação',
    status: 'processed',
    humanStatus: 'Pronto',
    sizeLabel: '36 KB',
    uploadedAt: '2026-04-24',
    updatedAtLabel: 'hoje',
    ownerLabel: 'João',
    confidentiality: 'normal',
    tags: ['Cliente', 'Reunião', 'Proposta'],
    summary: 'Notas da reunião com escopo, objeções, próximos passos e responsáveis.',
    suggestedActions: ['Criar follow-up', 'Gerar resumo', 'Perguntar']
  },
  {
    id: 'doc_politica_trocas',
    workspaceId: 'pequeno_negocio',
    fileName: 'politica-de-trocas.md',
    title: 'Política de trocas',
    filePath: '/Pequeno negócio/Atendimento/politica-de-trocas.md',
    typeLabel: 'FAQ',
    status: 'preprocessed',
    humanStatus: 'Organizando',
    sizeLabel: '18 KB',
    uploadedAt: '2026-04-25',
    updatedAtLabel: 'agora',
    ownerLabel: 'Equipe',
    confidentiality: 'normal',
    tags: ['Atendimento', 'Troca', 'Cliente'],
    summary: 'Regras de troca, prazo de devolução e exceções por categoria.',
    suggestedActions: ['Aguardar leitura', 'Revisar linguagem']
  },
  {
    id: 'doc_modulo_estudos',
    workspaceId: 'estudos',
    fileName: 'fundamentos-de-ia.pdf',
    title: 'Fundamentos de IA',
    filePath: '/Estudos/Cursos/fundamentos-de-ia.pdf',
    typeLabel: 'Material de estudo',
    status: 'processed',
    humanStatus: 'Pronto',
    sizeLabel: '5,1 MB',
    uploadedAt: '2026-04-11',
    updatedAtLabel: 'semana passada',
    ownerLabel: 'João',
    confidentiality: 'normal',
    tags: ['Curso', 'IA', 'Resumo'],
    summary: 'Material introdutório sobre agentes, prompts, modelos e avaliação.',
    suggestedActions: ['Criar quiz', 'Resumir', 'Explicar simples']
  }
]

export const chatThreadsFixture: ChatThreadFixture[] = [
  {
    id: 'chat_garantia_geladeira',
    workspaceId: 'casa',
    title: 'Garantia da geladeira',
    queryMode: 'hybrid',
    responseStyle: 'com_fontes',
    confidenceLabel: 'alta',
    updatedAtLabel: 'há 12 minutos',
    messages: [
      {
        id: 'chat_garantia_geladeira_user',
        role: 'user',
        content: 'Quando vence a garantia da geladeira?'
      },
      {
        id: 'chat_garantia_geladeira_assistant',
        role: 'assistant',
        content:
          'A garantia principal parece vencer em 12/03/2027. Há também uma garantia estendida citada na nota fiscal; vale confirmar a cobertura exata antes de acionar assistência.',
        citations: [
          {
            id: 'src_nota_geladeira',
            documentId: 'doc_nota_geladeira',
            title: 'Nota fiscal da geladeira',
            fileName: 'nota-fiscal-geladeira.pdf',
            excerpt: 'Compra realizada em 12/03/2026 com garantia estendida registrada no comprovante.',
            confidence: 'alta'
          },
          {
            id: 'src_manual_geladeira',
            documentId: 'doc_manual_geladeira',
            title: 'Manual da geladeira Brastemp',
            fileName: 'manual-geladeira-brastemp.pdf',
            excerpt: 'O manual orienta acionar assistência autorizada com nota fiscal e número de série.',
            confidence: 'alta'
          }
        ]
      }
    ]
  },
  {
    id: 'chat_multa_aluguel',
    workspaceId: 'casa',
    title: 'Multa do contrato de aluguel',
    queryMode: 'mix',
    responseStyle: 'simples',
    confidenceLabel: 'media',
    updatedAtLabel: 'ontem',
    messages: [
      {
        id: 'chat_multa_aluguel_user',
        role: 'user',
        content: 'Explique a multa do contrato de aluguel em linguagem simples.'
      },
      {
        id: 'chat_multa_aluguel_assistant',
        role: 'assistant',
        content:
          'O contrato indica multa proporcional se houver saída antes do prazo. A regra parece depender do tempo restante de contrato, então a recomendação segura é calcular antes de decidir.',
        citations: [
          {
            id: 'src_contrato_aluguel',
            documentId: 'doc_contrato_aluguel',
            title: 'Contrato de aluguel 2026',
            fileName: 'contrato-aluguel-2026.pdf',
            excerpt: 'Cláusula de rescisão prevê multa proporcional ao prazo restante.',
            confidence: 'media'
          }
        ]
      }
    ]
  },
  {
    id: 'chat_followup_aurora',
    workspaceId: 'trabalho',
    title: 'Follow-up Cliente Aurora',
    queryMode: 'hybrid',
    responseStyle: 'checklist',
    confidenceLabel: 'alta',
    updatedAtLabel: 'hoje',
    messages: [
      {
        id: 'chat_followup_aurora_user',
        role: 'user',
        content: 'Crie um checklist de próximos passos para o Cliente Aurora.'
      },
      {
        id: 'chat_followup_aurora_assistant',
        role: 'assistant',
        content:
          'Checklist sugerido: enviar proposta revisada, confirmar prazo de implantação, alinhar responsável técnico, responder objeção sobre suporte e marcar reunião de decisão.',
        citations: [
          {
            id: 'src_reuniao_cliente',
            documentId: 'doc_reuniao_cliente',
            title: 'Reunião com Cliente Aurora',
            fileName: 'reuniao-cliente-aurora.md',
            excerpt: 'Cliente pediu proposta revisada e confirmação de prazo antes da aprovação.',
            confidence: 'alta'
          }
        ]
      }
    ]
  }
]

export const assistantsFixture: AssistantFixture[] = [
  {
    id: 'organizador_casa',
    name: 'Organizador da Casa',
    tagline: 'Encontra manuais, garantias, contas e contratos.',
    description: 'Ajuda a transformar documentos da casa em respostas simples e checklists úteis.',
    workspaceIds: ['casa'],
    enabled: true,
    responseRules: ['Responder em linguagem simples', 'Mostrar fontes usadas', 'Sugerir próximos passos'],
    examplePrompts: [
      'Quando vence essa garantia?',
      'Crie um checklist de manutenção.',
      'Onde está a informação sobre assistência?'
    ],
    defaultProfile: 'equilibrado'
  },
  {
    id: 'ajudante_financeiro',
    name: 'Ajudante Financeiro',
    tagline: 'Resume recibos, vencimentos e despesas recorrentes.',
    description: 'Organiza dados financeiros sem prometer aconselhamento financeiro.',
    workspaceIds: ['financas'],
    enabled: true,
    responseRules: ['Não dar recomendação financeira definitiva', 'Marcar incertezas', 'Citar documentos'],
    examplePrompts: [
      'Quais recibos preciso revisar?',
      'Liste despesas recorrentes.',
      'Monte um resumo para imposto.'
    ],
    defaultProfile: 'privado'
  },
  {
    id: 'leitor_contratos',
    name: 'Leitor de Contratos',
    tagline: 'Explica cláusulas difíceis em português claro.',
    description: 'Ajuda a entender contratos, destacando riscos e pontos que merecem revisão humana.',
    workspaceIds: ['casa', 'trabalho', 'pequeno_negocio'],
    enabled: true,
    responseRules: ['Não substituir advogado', 'Destacar riscos', 'Sempre citar cláusulas/fontes'],
    examplePrompts: [
      'Explique esta multa em linguagem simples.',
      'Quais pontos parecem mais importantes?',
      'O que devo perguntar antes de assinar?'
    ],
    defaultProfile: 'inteligente'
  },
  {
    id: 'assistente_estudos',
    name: 'Assistente de Estudos',
    tagline: 'Cria resumos, quizzes e revisões rápidas.',
    description: 'Transforma PDFs e anotações em materiais de estudo fáceis de revisar.',
    workspaceIds: ['estudos'],
    enabled: true,
    responseRules: ['Explicar por etapas', 'Criar exemplos', 'Gerar perguntas de revisão'],
    examplePrompts: [
      'Explique como se eu fosse iniciante.',
      'Crie 10 perguntas de prova.',
      'Resuma este capítulo.'
    ],
    defaultProfile: 'equilibrado'
  },
  {
    id: 'atendente_negocio',
    name: 'Atendente do Pequeno Negócio',
    tagline: 'Ajuda a responder clientes com consistência.',
    description: 'Consulta políticas, produtos e FAQs para sugerir respostas claras.',
    workspaceIds: ['pequeno_negocio'],
    enabled: false,
    responseRules: ['Não prometer exceções', 'Usar tom cordial', 'Pedir revisão humana em casos sensíveis'],
    examplePrompts: [
      'Responda sobre prazo de troca.',
      'Explique a política de entrega.',
      'Crie uma resposta curta para WhatsApp.'
    ],
    defaultProfile: 'rapido'
  }
]

export const internalAgentsFixture: InternalAgentFixture[] = [
  {
    id: 'orchestrator_agent',
    displayName: 'Coordenador',
    mission: 'Escolhe a melhor rota segura entre pergunta, busca, ação e auditoria.',
    allowedSkills: ['query_knowledge', 'route_model_by_policy', 'audit_action'],
    visibleToHomeUsers: false,
    homeLabel: 'Organiza o trabalho por trás da tela',
    riskLevel: 'medium'
  },
  {
    id: 'planner_agent',
    displayName: 'Planejador',
    mission: 'Compara planos curtos e registra apenas a decisão final, sem expor raciocínio bruto.',
    allowedSkills: ['route_model_by_policy', 'validate_json_output', 'audit_action'],
    visibleToHomeUsers: false,
    homeLabel: 'Planeja antes de agir',
    riskLevel: 'medium'
  },
  {
    id: 'retrieval_agent',
    displayName: 'Buscador de conhecimento',
    mission: 'Consulta o motor de memória e recuperação com fontes.',
    allowedSkills: [
      'query_knowledge',
      'query_knowledge_context_only',
      'ingest_document',
      'ingest_batch'
    ],
    visibleToHomeUsers: false,
    homeLabel: 'Encontra respostas nos documentos',
    riskLevel: 'low'
  },
  {
    id: 'model_router_agent',
    displayName: 'Roteador de modelos',
    mission: 'Seleciona modelos disponíveis e permitidos por perfil, custo e privacidade.',
    allowedSkills: ['sync_model_catalog', 'get_model_catalog', 'route_model_by_policy'],
    visibleToHomeUsers: false,
    homeLabel: 'Escolhe o modo Rápido, Equilibrado, Inteligente ou Privado',
    riskLevel: 'medium'
  },
  {
    id: 'compliance_audit_agent',
    displayName: 'Guardião de segurança',
    mission: 'Verifica permissões, privacidade, retenção e trilha de auditoria.',
    allowedSkills: ['audit_action', 'check_cost_policy'],
    visibleToHomeUsers: false,
    homeLabel: 'Protege ações sensíveis',
    riskLevel: 'high'
  },
  {
    id: 'critic_evaluator_agent',
    displayName: 'Crítico de qualidade',
    mission: 'Procura lacunas, promessas sem fonte e riscos antes da resposta final.',
    allowedSkills: ['validate_json_output', 'audit_action'],
    visibleToHomeUsers: false,
    homeLabel: 'Confere se a resposta faz sentido',
    riskLevel: 'medium'
  }
]

export const internalSubagentsFixture: InternalSubagentFixture[] = [
  {
    id: 'doc_qa_subagent',
    displayName: 'Leitor de documentos',
    description: 'Responde perguntas usando apenas documentos e fontes da área atual.',
    usedByAssistantIds: ['organizador_casa', 'leitor_contratos', 'assistente_estudos'],
    visibleToHomeUsers: false
  },
  {
    id: 'summarizer_subagent',
    displayName: 'Resumidor',
    description: 'Transforma documentos longos em resumos, listas e checklists.',
    usedByAssistantIds: ['organizador_casa', 'assistente_estudos', 'atendente_negocio'],
    visibleToHomeUsers: false
  },
  {
    id: 'prompt_injection_guard_subagent',
    displayName: 'Filtro de instruções perigosas',
    description: 'Bloqueia pedidos que tentem ignorar regras, fontes ou permissões.',
    usedByAssistantIds: [
      'organizador_casa',
      'ajudante_financeiro',
      'leitor_contratos',
      'atendente_negocio'
    ],
    visibleToHomeUsers: false
  },
  {
    id: 'escalation_subagent',
    displayName: 'Encaminhador humano',
    description: 'Marca quando uma decisão precisa de revisão humana antes de continuar.',
    usedByAssistantIds: ['ajudante_financeiro', 'leitor_contratos', 'atendente_negocio'],
    visibleToHomeUsers: false
  }
]

export const skillActionsFixture: SkillActionFixture[] = [
  {
    id: 'query_knowledge',
    label: 'Perguntar aos documentos',
    description: 'Busca resposta na área selecionada e mostra fontes usadas.',
    inputKind: 'question',
    requiresWorkspace: true,
    isDestructive: false,
    requiresHumanApproval: false,
    homeVisible: true
  },
  {
    id: 'query_knowledge_context_only',
    label: 'Ver contexto encontrado',
    description: 'Mostra somente os trechos encontrados, sem gerar resposta final.',
    inputKind: 'question',
    requiresWorkspace: true,
    isDestructive: false,
    requiresHumanApproval: false,
    homeVisible: false
  },
  {
    id: 'ingest_document',
    label: 'Adicionar documento',
    description: 'Envia um arquivo para leitura, organização e uso em perguntas.',
    inputKind: 'document',
    requiresWorkspace: true,
    isDestructive: false,
    requiresHumanApproval: false,
    homeVisible: true
  },
  {
    id: 'ingest_batch',
    label: 'Adicionar vários documentos',
    description: 'Envia uma pasta ou lote de documentos para a área selecionada.',
    inputKind: 'document',
    requiresWorkspace: true,
    isDestructive: false,
    requiresHumanApproval: false,
    homeVisible: true
  },
  {
    id: 'reindex_workspace',
    label: 'Reorganizar área',
    description: 'Reconstrói a memória da área quando documentos mudaram muito.',
    inputKind: 'workspace',
    requiresWorkspace: true,
    isDestructive: false,
    requiresHumanApproval: true,
    homeVisible: false
  },
  {
    id: 'delete_document_by_id',
    label: 'Excluir documento',
    description: 'Remove um documento e seus dados derivados da área atual.',
    inputKind: 'document',
    requiresWorkspace: true,
    isDestructive: true,
    requiresHumanApproval: true,
    homeVisible: false
  },
  {
    id: 'merge_entities',
    label: 'Unir nomes duplicados',
    description: 'Junta entidades semelhantes, como nomes escritos de formas diferentes.',
    inputKind: 'admin',
    requiresWorkspace: true,
    isDestructive: false,
    requiresHumanApproval: true,
    homeVisible: false
  },
  {
    id: 'sync_model_catalog',
    label: 'Atualizar modelos',
    description: 'Sincroniza a lista de modelos disponíveis para a conta.',
    inputKind: 'model',
    requiresWorkspace: false,
    isDestructive: false,
    requiresHumanApproval: false,
    homeVisible: false
  },
  {
    id: 'route_model_by_policy',
    label: 'Escolher modo de resposta',
    description: 'Seleciona Rápido, Equilibrado, Mais inteligente ou Privado/local.',
    inputKind: 'model',
    requiresWorkspace: true,
    isDestructive: false,
    requiresHumanApproval: false,
    homeVisible: false
  },
  {
    id: 'audit_action',
    label: 'Registrar atividade',
    description: 'Grava uma trilha simples do que aconteceu no sistema.',
    inputKind: 'admin',
    requiresWorkspace: false,
    isDestructive: false,
    requiresHumanApproval: false,
    homeVisible: false
  }
]

export const approvalQueueFixture: ApprovalQueueFixture[] = [
  {
    id: 'approval_delete_contract',
    actionId: 'delete_document_by_id',
    workspaceId: 'casa',
    title: 'Confirmar exclusão do contrato de aluguel',
    reason: 'Documento privado com informações importantes. Excluir remove fontes das respostas.',
    requestedBy: 'João',
    createdAtLabel: 'há 4 minutos',
    status: 'pending'
  },
  {
    id: 'approval_reindex_financas',
    actionId: 'reindex_workspace',
    workspaceId: 'financas',
    title: 'Reorganizar área Finanças',
    reason: 'Novos recibos foram adicionados e alguns arquivos ainda precisam ser relidos.',
    requestedBy: 'Ajudante Financeiro',
    createdAtLabel: 'hoje',
    status: 'pending'
  },
  {
    id: 'approval_merge_aurora',
    actionId: 'merge_entities',
    workspaceId: 'trabalho',
    title: 'Unir Cliente Aurora e Aurora Ltda.',
    reason: 'O sistema encontrou dois nomes que parecem representar o mesmo cliente.',
    requestedBy: 'Coordenador',
    createdAtLabel: 'ontem',
    status: 'approved'
  }
]

export const workflowPlansFixture: WorkflowPlanFixture[] = [
  {
    id: 'plan_answer_with_sources',
    name: 'Responder com fontes da área atual',
    summary: 'Buscar trechos relevantes, gerar resposta curta e mostrar fontes usadas.',
    score: 92,
    selected: true,
    tradeoffs: ['Mais confiável para usuário leigo', 'Pode ser um pouco mais lento'],
    validationLabel: 'Seguro para perguntas comuns'
  },
  {
    id: 'plan_context_only',
    name: 'Mostrar apenas trechos encontrados',
    summary: 'Não gerar resposta; mostrar os documentos que parecem responder a pergunta.',
    score: 81,
    selected: false,
    tradeoffs: ['Mais transparente', 'Exige mais leitura do usuário'],
    validationLabel: 'Bom para revisão humana'
  },
  {
    id: 'plan_premium_reasoning',
    name: 'Usar modo Mais inteligente',
    summary: 'Escalar para raciocínio mais forte quando houver contrato, dinheiro ou risco.',
    score: 76,
    selected: false,
    tradeoffs: ['Melhor para casos difíceis', 'Custo e latência maiores'],
    validationLabel: 'Usar só com necessidade clara'
  }
]

export const criticFindingsFixture: CriticFindingFixture[] = [
  {
    id: 'finding_human_delete',
    rule: 'human_approval',
    severity: 'warning',
    message: 'Excluir contrato exige confirmação porque a ação remove fontes futuras.',
    workspaceId: 'casa',
    relatedActionId: 'delete_document_by_id'
  },
  {
    id: 'finding_dynamic_catalog',
    rule: 'dynamic_catalog',
    severity: 'info',
    message: 'Modelos devem vir do catálogo sincronizado da conta, não de lista fixa.',
    workspaceId: 'casa',
    relatedActionId: 'sync_model_catalog'
  },
  {
    id: 'finding_prompt_injection',
    rule: 'prompt_injection',
    severity: 'critical',
    message: 'Um documento tentou instruir o assistente a ignorar fontes e políticas.',
    workspaceId: 'pequeno_negocio',
    relatedActionId: 'query_knowledge'
  }
]

export const modelCatalogSyncFixture: ModelCatalogSyncFixture = {
  provider: 'OpenRouter',
  status: 'synced',
  lastSyncLabel: 'atualizado hoje',
  modelCount: 355,
  outputPath: 'rag_storage/model_catalog/openrouter_catalog.json',
  errorLabel: null
}

export const auditTrailFixture: AuditTrailFixture[] = [
  {
    id: 'audit_query_garantia',
    actor: 'Organizador da Casa',
    action: 'query_knowledge',
    workspaceId: 'casa',
    tenantId: 'little_bull_home_demo',
    createdAtLabel: 'há 12 minutos',
    metadataSummary: 'Pergunta respondida com 2 fontes e confiança alta.'
  },
  {
    id: 'audit_route_model',
    actor: 'Roteador de modelos',
    action: 'route_model_by_policy',
    workspaceId: 'casa',
    tenantId: 'little_bull_home_demo',
    createdAtLabel: 'há 12 minutos',
    metadataSummary: 'Perfil Equilibrado selecionado por custo e qualidade.'
  },
  {
    id: 'audit_ingest_manual',
    actor: 'Buscador de conhecimento',
    action: 'ingest_document',
    workspaceId: 'casa',
    tenantId: 'little_bull_home_demo',
    createdAtLabel: 'há 8 minutos',
    metadataSummary: 'manual-geladeira-brastemp.pdf ficou pronto para perguntas.'
  },
  {
    id: 'audit_guardrail_delete',
    actor: 'Guardião de segurança',
    action: 'human_approval_required',
    workspaceId: 'casa',
    tenantId: 'little_bull_home_demo',
    createdAtLabel: 'há 4 minutos',
    metadataSummary: 'Exclusão de documento privado enviada para fila de aprovação.'
  }
]

export const activitiesFixture: ActivityFixture[] = [
  {
    id: 'act_manual_ready',
    workspaceId: 'casa',
    kind: 'document',
    title: 'Manual da geladeira ficou pronto',
    description: 'O arquivo já pode ser usado em perguntas da área Casa.',
    createdAtLabel: 'há 8 minutos',
    status: 'success'
  },
  {
    id: 'act_question_garantia',
    workspaceId: 'casa',
    kind: 'question',
    title: 'Resposta criada sobre garantia',
    description: 'A resposta usou 2 fontes e teve confiança alta.',
    createdAtLabel: 'há 12 minutos',
    status: 'success'
  },
  {
    id: 'act_recibos_processing',
    workspaceId: 'financas',
    kind: 'document',
    title: 'Recibos médicos estão sendo lidos',
    description: 'Alguns arquivos do pacote ainda estão em processamento.',
    createdAtLabel: 'agora',
    status: 'info'
  },
  {
    id: 'act_area_created',
    workspaceId: 'pequeno_negocio',
    kind: 'area',
    title: 'Área Pequeno negócio criada',
    description: 'A área foi preparada para documentos de atendimento e produtos.',
    createdAtLabel: 'hoje',
    status: 'success'
  },
  {
    id: 'act_contract_review',
    workspaceId: 'casa',
    kind: 'alert',
    title: 'Contrato de aluguel merece revisão',
    description: 'O documento contém cláusulas marcadas como sensíveis.',
    createdAtLabel: 'ontem',
    status: 'warning'
  },
  {
    id: 'act_assistant_used',
    workspaceId: 'trabalho',
    kind: 'assistant',
    title: 'Memória do Trabalho usada',
    description: 'Foi criado um checklist para o Cliente Aurora.',
    createdAtLabel: 'hoje',
    status: 'success'
  }
]

export const modelProfilesFixture: ModelProfileFixture[] = [
  {
    id: 'rapido',
    label: 'Rápido',
    description: 'Bom para respostas simples, resumos curtos e alto volume.',
    audienceLabel: 'Dia a dia',
    visibleToHomeUsers: true
  },
  {
    id: 'equilibrado',
    label: 'Equilibrado',
    description: 'Melhor padrão para qualidade, custo e velocidade.',
    audienceLabel: 'Recomendado',
    visibleToHomeUsers: true
  },
  {
    id: 'inteligente',
    label: 'Mais inteligente',
    description: 'Use quando o assunto exigir mais cuidado ou raciocínio.',
    audienceLabel: 'Casos difíceis',
    visibleToHomeUsers: true
  },
  {
    id: 'privado',
    label: 'Privado/local',
    description: 'Preferido para dados sensíveis quando houver modelo local disponível.',
    audienceLabel: 'Privacidade',
    visibleToHomeUsers: true
  }
]

export const adminFixture: AdminFixture = {
  selectedProfile: 'equilibrado',
  providerStatus: {
    label: 'OpenRouter pronto',
    connected: true,
    lastSyncLabel: 'Catálogo atualizado hoje'
  },
  monthlyUsage: {
    capLabel: 'R$ 150/mês',
    usedLabel: 'R$ 37 usados',
    percentUsed: 25
  },
  privacyControls: [
    {
      id: 'block_sensitive_hosted',
      label: 'Proteger dados sensíveis',
      description: 'Impede envio de documentos privados para modelos hospedados.',
      enabled: true
    },
    {
      id: 'require_sources',
      label: 'Sempre mostrar fontes',
      description: 'Respostas importantes devem indicar de onde vieram.',
      enabled: true
    },
    {
      id: 'human_confirm_delete',
      label: 'Confirmar exclusões',
      description: 'Ações destrutivas exigem confirmação humana.',
      enabled: true
    }
  ]
}

export const homeSnapshotFixture: HomeSnapshotFixture = {
  activeWorkspaceId: LITTLE_BULL_DEFAULT_WORKSPACE_ID,
  greeting: 'Bom dia.',
  headline: 'Onde quer buscar conhecimento hoje?',
  summaryCards: [
    {
      label: 'Áreas',
      value: String(areasFixture.length),
      helper: 'espaços de conhecimento'
    },
    {
      label: 'Documentos',
      value: String(documentsFixture.length),
      helper: 'arquivos de exemplo'
    },
    {
      label: 'Prontos',
      value: String(documentsFixture.filter((document) => document.status === 'processed').length),
      helper: 'já podem responder'
    },
    {
      label: 'Assistentes',
      value: String(assistantsFixture.filter((assistant) => assistant.enabled).length),
      helper: 'ativos agora'
    }
  ],
  quickActions: [
    {
      label: 'Fazer uma pergunta',
      description: 'Pergunte em português e receba fontes.',
      target: 'ask'
    },
    {
      label: 'Adicionar arquivos',
      description: 'Envie PDFs, notas e documentos.',
      target: 'knowledge'
    },
    {
      label: 'Criar uma área',
      description: 'Separe casa, família, finanças e trabalho.',
      target: 'areas'
    },
    {
      label: 'Escolher assistente',
      description: 'Use ajudantes prontos para cada rotina.',
      target: 'assistants'
    }
  ]
}

export const littleBullKnowledgeFixtures = {
  areas: areasFixture,
  documents: documentsFixture,
  chats: chatThreadsFixture,
  assistants: assistantsFixture,
  internalAgents: internalAgentsFixture,
  internalSubagents: internalSubagentsFixture,
  skillActions: skillActionsFixture,
  approvals: approvalQueueFixture,
  workflowPlans: workflowPlansFixture,
  criticFindings: criticFindingsFixture,
  activities: activitiesFixture,
  modelProfiles: modelProfilesFixture,
  modelCatalogSync: modelCatalogSyncFixture,
  auditTrail: auditTrailFixture,
  admin: adminFixture,
  home: homeSnapshotFixture
}

export function getAreaById(areaId: string): AreaFixture | undefined {
  return areasFixture.find((area) => area.id === areaId)
}

export function getDocumentsByWorkspace(workspaceId: string): KnowledgeDocumentFixture[] {
  return documentsFixture.filter((document) => document.workspaceId === workspaceId)
}

export function getChatsByWorkspace(workspaceId: string): ChatThreadFixture[] {
  return chatThreadsFixture.filter((thread) => thread.workspaceId === workspaceId)
}

export function getActivitiesByWorkspace(workspaceId: string): ActivityFixture[] {
  return activitiesFixture.filter((activity) => activity.workspaceId === workspaceId)
}

export function getAssistantsByWorkspace(workspaceId: string): AssistantFixture[] {
  return assistantsFixture.filter((assistant) => assistant.workspaceIds.includes(workspaceId))
}

export function getSkillActionById(actionId: string): SkillActionFixture | undefined {
  return skillActionsFixture.find((action) => action.id === actionId)
}

export function getPendingApprovalsByWorkspace(workspaceId: string): ApprovalQueueFixture[] {
  return approvalQueueFixture.filter(
    (approval) => approval.workspaceId === workspaceId && approval.status === 'pending'
  )
}

export function getAuditTrailByWorkspace(workspaceId: string): AuditTrailFixture[] {
  return auditTrailFixture.filter((event) => event.workspaceId === workspaceId)
}

export function getWorkspaceSuggestedQuestions(workspaceId: string): string[] {
  const area = getAreaById(workspaceId)
  return area?.suggestedQuestions ?? []
}

export function toWorkspaceHeader(workspaceId: string): string {
  return workspaceId.replace(/[^a-zA-Z0-9_]/g, '_')
}
