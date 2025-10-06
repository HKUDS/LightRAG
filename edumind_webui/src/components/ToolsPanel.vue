<template>
  <v-card class="tool-shell" elevation="2">
    <v-tabs v-model="activeTab" class="tool-shell__tabs" color="primary" grow>
      <v-tab value="files" class="tool-shell__tab">
        <span>Files</span>
      </v-tab>
      <v-tab value="search" class="tool-shell__tab">
        <span>Search with AI</span>
      </v-tab>
      <v-tab value="creative" class="tool-shell__tab">
        <span>Creative Toolkit</span>
      </v-tab>
    </v-tabs>

    <v-window v-model="activeTab" class="tool-shell__window">
      <v-window-item value="files" class="tool-shell__pane">
        <div class="files-pane">
          <div class="files-pane__actions">
            <v-menu v-model="menuOpen" location="bottom" :offset="8" transition="fade-transition">
              <template #activator="{ props }">
                <v-btn color="primary" variant="flat" prepend-icon="mdi-plus" v-bind="props">
                  Add Content
                </v-btn>
              </template>
              <v-list>
                <v-list-item @click="handleOpenUpload">
                  <v-list-item-title>Upload Files</v-list-item-title>
                </v-list-item>
                <v-list-item @click="handleOpenLinks">
                  <v-list-item-title>Add Webpage Links</v-list-item-title>
                </v-list-item>
              </v-list>
            </v-menu>
            <v-btn
              variant="text"
              color="primary"
              prepend-icon="mdi-refresh"
              :loading="documentsLoading"
              @click="refreshDocuments"
            >
              Refresh
            </v-btn>
          </div>

          <div v-if="documentsLoading" class="files-pane__state">
            <v-progress-circular indeterminate color="primary" size="32" />
            <p class="text-body-2 text-medium-emphasis mt-3 mb-0">Loading documents…</p>
          </div>

          <div v-else-if="documentsError" class="files-pane__state">
            <v-icon size="32" color="primary">mdi-alert-circle-outline</v-icon>
            <p class="text-body-2 text-medium-emphasis mt-3 mb-2">Failed to load documents.</p>
            <v-btn color="primary" variant="flat" @click="refreshDocuments">Try Again</v-btn>
          </div>

          <div v-else-if="!hasDocuments" class="files-pane__state">
            <v-icon size="32" color="primary">mdi-folder-outline</v-icon>
            <p class="text-body-2 text-medium-emphasis mt-3 mb-0">
              No documents available yet. Upload files or add links to begin.
            </p>
          </div>

          <v-list v-else class="files-pane__list" density="comfortable">
            <v-list-item v-for="document in documents" :key="document.id" class="files-pane__list-item">
              <template #prepend>
                <v-avatar size="36" color="primary" variant="flat">
                  <v-icon size="18" color="white">mdi-file-document</v-icon>
                </v-avatar>
              </template>
              <v-list-item-title class="text-subtitle-2">{{ document.name }}</v-list-item-title>
              <v-list-item-subtitle class="text-caption text-medium-emphasis">
                Updated {{ formatDate(document.updatedAt) }} · Status: {{ document.status || 'unknown' }}
              </v-list-item-subtitle>
            </v-list-item>
          </v-list>
        </div>
      </v-window-item>

      <v-window-item value="search" class="tool-shell__pane">
        <div class="chat-pane">
          <div class="chat-pane__messages" ref="messagesContainer">
            <div v-if="chatError" class="chat-pane__state">
              <v-icon size="28" color="primary">mdi-alert-circle-outline</v-icon>
              <p class="text-body-2 text-medium-emphasis mt-2 mb-0">{{ chatError.message || 'Something went wrong' }}</p>
            </div>
            <div v-else-if="!hasMessages" class="chat-pane__state">
              <p class="text-body-2 text-medium-emphasis mt-2 mb-0">Start a conversation to see AI responses.</p>
            </div>
            <template v-else>
              <div
                v-for="message in messages"
                :key="message.id"
                class="chat-bubble"
                :class="chatBubbleClass(message.role)"
              >
                <div class="chat-bubble__content" v-html="renderMarkdown(message.content)"></div>
              </div>
            </template>
          </div>

          <div class="chat-pane__composer">
            <v-textarea
              variant="outlined"
              density="comfortable"
              rows="2"
              max-rows="4"
              auto-grow
              placeholder="Type your question..."
              v-model="chatInput"
              :disabled="streaming"
            />
            <div class="chat-pane__composer-actions">
              <v-btn variant="text" color="primary" prepend-icon="mdi-broom" @click="clearChat" :disabled="!hasMessages || streaming">
                Clear
              </v-btn>
              <v-btn color="primary" variant="flat" prepend-icon="mdi-send" @click="sendMessage" :disabled="!canSend" :loading="streaming">
                Send
              </v-btn>
            </div>
          </div>
        </div>
      </v-window-item>

      <v-window-item value="creative" class="tool-shell__pane">
        <CreativeToolkitPanel />
      </v-window-item>
    </v-window>

    <v-dialog v-model="uploadDialogOpen" max-width="520" persistent>
      <v-card>
        <v-card-title class="text-h6 font-weight-semibold">Upload Files</v-card-title>
        <v-card-text>
          <v-file-input
            variant="outlined"
            multiple
            show-size
            accept=".pdf,.doc,.docx,.txt,.md,.ppt,.pptx,.xls,.xlsx"
            label="Select files"
            prepend-inner-icon="mdi-paperclip"
            v-model="uploadFilesModel"
          />
        </v-card-text>
        <v-card-actions class="justify-end">
          <v-btn variant="text" @click="closeUpload">Cancel</v-btn>
          <v-btn color="primary" variant="flat" :disabled="!canSubmitFiles" :loading="uploadInProgress" @click="submitUpload">
            Upload
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-dialog v-model="linksDialogOpen" max-width="520" persistent>
      <v-card>
        <v-card-title class="text-h6 font-weight-semibold">Add Webpage Links</v-card-title>
        <v-card-text class="d-flex flex-column gap-3">
          <v-text-field
            v-for="(link, index) in linkEntries"
            :key="`link-${index}`"
            variant="outlined"
            density="comfortable"
            label="Web link"
            placeholder="https://example.com/article"
            :append-inner-icon="linkEntries.length > 1 ? 'mdi-close' : undefined"
            @click:append-inner="() => removeLink(index)"
            @update:model-value="(value) => updateLink(index, value)"
            :model-value="link"
          />
          <v-btn variant="text" color="primary" prepend-icon="mdi-plus" @click="addLinkField">
            Add another link
          </v-btn>
        </v-card-text>
        <v-card-actions class="justify-end">
          <v-btn variant="text" @click="closeLinks">Cancel</v-btn>
          <v-btn color="primary" variant="flat" :disabled="!canSubmitLinks" :loading="linkSubmissionInProgress" @click="submitLinks">
            Save Links
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-card>
</template>

<script setup lang="ts">
import { onMounted, watch, nextTick, ref, computed } from 'vue'
import { storeToRefs } from 'pinia'
import CreativeToolkitPanel from './CreativeToolkitPanel.vue'
import {
  useWorkspaceToolkitStore,
  useDocumentsStore,
  useChatStore,
  useWorkspaceContextStore,
} from '@/stores'

const toolkitStore = useWorkspaceToolkitStore()
const documentsStore = useDocumentsStore()
const chatStore = useChatStore()
const workspaceContext = useWorkspaceContextStore()

const { activeTab } = storeToRefs(toolkitStore)
const {
  documents,
  loading: documentsLoading,
  error: documentsError,
  menuOpen,
  uploadDialogOpen,
  linksDialogOpen,
  uploadFiles,
  uploadInProgress,
  linkEntries,
  linkSubmissionInProgress,
  hasDocuments,
  canSubmitFiles,
  canSubmitLinks,
} = storeToRefs(documentsStore)

const { messages, inputValue: chatInput, streaming, error: chatError, hasMessages, canSend } = storeToRefs(chatStore)

const messagesContainer = ref<HTMLElement | null>(null)

const uploadFilesModel = computed({
  get: () => uploadFiles.value,
  set: (value: File[] | FileList | null) => {
    documentsStore.setUploadFiles(value)
  },
})

const chatBubbleClass = (role: string) => (role === 'user' ? 'chat-bubble--user' : 'chat-bubble--assistant')

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const inlineFormat = (line: string) => {
  let formatted = escapeHtml(line)
  formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>')
  formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>')
  return formatted
}

const renderMarkdown = (content: string) => {
  if (!content) {
    return ''
  }

  const lines = content.split('\n')
  const html: string[] = []
  let listItems: string[] = []
  let inCodeBlock = false
  let codeBuffer: string[] = []

  const flushList = () => {
    if (!listItems.length) return
    const items = listItems.map((item) => `<li>${inlineFormat(item)}</li>`).join('')
    html.push(`<ul>${items}</ul>`)
    listItems = []
  }

  const flushCode = () => {
    if (!codeBuffer.length) return
    html.push(`<pre><code>${escapeHtml(codeBuffer.join('\n'))}</code></pre>`)
    codeBuffer = []
  }

  lines.forEach((line) => {
    const trimmed = line.trim()

    if (trimmed.startsWith('```')) {
      if (inCodeBlock) {
        flushCode()
        inCodeBlock = false
      } else {
        flushList()
        inCodeBlock = true
      }
      return
    }

    if (inCodeBlock) {
      codeBuffer.push(line)
      return
    }

    const listMatch = line.match(/^\s*[-*]\s+(.*)$/)
    if (listMatch) {
      listItems.push(listMatch[1])
      return
    }

    flushList()

    const headingMatch = line.match(/^(#{1,3})\s+(.*)$/)
    if (headingMatch) {
      const level = Math.min(headingMatch[1].length + 2, 6)
      html.push(`<h${level}>${inlineFormat(headingMatch[2])}</h${level}>`)
      return
    }

    if (!trimmed) {
      html.push('')
      return
    }

    html.push(`<p>${inlineFormat(line)}</p>`)
  })

  if (inCodeBlock) {
    flushCode()
  } else {
    flushList()
  }

  return html.filter(Boolean).join('')
}

const formatDate = (value?: string) => {
  if (!value) {
    return 'recently'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return 'recently'
  }
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

const handleOpenUpload = () => {
  documentsStore.openUploadDialog()
}

const handleOpenLinks = () => {
  documentsStore.openLinksDialog()
}

const closeUpload = () => {
  documentsStore.closeUploadDialog()
}

const closeLinks = () => {
  documentsStore.closeLinksDialog()
}

const addLinkField = () => {
  documentsStore.addLinkField()
}

const removeLink = (index: number) => {
  documentsStore.removeLinkField(index)
}

const updateLink = (index: number, value: string) => {
  documentsStore.updateLinkField(index, value)
}

const submitUpload = () => {
  documentsStore.uploadSelectedFiles()
}

const submitLinks = () => {
  documentsStore.submitLinks()
}

const refreshDocuments = () => {
  documentsStore.fetchDocuments()
}

const clearChat = () => {
  chatStore.clearConversation()
}

const sendMessage = () => {
  chatStore.sendMessage()
}

onMounted(() => {
  documentsStore.fetchDocuments()
})

watch(
  () => workspaceContext.workspaceId,
  (workspaceId, previous) => {
    if (workspaceId && workspaceId !== previous) {
      documentsStore.fetchDocuments()
      chatStore.clearConversation()
    }
  },
  { immediate: true }
)

watch(
  () => messages.value.map((message) => `${message.id}:${message.content}`).join('|'),
  async () => {
    await nextTick()
    const container = messagesContainer.value
    if (container) {
      container.scrollTop = container.scrollHeight
    }
  }
)
</script>

<style scoped>
.tool-shell {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0; 
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  backdrop-filter: blur(10px);
  min-height: 0;
}

.tool-shell__tabs {
  flex: 0 0 auto;
  border-bottom: 1px solid rgba(15, 23, 42, 0.08);
}

.tool-shell__tab {
  display: flex;
  align-items: center;
  gap: 8px;
  text-transform: none;
  font-weight: 600;
}

.tool-shell__tabs :deep(.v-slide-group__prev),
.tool-shell__tabs :deep(.v-slide-group__next) {
  display: none;
}

.tool-shell__tabs :deep(.v-tab) {
  min-height: 40px;
  padding-block: 8px;
}

.tool-shell__window {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.tool-shell__window :deep(.v-window) {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.tool-shell__window :deep(.v-window__container),
.tool-shell__window :deep(.v-window-item) {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;         /* critical to enable inner scroll */
}


.tool-shell__pane {
  display: flex;
  flex-direction: column;
  padding: 20px;
  min-height: 0;
  flex: 1;
}

.files-pane {
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
}

.files-pane__actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.files-pane__state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 12px;
  color: rgba(17, 24, 39, 0.72);
}

.files-pane__list {
  flex: 1;
  overflow-y: auto;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 20px;
}

.files-pane__list-item + .files-pane__list-item {
  border-top: 1px solid rgba(15, 23, 42, 0.04);
}

.chat-pane {
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
  min-height: 0;
}

.chat-pane__header {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.chat-pane__messages {
  flex: 1 1 0;
  min-height: 0;         
  overflow-y: auto;      
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px 4px 16px 0;
  margin-bottom: 12px;
  scrollbar-width: thin;
}

.chat-pane__state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 8px;
  padding: 24px;
  color: rgba(17, 24, 39, 0.72);
}

.chat-bubble {
  max-width: 92%;
  padding: 12px 16px;
  border-radius: 18px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background-color: #ffffff;
  align-self: flex-start;
  box-shadow: 0 12px 24px -16px rgba(15, 23, 42, 0.24);
}

.chat-bubble--user {
  align-self: flex-end;
  background-color: rgba(22, 101, 52, 0.08);
  border-color: rgba(22, 101, 52, 0.16);
}

.chat-bubble__content :deep(p) {
  margin: 0 0 8px 0;
}

.chat-bubble__content :deep(p:last-child) {
  margin-bottom: 0;
}

.chat-bubble__content :deep(code) {
  background-color: rgba(22, 101, 52, 0.08);
  padding: 2px 4px;
  border-radius: 4px;
}

.chat-bubble__content :deep(pre) {
  background-color: rgba(15, 23, 42, 0.92);
  color: #fefefe;
  padding: 12px;
  border-radius: 12px;
  overflow-x: auto;
  margin: 0 0 8px 0;
}

.chat-pane__composer {
  display: flex;
  flex-direction: column;
  gap: 12px;
  border-top: 1px solid rgba(15, 23, 42, 0.08);
  padding-top: 12px;
}

.chat-pane__composer-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

@media (max-width: 960px) {
  .tool-shell__pane {
    padding: 16px;
  }

  .files-pane__actions {
    justify-content: space-between;
  }
}
</style>
