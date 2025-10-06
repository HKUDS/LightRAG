import { defineStore } from 'pinia'
import { documentsApi } from '@/api'
import { useWorkspaceContextStore } from '../workspaceContext'

const createDefaultLinks = () => ['']

export const useDocumentsStore = defineStore('documents', {
  state: () => ({
    documents: [],
    loading: false,
    error: null,
    menuOpen: false,
    uploadDialogOpen: false,
    linksDialogOpen: false,
    uploadFiles: [],
    uploadInProgress: false,
    linkEntries: createDefaultLinks(),
    linkSubmissionInProgress: false,
  }),
  getters: {
    hasDocuments: (state) => state.documents.length > 0,
    canSubmitLinks: (state) =>
      state.linkEntries.some((entry) => entry && entry.trim().length > 0),
    canSubmitFiles: (state) => state.uploadFiles.length > 0,
  },
  actions: {
    toggleMenu(open) {
      this.menuOpen = typeof open === 'boolean' ? open : !this.menuOpen
    },
    openUploadDialog() {
      this.uploadFiles = []
      this.uploadDialogOpen = true
      this.toggleMenu(false)
    },
    closeUploadDialog() {
      this.uploadDialogOpen = false
      this.uploadFiles = []
    },
    openLinksDialog() {
      this.linkEntries = createDefaultLinks()
      this.linksDialogOpen = true
      this.toggleMenu(false)
    },
    closeLinksDialog() {
      this.linksDialogOpen = false
      this.linkEntries = createDefaultLinks()
      this.linkSubmissionInProgress = false
    },
    addLinkField() {
      this.linkEntries = [...this.linkEntries, '']
    },
    removeLinkField(index) {
      if (this.linkEntries.length === 1) {
        this.linkEntries = ['']
        return
      }
      this.linkEntries = this.linkEntries.filter((_, idx) => idx !== index)
    },
    updateLinkField(index, value) {
      this.linkEntries = this.linkEntries.map((entry, idx) =>
        idx === index ? value : entry
      )
    },
    setUploadFiles(files) {
      if (!files) {
        this.uploadFiles = []
        return
      }

      if (Array.isArray(files)) {
        this.uploadFiles = files
      } else if (files instanceof FileList) {
        this.uploadFiles = Array.from(files)
      } else {
        this.uploadFiles = []
      }
    },
    async fetchDocuments() {
      const workspaceStore = useWorkspaceContextStore()
      if (!workspaceStore.hasWorkspace) {
        this.documents = []
        return
      }

      this.loading = true
      this.error = null
      try {
        const response = await documentsApi.getPaginatedDocuments({
          payload: {
            page: 1,
            page_size: 50,
            sort_field: 'updated_at',
            sort_direction: 'desc',
          },
          headers: {
            'X-Workspace': workspaceStore.workspaceId,
          },
        })

        const docs = response?.documents || []
        this.documents = docs.map((doc) => ({
          id: doc.id,
          name: doc.file_path || doc.id,
          status: doc.status,
          createdAt: doc.created_at,
          updatedAt: doc.updated_at,
          summary: doc.content_summary,
        }))
      } catch (error) {
        console.error('Failed to fetch documents', error)
        this.error = error
      } finally {
        this.loading = false
      }
    },
    async uploadSelectedFiles() {
      if (!this.canSubmitFiles || this.uploadInProgress) {
        return
      }

      const workspaceStore = useWorkspaceContextStore()
      if (!workspaceStore.hasWorkspace) {
        return
      }

      this.uploadInProgress = true
      try {
        const uploads = this.uploadFiles.map((file) =>
          documentsApi.uploadDocument({
            file,
            headers: {
              'X-Workspace': workspaceStore.workspaceId,
            },
          })
        )

        await Promise.all(uploads)
        this.closeUploadDialog()
        await this.fetchDocuments()
      } catch (error) {
        console.error('File upload failed', error)
        this.error = error
      } finally {
        this.uploadInProgress = false
      }
    },
    async submitLinks() {
      if (!this.canSubmitLinks || this.linkSubmissionInProgress) {
        return
      }

      const workspaceStore = useWorkspaceContextStore()
      if (!workspaceStore.hasWorkspace) {
        return
      }

      const links = this.linkEntries
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0)

      if (!links.length) {
        return
      }

      this.linkSubmissionInProgress = true
      try {
        await documentsApi.insertLinks({
          payload: { links },
          headers: {
            'X-Workspace': workspaceStore.workspaceId,
          },
        })
        this.closeLinksDialog()
        await this.fetchDocuments()
      } catch (error) {
        console.error('Link submission failed', error)
        this.error = error
      } finally {
        this.linkSubmissionInProgress = false
      }
    },
  },
})
