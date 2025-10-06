<template>
  <div class="distractor-form">
    <v-divider class="distractor-form__divider" />

    <v-form class="distractor-form__content" @submit.prevent="handleGenerate">
      <section class="distractor-form__section">
        <div class="distractor-form__selector">
          <v-btn
            variant="outlined"
            color="primary"
            prepend-icon="mdi-cursor-default-click"
            :class="{ 'distractor-form__selector-btn--active': selectionActive }"
            @click.prevent="toggleSelection"
            :disabled="loading"
          >
            {{ selectionButtonLabel }}
          </v-btn>
          <transition name="fade">
            <v-chip
              v-if="questionId"
              label
              color="primary"
              variant="flat"
              size="small"
            >
              ID: {{ questionId }}
            </v-chip>
          </transition>
        </div>

        <transition name="fade">
          <v-alert
            v-if="selectionActive"
            type="info"
            variant="tonal"
            density="comfortable"
            class="distractor-form__selection-hint"
          >
            Click an MCQ card on the canvas to import it here.
          </v-alert>
        </transition>

        <transition name="fade">
          <v-sheet
            v-if="selectedQuestionText"
            class="distractor-form__preview"
            color="primary"
            variant="tonal"
          >
            <div class="distractor-form__preview-header">
              <p class="distractor-form__preview-label">Selected question</p>
              <v-btn
                variant="text"
                color="primary"
                size="small"
                prepend-icon="mdi-sync"
                @click.prevent="toggleSelection"
              >
                Choose different
              </v-btn>
            </div>
            <p class="distractor-form__preview-text">{{ selectedQuestionText }}</p>
          </v-sheet>
        </transition>
      </section>

      <section class="distractor-form__section">
        <v-text-field
          v-model="questionIdModel"
          label="Question ID"
          variant="outlined"
          density="comfortable"
          placeholder="Select a question or enter the ID manually"
          :disabled="loading"
          required
        />
        <v-textarea
          v-model="instructionsModel"
          label="Additional instructions (optional)"
          variant="outlined"
          density="comfortable"
          rows="3"
          auto-grow
          :disabled="loading"
          placeholder="e.g. Focus on misconceptions or highlight similar-sounding terms"
        />
      </section>

      <div class="distractor-form__actions">
        <div class="distractor-form__action-copy">
          <p class="text-caption text-medium-emphasis mb-0">
            Variants are added to the selected MCQ on the current canvas. You’ll see them expand under the card.
          </p>
        </div>
        <div class="distractor-form__action-buttons">
          <v-btn
            variant="text"
            color="primary"
            @click="reset"
            :disabled="loading"
          >
            Reset
          </v-btn>
          <v-btn
            color="primary"
            variant="flat"
            prepend-icon="mdi-arrange-bring-forward"
            type="submit"
            :loading="loading"
            :disabled="loading || !canSubmit"
          >
            Generate Distractors
          </v-btn>
        </div>
      </div>

      <transition name="fade">
        <v-alert
          v-if="error"
          type="error"
          variant="tonal"
          density="comfortable"
          class="distractor-form__alert"
        >
          {{ errorMessage }}
        </v-alert>
      </transition>
      <transition name="fade">
        <v-alert
          v-if="lastMessage"
          type="success"
          variant="tonal"
          density="comfortable"
          class="distractor-form__alert"
        >
          {{ lastMessage }}
        </v-alert>
      </transition>
    </v-form>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useDistractorGeneratorStore } from '@/stores'

const distractorStore = useDistractorGeneratorStore()
const {
  questionId,
  instructions,
  loading,
  error,
  lastMessage,
  selectionActive,
  selectedQuestionText,
} = storeToRefs(distractorStore)

const questionIdModel = computed({
  get: () => questionId.value,
  set: (value: string) => distractorStore.setQuestionId(value),
})

const instructionsModel = computed({
  get: () => instructions.value,
  set: (value: string) => distractorStore.setInstructions(value),
})

const canSubmit = computed(() => questionId.value.trim().length > 0 && !loading.value)
const errorMessage = computed(() => (error.value ? error.value.message || 'Failed to generate variants.' : ''))
const selectionButtonLabel = computed(() => {
  if (selectionActive.value) {
    return 'Click a question to use it'
  }
  if (questionId.value) {
    return 'Change selected question'
  }
  return 'Select a question from the canvas'
})

const handleGenerate = () => {
  distractorStore.generate()
}

const reset = () => {
  distractorStore.reset()
}

const toggleSelection = () => {
  if (selectionActive.value) {
    distractorStore.cancelSelection()
  } else {
    distractorStore.beginSelection()
  }
}
</script>

<style scoped>
.distractor-form {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 4px;
}

.distractor-form__header {
  display: flex;
  align-items: center;
  gap: 16px;
}

.distractor-form__header-copy {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.distractor-form__eyebrow-line {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.distractor-form__eyebrow {
  margin: 0;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(15, 23, 42, 0.6);
}

.distractor-form__badge {
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: rgba(255, 255, 255, 0.96);
}

.distractor-form__title {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 600;
  color: rgba(15, 23, 42, 0.92);
}

.distractor-form__subtitle {
  margin: 0;
  color: rgba(15, 23, 42, 0.72);
  font-size: 0.95rem;
}

.distractor-form__divider {
  margin: -4px 0 4px;
}

.distractor-form__content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.distractor-form__section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.distractor-form__selector {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.distractor-form__selector-btn--active {
  background-color: rgba(22, 101, 52, 0.08);
  box-shadow: 0 0 0 1px rgba(22, 101, 52, 0.16);
}

.distractor-form__selection-hint {
  margin-top: -4px;
}

.distractor-form__preview {
  border-radius: 18px;
  padding: 14px 18px;
  background-color: rgba(22, 101, 52, 0.08);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.distractor-form__preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.distractor-form__preview-label {
  margin: 0;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.distractor-form__preview-text {
  margin: 0;
  font-size: 0.96rem;
  line-height: 1.45;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.distractor-form__actions {
  display: flex;
  gap: 16px;
  align-items: flex-start;
  justify-content: right;
  flex-wrap: wrap;
}

.distractor-form__action-copy {
  flex: 1 1 200px;
}

.distractor-form__action-buttons {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.distractor-form__alert {
  margin-top: -4px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 960px) {
  .distractor-form__header {
    align-items: flex-start;
  }

  .distractor-form__actions {
    flex-direction: column;
    align-items: stretch;
  }

  .distractor-form__action-buttons {
    justify-content: flex-start;
  }
}
</style>
