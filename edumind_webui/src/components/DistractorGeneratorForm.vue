<template>
  <div class="distractor-form">
    <v-form @submit.prevent="handleGenerate">
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
      </div>

      <v-alert
        v-if="selectionActive"
        type="info"
        variant="tonal"
        density="comfortable"
        class="distractor-form__selection-hint"
      >
        Click an MCQ card on the canvas to select it.
      </v-alert>

      <v-sheet
        v-if="selectedQuestionText"
        class="distractor-form__preview"
        color="primary"
        variant="tonal"
      >
        <p class="distractor-form__preview-label">Selected question</p>
        <p class="distractor-form__preview-text">{{ selectedQuestionText.slice(0, 40) }}...</p>
      </v-sheet>

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
      />
      <div class="distractor-form__actions">
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
        <v-btn variant="text" color="primary" @click="reset" :disabled="loading">Clear</v-btn>
      </div>
      <v-alert
        v-if="error"
        type="error"
        variant="tonal"
        density="comfortable"
        class="mt-3"
      >
        {{ errorMessage }}
      </v-alert>
      <v-alert
        v-else-if="lastMessage"
        type="success"
        variant="tonal"
        density="comfortable"
        class="mt-3"
      >
        {{ lastMessage }}
      </v-alert>
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
  gap: 16px;
}

.distractor-form__selector {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.distractor-form__selector-btn--active {
  background-color: rgba(52, 225, 118, 0.08);
}

.distractor-form__selection-hint {
  margin-top: -4px;
}

.distractor-form__preview {
  margin: 4px;
  border-radius: 16px;
  padding: 12px 16px;
}

.distractor-form__preview-label {
  margin: 0 0 4px 0;
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.distractor-form__preview-text {
  margin: 0;
  font-size: 0.95rem;
  line-height: 1.4;
}

.distractor-form__actions {
  display: flex;
  align-items: center;
  gap: 12px;
}
</style>
