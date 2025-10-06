<template>
  <div class="distractor-form">
    <v-form @submit.prevent="handleGenerate">
      <v-text-field
        v-model="questionIdModel"
        label="Question ID"
        variant="outlined"
        density="comfortable"
        placeholder="Enter the question's ID"
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
const { questionId, instructions, loading, error, lastMessage } = storeToRefs(distractorStore)

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

const handleGenerate = () => {
  distractorStore.generate()
}

const reset = () => {
  distractorStore.reset()
}
</script>

<style scoped>
.distractor-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.distractor-form__actions {
  display: flex;
  align-items: center;
  gap: 12px;
}
</style>
