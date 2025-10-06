<template>
  <div class="mcq-generator">
    <div class="mcq-generator__body">
      <v-textarea
        v-model="topics"
        label="Topics"
        variant="outlined"
        auto-grow
        rows="2"
        max-rows="4"
        density="comfortable"
        placeholder="e.g. Photosynthesis basics, plant biology"
        :disabled="loading"
      />

      <div class="mcq-generator__inputs">
        <v-text-field
          v-model.number="quantityModel"
          label="Number of questions"
          type="number"
          min="1"
          max="20"
          variant="outlined"
          density="comfortable"
          :disabled="loading"
        />

        <v-select
          v-model="difficulty"
          :items="difficultyItems"
          label="Difficulty"
          variant="outlined"
          density="comfortable"
          :disabled="loading"
        />
      </div>

      <v-textarea
        v-model="userInstructions"
        label="Primary instructions"
        variant="outlined"
        rows="2"
        max-rows="4"
        auto-grow
        density="comfortable"
        placeholder="Include key misconceptions or emphasise real-world examples"
        :disabled="loading"
      />

      <v-switch
        v-model="allowMulti"
        inset
        label="Allow multiple correct answers"
        :disabled="loading"
      />

      <div v-if="errorMessage" class="mcq-generator__alert">
        <v-alert type="error" variant="tonal" density="compact">{{ errorMessage }}</v-alert>
      </div>

      <div v-if="successMessage" class="mcq-generator__alert">
        <v-alert type="success" variant="tonal" density="compact">{{ successMessage }}</v-alert>
      </div>
    </div>

    <div class="mcq-generator__actions">
      <v-spacer />
      <v-btn color="primary" variant="flat" :disabled="!canGenerate" :loading="loading" prepend-icon="mdi-timer-play" @click="generate">
        Generate Questions
      </v-btn>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useMcqGeneratorStore } from '@/stores'

const mcqStore = useMcqGeneratorStore()
const {
  topics,
  quantity,
  difficulty,
  allowMulti,
  userInstructions,
  loading,
  error,
  lastMessage,
} = storeToRefs(mcqStore)

const difficultyItems = [
  { title: 'Easy', value: 'easy' },
  { title: 'Medium', value: 'medium' },
  { title: 'Hard', value: 'hard' },
]

const quantityModel = computed({
  get: () => quantity.value,
  set: (value: number) => {
    const normalised = Number.isNaN(Number(value)) ? 1 : Math.min(Math.max(Number(value), 1), 20)
    mcqStore.quantity = normalised
  },
})

const canGenerate = computed(() => mcqStore.canGenerate)

const errorMessage = computed(() => (error.value ? error.value.message || 'Failed to generate questions.' : ''))
const successMessage = computed(() => lastMessage.value)

const generate = () => {
  mcqStore.generate()
}

onMounted(() => {
  mcqStore.initialise()
})
</script>

<style scoped>
.mcq-generator {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.mcq-generator__body {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.mcq-generator__inputs {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.mcq-generator__inputs :deep(.v-input) {
  flex: 1;
  min-width: 160px;
}

.mcq-generator__advanced {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  border: 1px dashed rgba(15, 23, 42, 0.12);
  border-radius: 16px;
  background-color: rgba(22, 101, 52, 0.04);
}

.mcq-generator__advanced-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.mcq-generator__advanced-row :deep(.v-input) {
  flex: 1;
  min-width: 180px;
}

.mcq-generator__actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.mcq-generator__alert {
  display: flex;
}

@media (max-width: 960px) {
  .mcq-generator__inputs {
    flex-direction: column;
  }

  .mcq-generator__advanced-row {
    flex-direction: column;
  }

  .mcq-generator__actions {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
