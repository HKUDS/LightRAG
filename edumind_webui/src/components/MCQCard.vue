<template>
  <v-card class="question-card" variant="outlined">
    <v-card-item>
      <div class="d-flex align-center justify-space-between flex-wrap gap-3">
        <div class="d-flex align-center gap-2">
          <v-chip label color="primary" variant="flat" size="small" class="chip--contrast">MCQ</v-chip>
          <v-chip label color="primary" variant="outlined" size="small" class="text-uppercase">
            {{ difficultyLabel }}
          </v-chip>
        </div>
        <v-btn
          v-if="showActions"
          variant="flat"
          size="small"
          color="primary"
          prepend-icon="mdi-delete-outline"
          @click="deleteItem"
        >
          Remove
        </v-btn>
      </div>
    </v-card-item>

    <v-divider class="mx-6" />

    <v-card-text>
      <p class="text-body-1 font-weight-medium mb-4">
        {{ mcq.question }}
      </p>

      <div class="options">
        <div
          v-for="(option, index) in mcq.options"
          :key="`${mcq.id}-${index}`"
          class="option"
          :class="{ 'option--correct': mcq.correctOptions.includes(index) }"
        >
          <v-avatar size="28" color="primary" variant="flat" class="option__index">
            <span class="option__index-text">{{ optionLabel(index) }}</span>
          </v-avatar>
          <span class="option-text">{{ option }}</span>
          <v-icon
            v-if="mcq.correctOptions.includes(index)"
            size="18"
            color="rgba(17, 24, 39, 1)"
          >
            mdi-check-circle-outline
          </v-icon>
        </div>
      </div>

      <v-divider class="my-4" />

      <v-row class="metadata" align="stretch" no-gutters>
        <v-col cols="12" md="6" class="pr-md-4 mb-4 mb-md-0">
          <p class="metadata__label">AI Rationale</p>
          <p class="metadata__value">{{ mcq.aiRational }}</p>
        </v-col>
        <v-col cols="12" md="3" class="pr-md-4 mb-4 mb-md-0">
          <p class="metadata__label">Source</p>
          <p class="metadata__value">{{ mcq.source }}</p>
        </v-col>
        <v-col cols="12" md="3">
          <p class="metadata__label">Tag</p>
          <v-chip label size="small" color="primary" variant="outlined">{{ mcq.tag }}</v-chip>
        </v-col>
      </v-row>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { computed } from 'vue';

interface MCQContent {
  id: string;
  question: string;
  options: string[];
  correctOptions: number[];
  difficultyLevel: 'easy' | 'medium' | 'hard';
  aiRational: string;
  source: string;
  tag: string;
}

const props = withDefaults(
  defineProps<{
    mcq: MCQContent;
    showActions?: boolean;
  }>(),
  {
    showActions: true,
  }
);

const emit = defineEmits<{
  delete: [id: string];
}>();

const difficultyLabel = computed(() => {
  const labels = {
    easy: 'Easy',
    medium: 'Medium',
    hard: 'Hard',
  } as const;

  return labels[props.mcq.difficultyLevel] ?? props.mcq.difficultyLevel;
});

const optionLabel = (index: number) => String.fromCharCode(65 + index);

const deleteItem = () => {
  emit('delete', props.mcq.id);
};
</script>

<style scoped>
.question-card {
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 24px 40px -24px rgba(15, 23, 42, 0.24);
}

.options {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 20px;
}

.option {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 16px;
  background-color: #ffffff;
}

.option--correct {
  border-color: #166534;
  box-shadow: 0 0 0 1px rgba(22, 101, 52, 0.2);
}

.option__index {
  flex-shrink: 0;
}

.option__index-text {
  font-size: 0.9rem;
  font-weight: 600;
  color: #ffffff;
}

.option-text {
  flex: 1;
  font-size: 0.95rem;
  color: rgba(15, 23, 42, 0.92);
}

.chip--contrast {
  color: #ffffff !important;
}

.metadata__label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(15, 23, 42, 0.65);
  margin-bottom: 6px;
  font-weight: 600;
}

.metadata__value {
  margin: 0;
  color: rgba(15, 23, 42, 0.8);
  line-height: 1.45;
}
</style>
