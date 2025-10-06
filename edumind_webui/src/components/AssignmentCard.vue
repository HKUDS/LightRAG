<template>
  <v-card class="question-card" variant="outlined">
    <v-card-item>
      <div class="d-flex align-center justify-space-between flex-wrap gap-3">
        <div class="d-flex align-center gap-2">
          <v-chip label color="primary" variant="flat" size="small" class="chip--contrast">Assignment</v-chip>
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
        {{ assignment.question }}
      </p>

      <v-row class="metadata" align="stretch" no-gutters>
        <v-col cols="12" md="6" class="pr-md-4 mb-4 mb-md-0">
          <p class="metadata__label">AI Rationale</p>
          <p class="metadata__value">{{ assignment.aiRational }}</p>
        </v-col>
        <v-col cols="12" md="3" class="pr-md-4 mb-4 mb-md-0">
          <p class="metadata__label">Source</p>
          <p class="metadata__value">{{ assignment.source }}</p>
        </v-col>
        <v-col cols="12" md="3">
          <p class="metadata__label">Tag</p>
          <v-chip label size="small" color="primary" variant="outlined">{{ assignment.tag }}</v-chip>
        </v-col>
      </v-row>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { computed } from 'vue';

interface AssignmentContent {
  id: string;
  question: string;
  difficultyLevel: 'easy' | 'medium' | 'hard';
  aiRational: string;
  source: string;
  tag: string;
}

const props = withDefaults(
  defineProps<{
    assignment: AssignmentContent;
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

  return labels[props.assignment.difficultyLevel] ?? props.assignment.difficultyLevel;
});

const deleteItem = () => {
  emit('delete', props.assignment.id);
};
</script>

<style scoped>
.question-card {
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 24px 40px -24px rgba(15, 23, 42, 0.18);
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

.chip--contrast {
  color: #ffffff !important;
}
</style>
