<template>
  <v-card
    class="question-card"
    :class="{
      'question-card--selectable': selectable,
      'question-card--selected': selected,
      'question-card--selection-active': selectable && !selected,
    }"
    variant="outlined"
    @click="handleSelect"
  >
    <v-card-item>
      <div class="d-flex align-center justify-space-between flex-wrap gap-3">
        <div class="d-flex align-center gap-2">
          <v-chip label color="primary" variant="flat" size="small" class="chip--contrast">MCQ</v-chip>
          <v-chip label color="primary" variant="outlined" size="small" class="text-uppercase">
            {{ difficultyLabel }}
          </v-chip>
        </div>
        <div class="question-card__header-actions">
          <v-btn
            v-if="hasVariants"
            variant="text"
            size="small"
            color="primary"
            prepend-icon="mdi-arrange-bring-forward"
            @click.stop="toggleVariants"
          >
            {{ showVariants ? 'Hide' : 'Show' }} Variants ({{ variants.length }})
          </v-btn>
          <v-btn
            v-if="showActions"
            variant="flat"
            size="small"
            color="primary"
            prepend-icon="mdi-delete-outline"
            @click.stop="deleteItem"
          >
            Remove
          </v-btn>
        </div>
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

      <v-expand-transition>
        <div v-if="showVariants" class="variants">
          <div
            v-for="variant in variants"
            :key="variant.id || variantLabel(variant)"
            class="variant-card"
          >
            <div class="variant-card__header">
              <v-chip
                label
                size="small"
                color="primary"
                variant="outlined"
              >
                {{ variantLabel(variant) }}
              </v-chip>
            </div>
            <div class="variant-card__options">
              <div
                v-for="(option, index) in variant.options"
                :key="`${variant.id || variantLabel(variant)}-${index}`"
                class="variant-option"
                :class="{ 'variant-option--correct': variant.correct_answers.includes(index) }"
              >
                <v-avatar size="24" color="primary" variant="flat" class="variant-option__index">
                  <span class="variant-option__index-text">{{ optionLabel(index) }}</span>
                </v-avatar>
                <span class="variant-option__text">{{ option }}</span>
              </div>
            </div>
            <p v-if="variant.rationale" class="variant-card__rationale">
              {{ variant.rationale }}
            </p>
          </div>
        </div>
      </v-expand-transition>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';

interface MCQContent {
  id: string;
  question: string;
  options: string[];
  correctOptions: number[];
  difficultyLevel: 'easy' | 'medium' | 'hard';
  aiRational: string;
  source: string;
  tag: string;
  variants?: Array<{
    id?: string;
    difficulty_level?: string;
    options: string[];
    correct_answers: number[];
    rationale?: string;
  }>;
}

const props = withDefaults(
  defineProps<{
    mcq: MCQContent;
    showActions?: boolean;
    selectable?: boolean;
    selected?: boolean;
  }>(),
  {
    showActions: true,
    selectable: false,
    selected: false,
  }
);

const emit = defineEmits<{
  delete: [id: string];
  select: [payload: { id: string; question: string }];
}>();

const showVariants = ref(false);
const variants = computed(() => Array.isArray(props.mcq.variants) ? props.mcq.variants : []);
const hasVariants = computed(() => variants.value.length > 0);

const difficultyLabel = computed(() => {
  const labels = {
    easy: 'Easy',
    medium: 'Medium',
    hard: 'Hard',
  } as const;

  return labels[props.mcq.difficultyLevel] ?? props.mcq.difficultyLevel;
});

const optionLabel = (index: number) => String.fromCharCode(65 + index);

const variantLabel = (variant: MCQContent['variants'][number]) => {
  if (variant.difficulty_level) {
    return `${variant.difficulty_level.replace(/_/g, ' ')}`.replace(/\b\w/g, (c) => c.toUpperCase());
  }
  return 'Variant';
};

const deleteItem = () => {
  emit('delete', props.mcq.id);
};

const toggleVariants = () => {
  showVariants.value = !showVariants.value;
};

const handleSelect = () => {
  if (!props.selectable) {
    return;
  }
  emit('select', { id: props.mcq.id, question: props.mcq.question });
};
</script>

<style scoped>
.question-card {
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 24px 40px -24px rgba(15, 23, 42, 0.24);
  transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
}

.question-card--selectable {
  cursor: pointer;
}

.question-card--selection-active:hover {
  border-color: rgba(22, 101, 52, 0.4);
  box-shadow: 0 0 0 2px rgba(22, 101, 52, 0.18);
  transform: translateY(-1px);
}

.question-card--selected {
  border-color: rgba(22, 101, 52, 0.5);
  box-shadow: 0 0 0 2px rgba(22, 101, 52, 0.24);
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

.question-card__header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
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

.variants {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.variant-card {
  border: 1px solid rgba(15, 23, 42, 0.1);
  border-radius: 16px;
  padding: 16px;
  background-color: rgba(249, 250, 251, 0.6);
}

.variant-card__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.variant-card__options {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.variant-option {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background-color: #ffffff;
}

.variant-option--correct {
  border-color: #0f766e;
  background-color: rgba(15, 118, 110, 0.08);
}

.variant-option__index {
  flex-shrink: 0;
}

.variant-option__index-text {
  color: #ffffff;
  font-size: 0.85rem;
  font-weight: 600;
}

.variant-option__text {
  flex: 1;
  color: rgba(15, 23, 42, 0.9);
  font-size: 0.9rem;
}

.variant-card__rationale {
  margin: 12px 0 0;
  font-size: 0.85rem;
  color: rgba(15, 23, 42, 0.7);
}
</style>
