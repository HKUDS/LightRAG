<template>
  <div class="questions-page">
    <v-app-bar flat color="white" class="questions-page__app-bar" height="76">
      <v-container class="py-0" fluid>
        <div class="questions-page__bar">
          <div class="questions-page__bar-left">
            <h1 class="text-h5 font-weight-semibold mb-1">Question Library</h1>
            <p class="text-body-2 text-medium-emphasis mb-0">
              Browse and refine every prompt your team has generated.
            </p>
          </div>
          <v-btn
            color="primary"
            variant="flat"
            class="px-6"
            prepend-icon="mdi-arrow-left"
            :to="{ name: 'Dashboard' }"
          >
            Back to Dashboard
          </v-btn>
        </div>
      </v-container>
    </v-app-bar>

    <v-container fluid class="questions-page__content">
      <v-row class="questions-page__grid" align="stretch" justify="stretch">
        <v-col cols="12" md="4" lg="3" class="questions-page__filters">
          <v-card elevation="2" class="filters-card">
            <v-card-item>
              <v-card-title class="text-h6 font-weight-semibold">Filters</v-card-title>
              <v-card-subtitle class="text-body-2 text-medium-emphasis">
                Narrow down the library by attributes.
              </v-card-subtitle>
            </v-card-item>
            <v-card-text class="pt-0">
              <v-text-field
                v-model="searchQuery"
                label="Search questions"
                variant="solo"
                color="primary"
                prepend-inner-icon="mdi-magnify"
                hide-details
                density="comfortable"
                class="mb-6"
              />

              <v-select
                v-model="filterType"
                :items="typeOptions"
                item-title="title"
                item-value="value"
                label="Question type"
                variant="solo"
                density="comfortable"
                hide-details
                class="mb-4"
              />

              <v-select
                v-model="filterDifficulty"
                :items="difficultyOptions"
                item-title="title"
                item-value="value"
                label="Difficulty"
                variant="solo"
                density="comfortable"
                hide-details
                class="mb-4"
              />

              <v-select
                v-model="filterTag"
                :items="tagOptions"
                item-title="title"
                item-value="value"
                label="Tag"
                variant="solo"
                density="comfortable"
                hide-details
                class="mb-6"
                :menu-props="{ maxHeight: 240 }"
              />

              <v-btn
                block
                variant="outlined"
                color="primary"
                prepend-icon="mdi-refresh"
                @click="resetFilters"
              >
                Reset Filters
              </v-btn>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" md="8" lg="9" class="questions-page__results">
          <v-card elevation="2" class="results-card">
            <v-card-item>
              <div class="results-card__header">
                <div>
                  <v-card-title class="text-h6 font-weight-semibold mb-1">Results</v-card-title>
                  <v-card-subtitle class="text-body-2 text-medium-emphasis">
                    {{ resultsSummary }}
                  </v-card-subtitle>
                </div>
                <v-chip color="primary" variant="flat" class="results-card__chip" label>
                  {{ filteredQuestions.length }} total
                </v-chip>
              </div>
            </v-card-item>

            <v-divider />

            <v-card-text class="results-card__body">
              <div v-if="loading" class="state state--loading">
                <v-progress-circular indeterminate color="primary" size="32" />
                <p class="text-body-2 text-medium-emphasis mt-4 mb-0">Loading questions…</p>
              </div>

              <div v-else-if="filteredQuestions.length === 0" class="state state--empty">
                <v-avatar color="primary" size="56" variant="flat">
                  <v-icon size="28" color="white">mdi-database-search</v-icon>
                </v-avatar>
                <h3 class="text-subtitle-1 font-weight-semibold mt-4 mb-2">No questions match</h3>
                <p class="text-body-2 text-medium-emphasis mb-0">
                  Adjust your filters or generate fresh content to fill the library.
                </p>
              </div>

              <div v-else class="results-card__list">
                <template v-for="question in filteredQuestions" :key="question.id">
                  <MCQCard
                    v-if="question.type === 'mcq'"
                    :mcq="convertToMCQ(question)"
                    :show-actions="false"
                  />
                  <AssignmentCard
                    v-else
                    :assignment="convertToAssignment(question)"
                    :show-actions="false"
                  />
                </template>
              </div>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { storeToRefs } from 'pinia';
import MCQCard from '@/components/MCQCard.vue';
import AssignmentCard from '@/components/AssignmentCard.vue';
import { useQuestionsStore } from '@/stores';

interface DatabaseQuestion {
  id: string;
  type: 'mcq' | 'assignment';
  question: string;
  options: string[] | null;
  correct_options: number[] | null;
  difficulty_level: 'easy' | 'medium' | 'hard';
  ai_rational: string;
  source: string;
  tag: string;
  created_at: string;
}

const questionsStore = useQuestionsStore();
const {
  loading,
  searchQuery,
  filterType,
  filterDifficulty,
  filterTag,
  uniqueTags,
  filteredQuestions,
  resultsSummary,
} = storeToRefs(questionsStore);

const typeOptions = [
  { title: 'All types', value: 'all' },
  { title: 'MCQ', value: 'mcq' },
  { title: 'Assignment', value: 'assignment' },
];

const difficultyOptions = [
  { title: 'All levels', value: 'all' },
  { title: 'Easy', value: 'easy' },
  { title: 'Medium', value: 'medium' },
  { title: 'Hard', value: 'hard' },
];

const tagOptions = computed(() => [
  { title: 'All tags', value: 'all' },
  ...uniqueTags.value.map((tag) => ({ title: tag, value: tag })),
]);

const resetFilters = () => {
  questionsStore.resetFilters();
};

const convertToMCQ = (question: DatabaseQuestion) => ({
  id: question.id,
  question: question.question,
  options: question.options ?? [],
  correctOptions: question.correct_options ?? [],
  difficultyLevel: question.difficulty_level,
  aiRational: question.ai_rational,
  source: question.source,
  tag: question.tag,
});

const convertToAssignment = (question: DatabaseQuestion) => ({
  id: question.id,
  question: question.question,
  difficultyLevel: question.difficulty_level,
  aiRational: question.ai_rational,
  source: question.source,
  tag: question.tag,
});

onMounted(() => {
  questionsStore.hydrateWithSampleData();
});
</script>

<style scoped>
.questions-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #ffffff;
  overflow: hidden;
}

.questions-page__app-bar {
  border-bottom: 1px solid rgba(22, 101, 52, 0.12);
}

.questions-page__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 32px;
  padding-inline: 16px;
}

.questions-page__bar-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.questions-page__content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 32px 40px;
  overflow: hidden;
}

.questions-page__grid {
  flex: 1;
  min-height: 0;
  gap: 24px;
}

.questions-page__filters,
.questions-page__results {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.filters-card,
.results-card {
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.results-card__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.results-card__chip {
  color: #ffffff !important;
}

.results-card__body {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px;
  overflow: hidden;
}

.results-card__list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding-right: 8px;
}

.state {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 64px 24px;
  border-radius: 24px;
  border: 1px dashed rgba(22, 101, 52, 0.18);
  background: rgba(255, 255, 255, 0.9);
  gap: 16px;
}

.state--loading {
  border-style: solid;
}

@media (max-width: 960px) {
  .questions-page__content {
    padding: 24px 16px;
  }

  .questions-page__bar {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }
}
</style>
