<template>
  <div class="questions-page">
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
                v-model="filterWorkspace"
                :items="workspaceOptions"
                item-title="title"
                item-value="value"
                label="Workspace"
                variant="solo"
                density="comfortable"
                hide-details
                class="mb-4"
                :menu-props="{ maxHeight: 240 }"
              />

              <v-select
                v-model="filterSession"
                :items="sessionOptions"
                item-title="title"
                item-value="value"
                label="Canvas"
                variant="solo"
                density="comfortable"
                hide-details
                class="mb-4"
                :menu-props="{ maxHeight: 240 }"
                :loading="sessionsLoading"
                :disabled="filterWorkspace === 'all'"
              />

              <v-select
                v-model="filterApproved"
                :items="approvalOptions"
                item-title="title"
                item-value="value"
                label="Approval status"
                variant="solo"
                density="comfortable"
                hide-details
                class="mb-4"
              />

              <v-select
                v-model="filterArchived"
                :items="archivedOptions"
                item-title="title"
                item-value="value"
                label="Archived"
                variant="solo"
                density="comfortable"
                hide-details
                class="mb-4"
              />

              <v-select
                v-model="filterHasVariants"
                :items="variantOptions"
                item-title="title"
                item-value="value"
                label="Variants"
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
                    v-if="question.type === 'mcq' || question.type === 'multiple_response'"
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

            <v-divider v-if="totalQuestions > 0" />

            <v-card-actions v-if="totalQuestions > 0" class="results-card__footer">
              <div class="results-card__footer-controls">
                <v-select
                  v-model="pageSize"
                  :items="pageSizeOptions"
                  item-title="title"
                  item-value="value"
                  label="Items per page"
                  variant="solo"
                  density="comfortable"
                  hide-details
                  class="results-card__page-size"
                />
              </div>
              <v-pagination
                v-model="currentPage"
                :length="pageCount"
                :disabled="pageCount <= 1"
                color="primary"
                variant="tonal"
                density="comfortable"
              />
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, watch } from 'vue';
import { storeToRefs } from 'pinia';
import MCQCard from '@/components/MCQCard.vue';
import AssignmentCard from '@/components/AssignmentCard.vue';
import { useQuestionsStore, useDashboardStore, useWorkspaceContextStore, useHeaderStore } from '@/stores';

interface DatabaseQuestion {
  id: string;
  type: 'mcq' | 'multiple_response' | 'assignment';
  question: string;
  options: string[] | null;
  correct_options: number[] | null;
  difficulty_level: 'easy' | 'medium' | 'hard';
  ai_rational: string;
  source: string;
  tag: string;
  tags?: string[];
  created_at: string;
  updated_at?: string;
  project_id?: string;
  session_id?: string;
  isApproved?: boolean;
  isArchived?: boolean;
  variants?: Array<{
    id?: string;
    difficulty_level?: string;
    options: string[];
    correct_answers: number[];
    rationale?: string;
  }>;
}

const questionsStore = useQuestionsStore();
const dashboardStore = useDashboardStore();
const workspaceContextStore = useWorkspaceContextStore();

const { loading, uniqueTags, filteredQuestions, resultsSummary, sessionsLoading } = storeToRefs(questionsStore);
const { workspaces } = storeToRefs(dashboardStore);
const totalQuestions = computed(() => questionsStore.totalQuestions);
const headerStore = useHeaderStore();

const searchQuery = computed({
  get: () => questionsStore.filters.searchQuery,
  set: (value: string) => questionsStore.setSearchQuery(value),
});

const filterType = computed({
  get: () => questionsStore.filters.type,
  set: (value: string) => questionsStore.setFilterType(value),
});

const filterDifficulty = computed({
  get: () => questionsStore.filters.difficulty,
  set: (value: string) => questionsStore.setFilterDifficulty(value),
});

const filterTag = computed({
  get: () => questionsStore.filters.tag,
  set: (value: string) => questionsStore.setFilterTag(value),
});

const filterWorkspace = computed({
  get: () => questionsStore.filters.workspaceId,
  set: (value: string) => questionsStore.setFilterWorkspace(value),
});

const filterSession = computed({
  get: () => questionsStore.filters.sessionId,
  set: (value: string) => questionsStore.setFilterSession(value),
});

const filterApproved = computed({
  get: () => questionsStore.filters.approved,
  set: (value: string) => questionsStore.setFilterApproved(value),
});

const filterArchived = computed({
  get: () => questionsStore.filters.archived,
  set: (value: string) => questionsStore.setFilterArchived(value),
});

const filterHasVariants = computed({
  get: () => questionsStore.filters.hasVariants,
  set: (value: string) => questionsStore.setFilterHasVariants(value),
});

const currentPage = computed({
  get: () => questionsStore.page,
  set: (value: number) => questionsStore.setPage(value),
});

const pageSize = computed({
  get: () => questionsStore.pageSize,
  set: (value: number) => questionsStore.setPageSize(value),
});

const pageCount = computed(() => questionsStore.pageCount);

const typeOptions = [
  { title: 'All types', value: 'all' },
  { title: 'Multiple Choice (single answer)', value: 'mcq' },
  { title: 'Multiple Response', value: 'multiple_response' },
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

const workspaceOptions = computed(() => {
  const items = Array.isArray(workspaces.value) ? workspaces.value : [];
  return [
    { title: 'All workspaces', value: 'all' },
    ...items.map((workspace) => ({ title: workspace.name, value: workspace.id })),
  ];
});

const sessionOptions = computed(() => {
  const options = questionsStore.sessionsForWorkspace(filterWorkspace.value);
  return [
    { title: 'All canvases', value: 'all' },
    ...options.map((session) => ({ title: session.name, value: session.id })),
  ];
});

const approvalOptions = [
  { title: 'All statuses', value: 'all' },
  { title: 'Approved', value: 'true' },
  { title: 'Not approved', value: 'false' },
];

const archivedOptions = [
  { title: 'Only active', value: 'false' },
  { title: 'All', value: 'all' },
  { title: 'Only archived', value: 'true' },
];

const variantOptions = [
  { title: 'All questions', value: 'all' },
  { title: 'With variants only', value: 'true' },
];

const pageSizeOptions = [
  { title: '20 per page', value: 20 },
  { title: '40 per page', value: 40 },
  { title: '60 per page', value: 60 },
  { title: '80 per page', value: 80 },
  { title: '100 per page', value: 100 },
];

const resetFilters = () => {
  questionsStore.resetFilters();
  scheduleFetch();
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
  variants: question.variants || [],
});

const convertToAssignment = (question: DatabaseQuestion) => ({
  id: question.id,
  question: question.question,
  difficultyLevel: question.difficulty_level,
  aiRational: question.ai_rational,
  source: question.source,
  tag: question.tag,
});

let fetchTimeout: ReturnType<typeof setTimeout> | null = null;
const scheduleFetch = (delay = 300) => {
  if (fetchTimeout) {
    clearTimeout(fetchTimeout);
  }
  fetchTimeout = setTimeout(() => {
    questionsStore.fetchQuestions();
  }, delay);
};

onUnmounted(() => {
  if (fetchTimeout) {
    clearTimeout(fetchTimeout);
    fetchTimeout = null;
  }
});

watch(searchQuery, () => {
  scheduleFetch(400);
});

watch([filterType, filterDifficulty, filterTag, filterApproved, filterArchived, filterSession, filterHasVariants], () => {
  scheduleFetch();
});

watch(
  filterWorkspace,
  async (value) => {
    await questionsStore.fetchSessionsForWorkspace(value);
    scheduleFetch();
  },
  { immediate: true }
);

watch(currentPage, () => {
  scheduleFetch(0);
});

watch(pageSize, () => {
  scheduleFetch(0);
});

onMounted(async () => {
  if (typeof dashboardStore.initialise === 'function') {
    await dashboardStore.initialise();
  }

  if (workspaceContextStore.workspaceId) {
    questionsStore.setFilterWorkspace(workspaceContextStore.workspaceId);
    await questionsStore.fetchSessionsForWorkspace(workspaceContextStore.workspaceId);
  }

  await questionsStore.fetchQuestions();
  headerStore.setHeader({
    title: 'Question Library',
    description: 'Browse and refine every prompt your team has generated.',
    showBack: true,
  });
});

onUnmounted(() => {
  if (fetchTimeout) {
    clearTimeout(fetchTimeout);
    fetchTimeout = null;
  }
  headerStore.resetHeader();
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

.questions-page__content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 32px 40px;
  overflow: hidden; /* keep internal scrolling */
}

.questions-page__grid {
  flex: 1;
  min-height: 0;            /* allow children to size/scroll */
  /* REMOVE gap to avoid forcing wrap */
  /* gap: 24px; */
}

/* Create the gutter without using row gap */
.questions-page__filters,
.questions-page__results {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

@media (min-width: 960px) {
  .questions-page__grid {
    flex-wrap: nowrap;      /* keep both columns side by side on md+ */
  }
  .questions-page__filters { padding-right: 12px; }
  .questions-page__results { padding-left: 12px; }
}

/* Cards */
.filters-card,
.results-card {
  border-radius: 24px;
  border: 1px solid rgba(15, 23, 42, 0.06);
  display: flex;
  flex-direction: column;
}

.questions-page__results .results-card {
  flex: 1 1 auto;   /* fills the right column */
  min-height: 0;    /* allow body to get height */
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
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  padding: 24px;
  overflow: hidden;  /* contain inner scroll */
  min-height: 0;     /* critical */
}

.results-card__list {
  flex: 1 1 auto;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding-right: 8px;
  min-height: 0;
}

.results-card__list > * {
  flex: 0 0 auto;
  align-self: stretch;
}

.results-card__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
  padding: 16px 24px;
}

.results-card__footer-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.results-card__page-size {
  min-width: 184px;
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

.state--loading { border-style: solid; }

@media (max-width: 960px) {
  .questions-page__content { padding: 24px 16px; }
  /* On small screens allow wrap + no side paddings */
  .questions-page__filters,
  .questions-page__results {
    padding: 0;
  }
}
</style>
