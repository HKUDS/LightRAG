package orchestrator

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/kamir/memory-connector/pkg/client"
	"github.com/kamir/memory-connector/pkg/models"
	"github.com/kamir/memory-connector/pkg/state"
	"github.com/kamir/memory-connector/pkg/transformer"
	"go.uber.org/zap"
)

// Orchestrator coordinates the memory ingestion process
type Orchestrator struct {
	memoryClient  *client.MemoryClient
	lightragClient *client.LightRAGClient
	transformer   *transformer.Transformer
	stateManager  state.StateManager
	logger        *zap.Logger
}

// NewOrchestrator creates a new orchestrator
func NewOrchestrator(
	memoryClient *client.MemoryClient,
	lightragClient *client.LightRAGClient,
	transformer *transformer.Transformer,
	stateManager state.StateManager,
	logger *zap.Logger,
) *Orchestrator {
	return &Orchestrator{
		memoryClient:   memoryClient,
		lightragClient: lightragClient,
		transformer:    transformer,
		stateManager:   stateManager,
		logger:         logger,
	}
}

// SyncConnector performs a full sync for a connector
func (o *Orchestrator) SyncConnector(ctx context.Context, config *models.ConnectorConfig) (*models.SyncReport, error) {
	o.logger.Info("Starting sync",
		zap.String("connector_id", config.ID),
		zap.String("context_id", config.ContextID),
	)

	report := &models.SyncReport{
		ConnectorID: config.ID,
		ContextID:   config.ContextID,
		StartTime:   time.Now(),
		Status:      "success",
		Metrics:     models.SyncMetrics{},
	}

	// Get current state
	syncState, err := o.stateManager.GetState(ctx, config.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to get sync state: %w", err)
	}

	// If state doesn't have context ID, set it
	if syncState.ContextID == "" {
		syncState.ContextID = config.ContextID
	}

	// Fetch memories from Memory API
	fetchStart := time.Now()
	memoryList, err := o.memoryClient.GetMemories(
		ctx,
		config.ContextID,
		config.Ingestion.QueryLimit,
		config.Ingestion.QueryRange,
	)
	if err != nil {
		report.Status = "failed"
		report.ErrorMessage = fmt.Sprintf("Failed to fetch memories: %v", err)
		report.EndTime = time.Now()
		report.Duration = report.EndTime.Sub(report.StartTime)
		return report, fmt.Errorf("failed to fetch memories: %w", err)
	}
	fetchDuration := time.Since(fetchStart)

	report.TotalFetched = len(memoryList.Memories)
	o.logger.Info("Fetched memories",
		zap.Int("count", report.TotalFetched),
		zap.Duration("duration", fetchDuration),
	)

	if report.TotalFetched > 0 {
		report.Metrics.AvgFetchTimeMs = fetchDuration.Milliseconds() / int64(report.TotalFetched)
	}

	// Filter out already-processed memories
	newMemories := make([]models.Memory, 0)
	for _, memory := range memoryList.Memories {
		if !syncState.IsProcessed(memory.ID) {
			newMemories = append(newMemories, memory)
		} else {
			report.TotalSkipped++
			report.MemoriesSkipped = append(report.MemoriesSkipped, memory.ID)
		}
	}

	o.logger.Info("Filtered memories",
		zap.Int("new", len(newMemories)),
		zap.Int("skipped", report.TotalSkipped),
	)

	// Process new memories with concurrency control (as per user's answer: configurable)
	if len(newMemories) > 0 {
		err = o.processMemoriesConcurrent(ctx, newMemories, config, syncState, report)
		if err != nil && report.TotalProcessed == 0 {
			// Complete failure
			report.Status = "failed"
			report.ErrorMessage = fmt.Sprintf("Failed to process memories: %v", err)
		} else if report.TotalFailed > 0 {
			// Partial success (as per user's answer: "Process what we got and track what was lost")
			report.Status = "partial"
		}
	}

	// Update state
	syncState.LastSyncTime = time.Now()
	syncState.LastSyncReport = report
	syncState.TotalSyncCount++
	syncState.UpdatedAt = time.Now()

	if err := o.stateManager.SaveState(ctx, syncState); err != nil {
		o.logger.Error("Failed to save state", zap.Error(err))
		// Don't fail the entire sync just because we couldn't save state
	}

	report.EndTime = time.Now()
	report.Duration = report.EndTime.Sub(report.StartTime)

	o.logger.Info("Sync completed",
		zap.String("connector_id", config.ID),
		zap.String("status", report.Status),
		zap.Int("processed", report.TotalProcessed),
		zap.Int("failed", report.TotalFailed),
		zap.Duration("duration", report.Duration),
	)

	return report, nil
}

// processMemoriesConcurrent processes memories with concurrency control
func (o *Orchestrator) processMemoriesConcurrent(
	ctx context.Context,
	memories []models.Memory,
	config *models.ConnectorConfig,
	syncState *models.SyncState,
	report *models.SyncReport,
) error {
	// Create semaphore for concurrency control (as per user's answer: configurable)
	semaphore := make(chan struct{}, config.Ingestion.MaxConcurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex

	transformConfig := transformer.TransformConfig{
		IncludeMetadata: config.Transform.IncludeMetadata,
		EnrichLocation:  config.Transform.EnrichLocation,
		ContextID:       config.ContextID,
	}

	for i := range memories {
		wg.Add(1)
		go func(memory models.Memory) {
			defer wg.Done()

			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			// Process individual memory
			err := o.processMemory(ctx, &memory, transformConfig)

			// Update report (thread-safe)
			mu.Lock()
			defer mu.Unlock()

			if err != nil {
				report.TotalFailed++
				failedItem := models.FailedItem{
					MemoryID:     memory.ID,
					ErrorMessage: err.Error(),
					FailedAt:     time.Now(),
					Retryable:    true,
					RetryCount:   0,
				}
				report.MemoriesFailed = append(report.MemoriesFailed, failedItem)
				syncState.AddFailedItem(failedItem)

				o.logger.Warn("Failed to process memory",
					zap.String("memory_id", memory.ID),
					zap.Error(err),
				)
			} else {
				report.TotalProcessed++
				report.MemoriesIngested = append(report.MemoriesIngested, memory.ID)
				syncState.MarkProcessed(memory.ID)

				o.logger.Debug("Processed memory", zap.String("memory_id", memory.ID))
			}
		}(memories[i])
	}

	wg.Wait()
	return nil
}

// processMemory processes a single memory
func (o *Orchestrator) processMemory(
	ctx context.Context,
	memory *models.Memory,
	transformConfig transformer.TransformConfig,
) error {
	// Transform memory to LightRAG document format
	transformStart := time.Now()
	text, metadata, err := o.transformer.Transform(memory, transformConfig)
	if err != nil {
		return fmt.Errorf("transformation failed: %w", err)
	}
	transformDuration := time.Since(transformStart)

	// Insert document into LightRAG
	insertStart := time.Now()
	_, err = o.lightragClient.InsertDocument(ctx, text, metadata)
	if err != nil {
		return fmt.Errorf("insertion failed: %w", err)
	}
	insertDuration := time.Since(insertStart)

	o.logger.Debug("Memory processed",
		zap.String("memory_id", memory.ID),
		zap.Duration("transform_time", transformDuration),
		zap.Duration("insert_time", insertDuration),
	)

	return nil
}
