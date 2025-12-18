package models

import (
	"time"
)

// SyncReport represents the result of a sync operation
type SyncReport struct {
	ConnectorID      string        `json:"connector_id"`
	ContextID        string        `json:"context_id"`
	StartTime        time.Time     `json:"start_time"`
	EndTime          time.Time     `json:"end_time"`
	Duration         time.Duration `json:"duration"`
	Status           string        `json:"status"` // success, partial, failed
	TotalFetched     int           `json:"total_fetched"`
	TotalProcessed   int           `json:"total_processed"`
	TotalSkipped     int           `json:"total_skipped"`
	TotalFailed      int           `json:"total_failed"`
	MemoriesIngested []string      `json:"memories_ingested,omitempty"`
	MemoriesSkipped  []string      `json:"memories_skipped,omitempty"`
	MemoriesFailed   []FailedItem  `json:"memories_failed,omitempty"`
	ErrorMessage     string        `json:"error_message,omitempty"`
	Metrics          SyncMetrics   `json:"metrics"`
}

// FailedItem represents a memory that failed to process
// As per user's answer: "Process what we got and track what was lost and what went wrong, capture the errors like in a DLQ"
type FailedItem struct {
	MemoryID     string    `json:"memory_id"`
	ErrorMessage string    `json:"error_message"`
	FailedAt     time.Time `json:"failed_at"`
	Retryable    bool      `json:"retryable"`
	RetryCount   int       `json:"retry_count"`
}

// SyncMetrics contains performance metrics for a sync operation
type SyncMetrics struct {
	AvgFetchTimeMs    int64 `json:"avg_fetch_time_ms"`
	AvgTransformTimeMs int64 `json:"avg_transform_time_ms"`
	AvgInsertTimeMs   int64 `json:"avg_insert_time_ms"`
	TotalBytesProcessed int64 `json:"total_bytes_processed"`
}

// SyncHistory represents historical sync records
type SyncHistory struct {
	Reports []SyncReport `json:"reports"`
}

// SyncState tracks the state of a connector for idempotency
type SyncState struct {
	ConnectorID     string             `json:"connector_id"`
	ContextID       string             `json:"context_id"`
	LastSyncTime    time.Time          `json:"last_sync_time"`
	ProcessedIDs    map[string]bool    `json:"processed_ids"` // Set of memory IDs already processed
	LastSyncReport  *SyncReport        `json:"last_sync_report,omitempty"`
	FailedItems     []FailedItem       `json:"failed_items,omitempty"` // Dead Letter Queue
	TotalSyncCount  int                `json:"total_sync_count"`
	UpdatedAt       time.Time          `json:"updated_at"`
}

// IsProcessed checks if a memory ID has already been processed
func (s *SyncState) IsProcessed(memoryID string) bool {
	if s.ProcessedIDs == nil {
		return false
	}
	return s.ProcessedIDs[memoryID]
}

// MarkProcessed marks a memory ID as processed
func (s *SyncState) MarkProcessed(memoryID string) {
	if s.ProcessedIDs == nil {
		s.ProcessedIDs = make(map[string]bool)
	}
	s.ProcessedIDs[memoryID] = true
	s.UpdatedAt = time.Now()
}

// AddFailedItem adds a failed item to the DLQ
func (s *SyncState) AddFailedItem(item FailedItem) {
	s.FailedItems = append(s.FailedItems, item)
	s.UpdatedAt = time.Now()
}

// GetRetryableFailedItems returns items that can be retried
func (s *SyncState) GetRetryableFailedItems(maxRetries int) []FailedItem {
	var retryable []FailedItem
	for _, item := range s.FailedItems {
		if item.Retryable && item.RetryCount < maxRetries {
			retryable = append(retryable, item)
		}
	}
	return retryable
}

// CalculateSuccessRate calculates the success rate for a sync report
func (r *SyncReport) CalculateSuccessRate() float64 {
	if r.TotalFetched == 0 {
		return 0.0
	}
	return float64(r.TotalProcessed) / float64(r.TotalFetched) * 100.0
}

// IsSuccess returns true if the sync was successful
func (r *SyncReport) IsSuccess() bool {
	return r.Status == "success"
}

// IsPartial returns true if the sync was partially successful
func (r *SyncReport) IsPartial() bool {
	return r.Status == "partial"
}

// IsFailed returns true if the sync completely failed
func (r *SyncReport) IsFailed() bool {
	return r.Status == "failed"
}
