package scheduler

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/kamir/memory-connector/pkg/models"
	"github.com/kamir/memory-connector/pkg/orchestrator"
	"github.com/robfig/cron/v3"
	"go.uber.org/zap"
)

// Scheduler manages scheduled sync jobs
type Scheduler struct {
	cron         *cron.Cron
	orchestrator *orchestrator.Orchestrator
	logger       *zap.Logger
	jobs         map[string]cron.EntryID // connector ID -> cron entry ID
	mu           sync.RWMutex
	ctx          context.Context
	cancel       context.CancelFunc
}

// NewScheduler creates a new scheduler
func NewScheduler(orchestrator *orchestrator.Orchestrator, logger *zap.Logger) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())

	return &Scheduler{
		cron:         cron.New(cron.WithSeconds()),
		orchestrator: orchestrator,
		logger:       logger,
		jobs:         make(map[string]cron.EntryID),
		ctx:          ctx,
		cancel:       cancel,
	}
}

// Start starts the scheduler
func (s *Scheduler) Start() {
	s.cron.Start()
	s.logger.Info("Scheduler started")
}

// Stop stops the scheduler and waits for running jobs to complete
func (s *Scheduler) Stop() {
	s.logger.Info("Stopping scheduler...")
	s.cancel()

	ctx := s.cron.Stop()
	<-ctx.Done()

	s.logger.Info("Scheduler stopped")
}

// AddConnector adds a connector to the schedule
func (s *Scheduler) AddConnector(config *models.ConnectorConfig) error {
	if !config.Enabled {
		s.logger.Info("Connector is disabled, skipping scheduling",
			zap.String("connector_id", config.ID),
		)
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Remove existing job if present
	if entryID, exists := s.jobs[config.ID]; exists {
		s.cron.Remove(entryID)
		delete(s.jobs, config.ID)
		s.logger.Info("Removed existing job for connector",
			zap.String("connector_id", config.ID),
		)
	}

	// Determine schedule based on type
	var schedule string
	switch config.Schedule.Type {
	case "interval":
		// Convert interval to cron expression
		// For hourly intervals, we use: 0 0 */N * * * (every N hours)
		schedule = fmt.Sprintf("0 0 */%d * * *", config.Schedule.IntervalHours)
	case "cron":
		schedule = config.Schedule.CronExpr
	case "manual":
		s.logger.Info("Connector is manual, not scheduling",
			zap.String("connector_id", config.ID),
		)
		return nil
	default:
		return fmt.Errorf("unsupported schedule type: %s", config.Schedule.Type)
	}

	// Create job function
	jobFunc := func() {
		s.runSync(config)
	}

	// Add job to cron
	entryID, err := s.cron.AddFunc(schedule, jobFunc)
	if err != nil {
		return fmt.Errorf("failed to add cron job: %w", err)
	}

	s.jobs[config.ID] = entryID

	s.logger.Info("Scheduled connector",
		zap.String("connector_id", config.ID),
		zap.String("schedule", schedule),
		zap.String("description", config.GetScheduleDescription()),
	)

	return nil
}

// RemoveConnector removes a connector from the schedule
func (s *Scheduler) RemoveConnector(connectorID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	entryID, exists := s.jobs[connectorID]
	if !exists {
		return fmt.Errorf("connector not found in schedule: %s", connectorID)
	}

	s.cron.Remove(entryID)
	delete(s.jobs, connectorID)

	s.logger.Info("Removed connector from schedule",
		zap.String("connector_id", connectorID),
	)

	return nil
}

// TriggerSync manually triggers a sync for a connector
func (s *Scheduler) TriggerSync(config *models.ConnectorConfig) (*models.SyncReport, error) {
	s.logger.Info("Manually triggering sync",
		zap.String("connector_id", config.ID),
	)

	return s.orchestrator.SyncConnector(s.ctx, config)
}

// runSync executes a sync job (called by cron)
func (s *Scheduler) runSync(config *models.ConnectorConfig) {
	s.logger.Info("Starting scheduled sync",
		zap.String("connector_id", config.ID),
		zap.String("context_id", config.ContextID),
	)

	report, err := s.orchestrator.SyncConnector(s.ctx, config)
	if err != nil {
		s.logger.Error("Scheduled sync failed",
			zap.String("connector_id", config.ID),
			zap.Error(err),
		)
		return
	}

	s.logger.Info("Scheduled sync completed",
		zap.String("connector_id", config.ID),
		zap.String("status", report.Status),
		zap.Int("processed", report.TotalProcessed),
		zap.Int("failed", report.TotalFailed),
		zap.Duration("duration", report.Duration),
		zap.Float64("success_rate", report.CalculateSuccessRate()),
	)
}

// GetScheduledJobs returns information about all scheduled jobs
func (s *Scheduler) GetScheduledJobs() map[string]JobInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]JobInfo)

	for connectorID, entryID := range s.jobs {
		entry := s.cron.Entry(entryID)
		result[connectorID] = JobInfo{
			ConnectorID: connectorID,
			EntryID:     int(entryID),
			NextRun:     entry.Next,
			PrevRun:     entry.Prev,
		}
	}

	return result
}

// JobInfo contains information about a scheduled job
type JobInfo struct {
	ConnectorID string    `json:"connector_id"`
	EntryID     int       `json:"entry_id"`
	NextRun     time.Time `json:"next_run"`
	PrevRun     time.Time `json:"prev_run,omitempty"`
}
