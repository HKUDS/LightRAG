package models

import (
	"fmt"
	"time"
)

// ConnectorConfig represents a single memory ingestion connector
type ConnectorConfig struct {
	ID         string            `json:"id" yaml:"id" validate:"required"`
	Enabled    bool              `json:"enabled" yaml:"enabled"`
	ContextID  string            `json:"context_id" yaml:"context_id" validate:"required"`
	Schedule   ScheduleConfig    `json:"schedule" yaml:"schedule"`
	Ingestion  IngestionConfig   `json:"ingestion" yaml:"ingestion"`
	Transform  TransformConfig   `json:"transform" yaml:"transform"`
	Metadata   map[string]string `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

// ScheduleConfig defines when the connector should run
type ScheduleConfig struct {
	Type          string `json:"type" yaml:"type" validate:"required,oneof=interval cron manual"`
	IntervalHours int    `json:"interval_hours,omitempty" yaml:"interval_hours,omitempty"`
	CronExpr      string `json:"cron_expr,omitempty" yaml:"cron_expr,omitempty"`
}

// IngestionConfig defines what data to pull
type IngestionConfig struct {
	QueryRange      string `json:"query_range" yaml:"query_range" validate:"required"`
	QueryLimit      int    `json:"query_limit" yaml:"query_limit" validate:"min=1,max=1000"`
	IncludeAudio    bool   `json:"include_audio" yaml:"include_audio"`
	IncludeImages   bool   `json:"include_images" yaml:"include_images"`
	MaxConcurrency  int    `json:"max_concurrency" yaml:"max_concurrency" validate:"min=1,max=50"`
}

// TransformConfig defines transformation options
type TransformConfig struct {
	Strategy       string `json:"strategy" yaml:"strategy" validate:"required,oneof=standard rich"`
	IncludeMetadata bool  `json:"include_metadata" yaml:"include_metadata"`
	EnrichLocation bool   `json:"enrich_location" yaml:"enrich_location"`
}

// ConnectorStatus represents the current state of a connector
type ConnectorStatus struct {
	ConnectorID    string         `json:"connector_id"`
	State          string         `json:"state"` // idle, running, paused, error
	LastSyncTime   *time.Time     `json:"last_sync_time,omitempty"`
	NextSyncTime   *time.Time     `json:"next_sync_time,omitempty"`
	LastSyncReport *SyncReport    `json:"last_sync_report,omitempty"`
	ErrorMessage   string         `json:"error_message,omitempty"`
}

// Validate checks if the connector configuration is valid
func (c *ConnectorConfig) Validate() error {
	if c.ID == "" {
		return fmt.Errorf("connector ID is required")
	}
	if c.ContextID == "" {
		return fmt.Errorf("context_id is required")
	}

	// Validate schedule
	switch c.Schedule.Type {
	case "interval":
		if c.Schedule.IntervalHours <= 0 {
			return fmt.Errorf("interval_hours must be positive")
		}
	case "cron":
		if c.Schedule.CronExpr == "" {
			return fmt.Errorf("cron_expr is required for cron schedule type")
		}
	case "manual":
		// No additional validation needed
	default:
		return fmt.Errorf("invalid schedule type: %s (must be interval, cron, or manual)", c.Schedule.Type)
	}

	// Validate ingestion config
	if c.Ingestion.QueryLimit <= 0 {
		c.Ingestion.QueryLimit = 100 // Default
	}
	if c.Ingestion.MaxConcurrency <= 0 {
		c.Ingestion.MaxConcurrency = 5 // Default from user's answer: configurable
	}

	return nil
}

// GetScheduleDescription returns a human-readable schedule description
func (c *ConnectorConfig) GetScheduleDescription() string {
	switch c.Schedule.Type {
	case "interval":
		return fmt.Sprintf("Every %d hour(s)", c.Schedule.IntervalHours)
	case "cron":
		return fmt.Sprintf("Cron: %s", c.Schedule.CronExpr)
	case "manual":
		return "Manual trigger only"
	default:
		return "Unknown"
	}
}
