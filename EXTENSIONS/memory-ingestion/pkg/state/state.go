package state

import (
	"context"
	"fmt"

	"github.com/kamir/memory-connector/pkg/models"
	"go.uber.org/zap"
)

// StateManager defines the interface for state persistence
type StateManager interface {
	// GetState retrieves the sync state for a connector
	GetState(ctx context.Context, connectorID string) (*models.SyncState, error)

	// SaveState saves the sync state for a connector
	SaveState(ctx context.Context, state *models.SyncState) error

	// DeleteState removes the sync state for a connector
	DeleteState(ctx context.Context, connectorID string) error

	// ListStates lists all connector states
	ListStates(ctx context.Context) ([]models.SyncState, error)

	// Close closes the state manager
	Close() error
}

// Config holds state manager configuration
type Config struct {
	Type string // json or sqlite (as per user's answer: both in parallel)
	Path string // directory for json files or sqlite db path
}

// NewStateManager creates a new state manager based on configuration
func NewStateManager(config Config, logger *zap.Logger) (StateManager, error) {
	switch config.Type {
	case "json":
		return NewJSONStore(config.Path, logger)
	case "sqlite":
		return NewSQLiteStore(config.Path, logger)
	default:
		return nil, fmt.Errorf("unsupported state manager type: %s (must be 'json' or 'sqlite')", config.Type)
	}
}
