package state

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/kamir/memory-connector/pkg/models"
	"go.uber.org/zap"
)

// JSONStore implements StateManager using JSON files
type JSONStore struct {
	dirPath string
	logger  *zap.Logger
	mu      sync.RWMutex
}

// NewJSONStore creates a new JSON-based state store
func NewJSONStore(dirPath string, logger *zap.Logger) (*JSONStore, error) {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create state directory: %w", err)
	}

	logger.Info("Initialized JSON state store", zap.String("path", dirPath))

	return &JSONStore{
		dirPath: dirPath,
		logger:  logger,
	}, nil
}

// GetState retrieves the sync state for a connector
func (s *JSONStore) GetState(ctx context.Context, connectorID string) (*models.SyncState, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	filePath := s.getFilePath(connectorID)

	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		// Return empty state
		return &models.SyncState{
			ConnectorID:  connectorID,
			ProcessedIDs: make(map[string]bool),
		}, nil
	}

	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read state file: %w", err)
	}

	// Unmarshal JSON
	var state models.SyncState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("failed to unmarshal state: %w", err)
	}

	s.logger.Debug("Retrieved state from JSON",
		zap.String("connector_id", connectorID),
		zap.Int("processed_count", len(state.ProcessedIDs)),
	)

	return &state, nil
}

// SaveState saves the sync state for a connector
func (s *JSONStore) SaveState(ctx context.Context, state *models.SyncState) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	filePath := s.getFilePath(state.ConnectorID)

	// Marshal to JSON with indentation for readability
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal state: %w", err)
	}

	// Write to temporary file first
	tmpPath := filePath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write state file: %w", err)
	}

	// Atomic rename
	if err := os.Rename(tmpPath, filePath); err != nil {
		return fmt.Errorf("failed to rename state file: %w", err)
	}

	s.logger.Debug("Saved state to JSON",
		zap.String("connector_id", state.ConnectorID),
		zap.Int("processed_count", len(state.ProcessedIDs)),
	)

	return nil
}

// DeleteState removes the sync state for a connector
func (s *JSONStore) DeleteState(ctx context.Context, connectorID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	filePath := s.getFilePath(connectorID)

	if err := os.Remove(filePath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete state file: %w", err)
	}

	s.logger.Info("Deleted state", zap.String("connector_id", connectorID))

	return nil
}

// ListStates lists all connector states
func (s *JSONStore) ListStates(ctx context.Context) ([]models.SyncState, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries, err := os.ReadDir(s.dirPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read state directory: %w", err)
	}

	var states []models.SyncState

	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		filePath := filepath.Join(s.dirPath, entry.Name())
		data, err := os.ReadFile(filePath)
		if err != nil {
			s.logger.Warn("Failed to read state file", zap.String("file", entry.Name()), zap.Error(err))
			continue
		}

		var state models.SyncState
		if err := json.Unmarshal(data, &state); err != nil {
			s.logger.Warn("Failed to unmarshal state", zap.String("file", entry.Name()), zap.Error(err))
			continue
		}

		states = append(states, state)
	}

	s.logger.Debug("Listed states", zap.Int("count", len(states)))

	return states, nil
}

// Close closes the JSON store (no-op for JSON)
func (s *JSONStore) Close() error {
	return nil
}

// getFilePath returns the file path for a connector's state
func (s *JSONStore) getFilePath(connectorID string) string {
	return filepath.Join(s.dirPath, fmt.Sprintf("%s.json", connectorID))
}
