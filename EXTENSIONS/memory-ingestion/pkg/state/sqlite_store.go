package state

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/kamir/memory-connector/pkg/models"
	"go.uber.org/zap"
	_ "modernc.org/sqlite" // Pure Go SQLite driver (as per user's answer)
)

// SQLiteStore implements StateManager using SQLite
type SQLiteStore struct {
	db     *sql.DB
	logger *zap.Logger
}

// NewSQLiteStore creates a new SQLite-based state store
func NewSQLiteStore(dbPath string, logger *zap.Logger) (*SQLiteStore, error) {
	// Open database using pure Go driver (modernc.org/sqlite)
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Set connection pool settings
	db.SetMaxOpenConns(1) // SQLite works best with single connection
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(time.Hour)

	store := &SQLiteStore{
		db:     db,
		logger: logger,
	}

	// Initialize schema
	if err := store.initSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	logger.Info("Initialized SQLite state store", zap.String("path", dbPath))

	return store, nil
}

// initSchema creates the necessary tables
func (s *SQLiteStore) initSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS sync_states (
		connector_id TEXT PRIMARY KEY,
		context_id TEXT NOT NULL,
		last_sync_time TIMESTAMP,
		processed_ids TEXT, -- JSON array of processed memory IDs
		last_sync_report TEXT, -- JSON serialized SyncReport
		failed_items TEXT, -- JSON array of FailedItem
		total_sync_count INTEGER DEFAULT 0,
		updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE INDEX IF NOT EXISTS idx_context_id ON sync_states(context_id);
	CREATE INDEX IF NOT EXISTS idx_updated_at ON sync_states(updated_at);
	`

	_, err := s.db.Exec(schema)
	if err != nil {
		return fmt.Errorf("failed to create schema: %w", err)
	}

	return nil
}

// GetState retrieves the sync state for a connector
func (s *SQLiteStore) GetState(ctx context.Context, connectorID string) (*models.SyncState, error) {
	query := `
		SELECT connector_id, context_id, last_sync_time, processed_ids,
		       last_sync_report, failed_items, total_sync_count, updated_at
		FROM sync_states
		WHERE connector_id = ?
	`

	var state models.SyncState
	var lastSyncTime sql.NullTime
	var processedIDsJSON, lastSyncReportJSON, failedItemsJSON sql.NullString
	var updatedAt time.Time

	err := s.db.QueryRowContext(ctx, query, connectorID).Scan(
		&state.ConnectorID,
		&state.ContextID,
		&lastSyncTime,
		&processedIDsJSON,
		&lastSyncReportJSON,
		&failedItemsJSON,
		&state.TotalSyncCount,
		&updatedAt,
	)

	if err == sql.ErrNoRows {
		// Return empty state
		return &models.SyncState{
			ConnectorID:  connectorID,
			ProcessedIDs: make(map[string]bool),
		}, nil
	}

	if err != nil {
		return nil, fmt.Errorf("failed to query state: %w", err)
	}

	// Parse nullable fields
	if lastSyncTime.Valid {
		state.LastSyncTime = lastSyncTime.Time
	}
	state.UpdatedAt = updatedAt

	// Unmarshal JSON fields
	if processedIDsJSON.Valid && processedIDsJSON.String != "" {
		var processedIDs map[string]bool
		if err := json.Unmarshal([]byte(processedIDsJSON.String), &processedIDs); err != nil {
			s.logger.Warn("Failed to unmarshal processed_ids", zap.Error(err))
			state.ProcessedIDs = make(map[string]bool)
		} else {
			state.ProcessedIDs = processedIDs
		}
	} else {
		state.ProcessedIDs = make(map[string]bool)
	}

	if lastSyncReportJSON.Valid && lastSyncReportJSON.String != "" {
		var report models.SyncReport
		if err := json.Unmarshal([]byte(lastSyncReportJSON.String), &report); err != nil {
			s.logger.Warn("Failed to unmarshal last_sync_report", zap.Error(err))
		} else {
			state.LastSyncReport = &report
		}
	}

	if failedItemsJSON.Valid && failedItemsJSON.String != "" {
		var failedItems []models.FailedItem
		if err := json.Unmarshal([]byte(failedItemsJSON.String), &failedItems); err != nil {
			s.logger.Warn("Failed to unmarshal failed_items", zap.Error(err))
		} else {
			state.FailedItems = failedItems
		}
	}

	s.logger.Debug("Retrieved state from SQLite",
		zap.String("connector_id", connectorID),
		zap.Int("processed_count", len(state.ProcessedIDs)),
	)

	return &state, nil
}

// SaveState saves the sync state for a connector
func (s *SQLiteStore) SaveState(ctx context.Context, state *models.SyncState) error {
	// Marshal JSON fields
	processedIDsJSON, err := json.Marshal(state.ProcessedIDs)
	if err != nil {
		return fmt.Errorf("failed to marshal processed_ids: %w", err)
	}

	var lastSyncReportJSON []byte
	if state.LastSyncReport != nil {
		lastSyncReportJSON, err = json.Marshal(state.LastSyncReport)
		if err != nil {
			return fmt.Errorf("failed to marshal last_sync_report: %w", err)
		}
	}

	var failedItemsJSON []byte
	if state.FailedItems != nil {
		failedItemsJSON, err = json.Marshal(state.FailedItems)
		if err != nil {
			return fmt.Errorf("failed to marshal failed_items: %w", err)
		}
	}

	query := `
		INSERT INTO sync_states
			(connector_id, context_id, last_sync_time, processed_ids,
			 last_sync_report, failed_items, total_sync_count, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(connector_id) DO UPDATE SET
			context_id = excluded.context_id,
			last_sync_time = excluded.last_sync_time,
			processed_ids = excluded.processed_ids,
			last_sync_report = excluded.last_sync_report,
			failed_items = excluded.failed_items,
			total_sync_count = excluded.total_sync_count,
			updated_at = excluded.updated_at
	`

	_, err = s.db.ExecContext(ctx, query,
		state.ConnectorID,
		state.ContextID,
		state.LastSyncTime,
		string(processedIDsJSON),
		string(lastSyncReportJSON),
		string(failedItemsJSON),
		state.TotalSyncCount,
		time.Now(),
	)

	if err != nil {
		return fmt.Errorf("failed to save state: %w", err)
	}

	s.logger.Debug("Saved state to SQLite",
		zap.String("connector_id", state.ConnectorID),
		zap.Int("processed_count", len(state.ProcessedIDs)),
	)

	return nil
}

// DeleteState removes the sync state for a connector
func (s *SQLiteStore) DeleteState(ctx context.Context, connectorID string) error {
	query := `DELETE FROM sync_states WHERE connector_id = ?`

	_, err := s.db.ExecContext(ctx, query, connectorID)
	if err != nil {
		return fmt.Errorf("failed to delete state: %w", err)
	}

	s.logger.Info("Deleted state", zap.String("connector_id", connectorID))

	return nil
}

// ListStates lists all connector states
func (s *SQLiteStore) ListStates(ctx context.Context) ([]models.SyncState, error) {
	query := `
		SELECT connector_id, context_id, last_sync_time, processed_ids,
		       last_sync_report, failed_items, total_sync_count, updated_at
		FROM sync_states
		ORDER BY updated_at DESC
	`

	rows, err := s.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query states: %w", err)
	}
	defer rows.Close()

	var states []models.SyncState

	for rows.Next() {
		var state models.SyncState
		var lastSyncTime sql.NullTime
		var processedIDsJSON, lastSyncReportJSON, failedItemsJSON sql.NullString
		var updatedAt time.Time

		err := rows.Scan(
			&state.ConnectorID,
			&state.ContextID,
			&lastSyncTime,
			&processedIDsJSON,
			&lastSyncReportJSON,
			&failedItemsJSON,
			&state.TotalSyncCount,
			&updatedAt,
		)

		if err != nil {
			s.logger.Warn("Failed to scan state row", zap.Error(err))
			continue
		}

		if lastSyncTime.Valid {
			state.LastSyncTime = lastSyncTime.Time
		}
		state.UpdatedAt = updatedAt

		// Unmarshal JSON fields
		if processedIDsJSON.Valid {
			json.Unmarshal([]byte(processedIDsJSON.String), &state.ProcessedIDs)
		}
		if state.ProcessedIDs == nil {
			state.ProcessedIDs = make(map[string]bool)
		}

		if lastSyncReportJSON.Valid {
			var report models.SyncReport
			if err := json.Unmarshal([]byte(lastSyncReportJSON.String), &report); err == nil {
				state.LastSyncReport = &report
			}
		}

		if failedItemsJSON.Valid {
			json.Unmarshal([]byte(failedItemsJSON.String), &state.FailedItems)
		}

		states = append(states, state)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating states: %w", err)
	}

	s.logger.Debug("Listed states", zap.Int("count", len(states)))

	return states, nil
}

// Close closes the database connection
func (s *SQLiteStore) Close() error {
	return s.db.Close()
}
