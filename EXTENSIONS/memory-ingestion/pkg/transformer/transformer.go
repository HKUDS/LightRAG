package transformer

import (
	"fmt"

	"github.com/kamir/memory-connector/pkg/models"
	"go.uber.org/zap"
)

// Transformer converts Memory API data to LightRAG document format
type Transformer struct {
	strategy Strategy
	logger   *zap.Logger
}

// Strategy defines the interface for transformation strategies
type Strategy interface {
	Transform(memory *models.Memory, config TransformConfig) (string, map[string]string, error)
	Name() string
}

// TransformConfig holds configuration for transformation
type TransformConfig struct {
	IncludeMetadata bool
	EnrichLocation  bool
	ContextID       string
}

// NewTransformer creates a new transformer with the specified strategy
func NewTransformer(strategyName string, logger *zap.Logger) (*Transformer, error) {
	var strategy Strategy

	switch strategyName {
	case "standard":
		strategy = &StandardStrategy{}
	case "rich":
		strategy = &RichStrategy{}
	default:
		return nil, fmt.Errorf("unknown transformation strategy: %s", strategyName)
	}

	logger.Info("Initialized transformer", zap.String("strategy", strategy.Name()))

	return &Transformer{
		strategy: strategy,
		logger:   logger,
	}, nil
}

// Transform converts a memory to LightRAG document format
func (t *Transformer) Transform(memory *models.Memory, config TransformConfig) (string, map[string]string, error) {
	t.logger.Debug("Transforming memory",
		zap.String("memory_id", memory.ID),
		zap.String("strategy", t.strategy.Name()),
	)

	text, metadata, err := t.strategy.Transform(memory, config)
	if err != nil {
		return "", nil, fmt.Errorf("transformation failed: %w", err)
	}

	t.logger.Debug("Transformation complete",
		zap.String("memory_id", memory.ID),
		zap.Int("text_length", len(text)),
		zap.Int("metadata_count", len(metadata)),
	)

	return text, metadata, nil
}

// TransformBatch transforms multiple memories
func (t *Transformer) TransformBatch(memories []models.Memory, config TransformConfig) ([]TransformResult, error) {
	results := make([]TransformResult, 0, len(memories))

	for i := range memories {
		text, metadata, err := t.Transform(&memories[i], config)

		result := TransformResult{
			MemoryID: memories[i].ID,
			Text:     text,
			Metadata: metadata,
			Error:    err,
		}

		results = append(results, result)

		if err != nil {
			t.logger.Warn("Failed to transform memory",
				zap.String("memory_id", memories[i].ID),
				zap.Error(err),
			)
		}
	}

	return results, nil
}

// TransformResult holds the result of a single transformation
type TransformResult struct {
	MemoryID string
	Text     string
	Metadata map[string]string
	Error    error
}

// IsSuccess returns true if the transformation was successful
func (r *TransformResult) IsSuccess() bool {
	return r.Error == nil
}
