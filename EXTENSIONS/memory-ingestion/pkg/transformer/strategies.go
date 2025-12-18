package transformer

import (
	"fmt"
	"strings"

	"github.com/kamir/memory-connector/pkg/models"
)

// StandardStrategy provides basic transformation of memory to text
type StandardStrategy struct{}

// Name returns the strategy name
func (s *StandardStrategy) Name() string {
	return "standard"
}

// Transform converts a memory to a simple text format
func (s *StandardStrategy) Transform(memory *models.Memory, config TransformConfig) (string, map[string]string, error) {
	if memory.Transcript == "" {
		return "", nil, fmt.Errorf("memory %s has no transcript", memory.ID)
	}

	// Build text content
	var builder strings.Builder
	builder.WriteString(memory.Transcript)

	// Build metadata
	metadata := make(map[string]string)

	if config.IncludeMetadata {
		metadata["memory_id"] = memory.ID
		metadata["memory_type"] = memory.Type
		metadata["created_at"] = memory.CreatedAt
		metadata["context_id"] = config.ContextID

		if memory.HasLocation() && config.EnrichLocation {
			metadata["location_lat"] = fmt.Sprintf("%f", *memory.LocationLat)
			metadata["location_lon"] = fmt.Sprintf("%f", *memory.LocationLon)
		}

		if memory.HasAudio() {
			metadata["has_audio"] = "true"
		}

		if memory.HasImage() {
			metadata["has_image"] = "true"
		}
	}

	return builder.String(), metadata, nil
}

// RichStrategy provides enriched transformation with contextual information
type RichStrategy struct{}

// Name returns the strategy name
func (s *RichStrategy) Name() string {
	return "rich"
}

// Transform converts a memory to a rich, context-enhanced format
func (s *RichStrategy) Transform(memory *models.Memory, config TransformConfig) (string, map[string]string, error) {
	if memory.Transcript == "" {
		return "", nil, fmt.Errorf("memory %s has no transcript", memory.ID)
	}

	// Build rich text content with contextual information
	var builder strings.Builder

	// Add temporal context
	parsedTime, err := memory.ParseCreatedAt()
	if err == nil {
		builder.WriteString(fmt.Sprintf("[Memory from %s]\n\n", parsedTime.Format("2006-01-02 15:04:05")))
	}

	// Add location context if available
	if memory.HasLocation() && config.EnrichLocation {
		builder.WriteString(fmt.Sprintf("[Location: %.6f, %.6f]\n\n", *memory.LocationLat, *memory.LocationLon))
	}

	// Add media availability context
	mediaInfo := []string{}
	if memory.HasAudio() {
		mediaInfo = append(mediaInfo, "audio recording available")
	}
	if memory.HasImage() {
		mediaInfo = append(mediaInfo, "image available")
	}
	if len(mediaInfo) > 0 {
		builder.WriteString(fmt.Sprintf("[Media: %s]\n\n", strings.Join(mediaInfo, ", ")))
	}

	// Add the main transcript
	builder.WriteString("Transcript:\n")
	builder.WriteString(memory.Transcript)
	builder.WriteString("\n")

	// Add memory type context
	if memory.Type != "" {
		builder.WriteString(fmt.Sprintf("\n[Type: %s]", memory.Type))
	}

	// Build metadata (similar to standard but with additional enrichments)
	metadata := make(map[string]string)

	if config.IncludeMetadata {
		metadata["memory_id"] = memory.ID
		metadata["memory_type"] = memory.Type
		metadata["created_at"] = memory.CreatedAt
		metadata["context_id"] = config.ContextID
		metadata["transformation_strategy"] = "rich"

		if memory.HasLocation() {
			metadata["location_lat"] = fmt.Sprintf("%f", *memory.LocationLat)
			metadata["location_lon"] = fmt.Sprintf("%f", *memory.LocationLon)

			// Add enrichment flag
			if config.EnrichLocation {
				metadata["location_enriched"] = "true"
			}
		}

		if memory.HasAudio() {
			metadata["has_audio"] = "true"
			if memory.Audio != nil {
				metadata["audio_reference"] = *memory.Audio
			}
		}

		if memory.HasImage() {
			metadata["has_image"] = "true"
			if memory.Image != nil {
				metadata["image_reference"] = *memory.Image
			}
		}

		// Add temporal metadata
		if parsedTime, err := memory.ParseCreatedAt(); err == nil {
			metadata["year"] = fmt.Sprintf("%d", parsedTime.Year())
			metadata["month"] = fmt.Sprintf("%d", parsedTime.Month())
			metadata["day"] = fmt.Sprintf("%d", parsedTime.Day())
			metadata["hour"] = fmt.Sprintf("%d", parsedTime.Hour())
			metadata["weekday"] = parsedTime.Weekday().String()
		}
	}

	return builder.String(), metadata, nil
}
