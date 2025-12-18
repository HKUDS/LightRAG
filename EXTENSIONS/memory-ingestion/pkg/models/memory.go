package models

import (
	"time"
)

// Memory represents a memory item from the Memory API
type Memory struct {
	ID          string    `json:"id" yaml:"id"`
	Type        string    `json:"type" yaml:"type"`
	Audio       *string   `json:"audio,omitempty" yaml:"audio,omitempty"`
	Image       *string   `json:"image,omitempty" yaml:"image,omitempty"`
	Transcript  string    `json:"transcript" yaml:"transcript"`
	LocationLat *float64  `json:"location_lat,omitempty" yaml:"location_lat,omitempty"`
	LocationLon *float64  `json:"location_lon,omitempty" yaml:"location_lon,omitempty"`
	CreatedAt   string    `json:"created_at" yaml:"created_at"`
	UpdatedAt   *string   `json:"updated_at,omitempty" yaml:"updated_at,omitempty"`
}

// MemoryList represents a list of memories from the API
type MemoryList struct {
	Memories []Memory `json:"memories" yaml:"memories"`
	Count    int      `json:"count" yaml:"count"`
}

// ParseCreatedAt parses the CreatedAt timestamp into a time.Time object
func (m *Memory) ParseCreatedAt() (time.Time, error) {
	// Try RFC3339 format first
	t, err := time.Parse(time.RFC3339, m.CreatedAt)
	if err == nil {
		return t, nil
	}

	// Try ISO8601 without timezone
	t, err = time.Parse("2006-01-02T15:04:05", m.CreatedAt)
	if err == nil {
		return t, nil
	}

	return time.Time{}, err
}

// HasLocation returns true if the memory has location data
func (m *Memory) HasLocation() bool {
	return m.LocationLat != nil && m.LocationLon != nil
}

// HasAudio returns true if the memory has audio data
func (m *Memory) HasAudio() bool {
	return m.Audio != nil && *m.Audio != ""
}

// HasImage returns true if the memory has image data
func (m *Memory) HasImage() bool {
	return m.Image != nil && *m.Image != ""
}
