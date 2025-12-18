package config

import (
	"fmt"
	"os"

	"github.com/kamir/memory-connector/pkg/models"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

// Config represents the application configuration
type Config struct {
	Server     ServerConfig              `yaml:"server"`
	MemoryAPI  MemoryAPIConfig           `yaml:"memory_api"`
	LightRAG   LightRAGConfig            `yaml:"lightrag"`
	Logging    LoggingConfig             `yaml:"logging"`
	Storage    StorageConfig             `yaml:"storage"`
	Connectors []models.ConnectorConfig  `yaml:"connectors"`
}

// ServerConfig holds HTTP server configuration
type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

// MemoryAPIConfig holds Memory API client configuration
type MemoryAPIConfig struct {
	URL        string `yaml:"url" validate:"required,url"`
	APIKey     string `yaml:"api_key" validate:"required"`
	Timeout    int    `yaml:"timeout"`    // seconds
	MaxRetries int    `yaml:"max_retries"`
	RetryDelay int    `yaml:"retry_delay"` // seconds
}

// LightRAGConfig holds LightRAG API configuration
type LightRAGConfig struct {
	URL        string `yaml:"url" validate:"required,url"`
	Timeout    int    `yaml:"timeout"`    // seconds
	MaxRetries int    `yaml:"max_retries"`
	RetryDelay int    `yaml:"retry_delay"` // seconds
}

// LoggingConfig holds logging configuration
type LoggingConfig struct {
	Level      string `yaml:"level"`       // debug, info, warn, error
	Format     string `yaml:"format"`      // json or console (as per user's answer: both, configurable)
	OutputPath string `yaml:"output_path"` // file path or stdout
}

// StorageConfig holds state storage configuration
type StorageConfig struct {
	Type string `yaml:"type"` // json or sqlite (as per user's answer: both in parallel)
	Path string `yaml:"path"` // directory for json files or sqlite db path
}

// LoadConfig loads configuration from file and environment variables
func LoadConfig(configPath string, logger *zap.Logger) (*Config, error) {
	v := viper.New()

	// Set defaults
	setDefaults(v)

	// Set config file
	if configPath != "" {
		v.SetConfigFile(configPath)
	} else {
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		v.AddConfigPath("./configs")
		v.AddConfigPath(".")
	}

	// Read environment variables
	v.SetEnvPrefix("MEMCON")
	v.AutomaticEnv()

	// Read config file
	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			logger.Warn("Config file not found, using defaults and environment variables")
		} else {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	} else {
		logger.Info("Loaded configuration", zap.String("file", v.ConfigFileUsed()))
	}

	// Unmarshal config
	var config Config
	if err := v.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Override sensitive values from environment if present
	if apiKey := os.Getenv("MEMCON_MEMORY_API_API_KEY"); apiKey != "" {
		config.MemoryAPI.APIKey = apiKey
		logger.Info("Using Memory API key from environment")
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return &config, nil
}

// setDefaults sets default configuration values
func setDefaults(v *viper.Viper) {
	// Server defaults
	v.SetDefault("server.host", "0.0.0.0")
	v.SetDefault("server.port", 8080)

	// Memory API defaults
	v.SetDefault("memory_api.timeout", 30)
	v.SetDefault("memory_api.max_retries", 3)
	v.SetDefault("memory_api.retry_delay", 2)

	// LightRAG defaults
	v.SetDefault("lightrag.timeout", 60)
	v.SetDefault("lightrag.max_retries", 3)
	v.SetDefault("lightrag.retry_delay", 2)

	// Logging defaults (as per user's answer: both formats, configurable)
	v.SetDefault("logging.level", "info")
	v.SetDefault("logging.format", "console")
	v.SetDefault("logging.output_path", "stdout")

	// Storage defaults (as per user's answer: both JSON and SQLite)
	v.SetDefault("storage.type", "json")
	v.SetDefault("storage.path", "./data")
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.MemoryAPI.URL == "" {
		return fmt.Errorf("memory_api.url is required")
	}
	if c.MemoryAPI.APIKey == "" {
		return fmt.Errorf("memory_api.api_key is required")
	}
	if c.LightRAG.URL == "" {
		return fmt.Errorf("lightrag.url is required")
	}

	// Validate logging format (as per user's answer: json or console)
	if c.Logging.Format != "json" && c.Logging.Format != "console" {
		return fmt.Errorf("logging.format must be 'json' or 'console', got '%s'", c.Logging.Format)
	}

	// Validate storage type (as per user's answer: both in parallel)
	if c.Storage.Type != "json" && c.Storage.Type != "sqlite" {
		return fmt.Errorf("storage.type must be 'json' or 'sqlite', got '%s'", c.Storage.Type)
	}

	// Validate each connector
	for i, connector := range c.Connectors {
		if err := connector.Validate(); err != nil {
			return fmt.Errorf("connector %d validation failed: %w", i, err)
		}
	}

	return nil
}

// GetConnectorByID returns a connector by its ID
func (c *Config) GetConnectorByID(id string) (*models.ConnectorConfig, error) {
	for i := range c.Connectors {
		if c.Connectors[i].ID == id {
			return &c.Connectors[i], nil
		}
	}
	return nil, fmt.Errorf("connector not found: %s", id)
}
