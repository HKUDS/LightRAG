package logger

import (
	"fmt"
	"os"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// LogConfig holds logger configuration
type LogConfig struct {
	Level      string // debug, info, warn, error
	Format     string // json or console (as per user's answer: both, configurable)
	OutputPath string // file path or stdout
}

// NewLogger creates a new zap logger based on configuration
func NewLogger(config LogConfig) (*zap.Logger, error) {
	// Parse log level
	level, err := parseLevel(config.Level)
	if err != nil {
		return nil, err
	}

	// Create encoder config
	encoderConfig := zapcore.EncoderConfig{
		TimeKey:        "timestamp",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		FunctionKey:    zapcore.OmitKey,
		MessageKey:     "message",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.LowercaseLevelEncoder,
		EncodeTime:     zapcore.ISO8601TimeEncoder,
		EncodeDuration: zapcore.SecondsDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	}

	// Select encoder based on format (as per user's answer: both formats, configurable)
	var encoder zapcore.Encoder
	switch config.Format {
	case "json":
		encoder = zapcore.NewJSONEncoder(encoderConfig)
	case "console":
		encoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
		encoder = zapcore.NewConsoleEncoder(encoderConfig)
	default:
		return nil, fmt.Errorf("invalid log format: %s (must be 'json' or 'console')", config.Format)
	}

	// Configure output
	var writeSyncer zapcore.WriteSyncer
	if config.OutputPath == "stdout" || config.OutputPath == "" {
		writeSyncer = zapcore.AddSync(os.Stdout)
	} else {
		file, err := os.OpenFile(config.OutputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return nil, fmt.Errorf("failed to open log file: %w", err)
		}
		writeSyncer = zapcore.AddSync(file)
	}

	// Create core
	core := zapcore.NewCore(encoder, writeSyncer, level)

	// Create logger
	logger := zap.New(core, zap.AddCaller(), zap.AddStacktrace(zapcore.ErrorLevel))

	return logger, nil
}

// parseLevel converts string level to zapcore.Level
func parseLevel(level string) (zapcore.Level, error) {
	switch level {
	case "debug":
		return zapcore.DebugLevel, nil
	case "info":
		return zapcore.InfoLevel, nil
	case "warn":
		return zapcore.WarnLevel, nil
	case "error":
		return zapcore.ErrorLevel, nil
	default:
		return zapcore.InfoLevel, fmt.Errorf("invalid log level: %s (using info)", level)
	}
}

// NewDefaultLogger creates a logger with sensible defaults
func NewDefaultLogger() *zap.Logger {
	logger, err := NewLogger(LogConfig{
		Level:      "info",
		Format:     "console",
		OutputPath: "stdout",
	})
	if err != nil {
		// Fallback to basic logger
		logger, _ = zap.NewProduction()
	}
	return logger
}
