package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/kamir/memory-connector/internal/logger"
	"github.com/kamir/memory-connector/pkg/client"
	"github.com/kamir/memory-connector/pkg/config"
	"github.com/kamir/memory-connector/pkg/orchestrator"
	"github.com/kamir/memory-connector/pkg/scheduler"
	"github.com/kamir/memory-connector/pkg/state"
	"github.com/kamir/memory-connector/pkg/transformer"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var (
	cfgFile    string
	jsonOutput bool
	log        *zap.Logger
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "memory-connector",
		Short: "Memory API to LightRAG ingestion connector",
		Long: `Memory Connector pulls memory items from the Memory API and ingests them
into LightRAG's knowledge graph for entity and relation extraction.`,
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			// Initialize logger with default config
			log = logger.NewDefaultLogger()
		},
		PersistentPostRun: func(cmd *cobra.Command, args []string) {
			if log != nil {
				log.Sync()
			}
		},
	}

	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "./configs/config.yaml", "config file path")
	rootCmd.PersistentFlags().BoolVar(&jsonOutput, "json", false, "output in JSON format (as per user's answer: both text and JSON)")

	// Add commands
	rootCmd.AddCommand(versionCmd())
	rootCmd.AddCommand(syncCmd())
	rootCmd.AddCommand(serveCmd())
	rootCmd.AddCommand(listCmd())
	rootCmd.AddCommand(statusCmd())

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// versionCmd returns the version command
func versionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print version information",
		Run: func(cmd *cobra.Command, args []string) {
			version := map[string]string{
				"version":    "0.1.0",
				"go_version": "1.21",
				"build_date": time.Now().Format(time.RFC3339),
			}

			if jsonOutput {
				data, _ := json.MarshalIndent(version, "", "  ")
				fmt.Println(string(data))
			} else {
				fmt.Printf("Memory Connector v%s\n", version["version"])
				fmt.Printf("Go version: %s\n", version["go_version"])
			}
		},
	}
}

// syncCmd returns the sync command
func syncCmd() *cobra.Command {
	var connectorID string

	cmd := &cobra.Command{
		Use:   "sync",
		Short: "Trigger a manual sync for a connector",
		Long:  "Manually trigger a sync operation for the specified connector",
		Run: func(cmd *cobra.Command, args []string) {
			runSync(connectorID)
		},
	}

	cmd.Flags().StringVarP(&connectorID, "connector", "c", "", "connector ID to sync (required)")
	cmd.MarkFlagRequired("connector")

	return cmd
}

// serveCmd returns the serve command
func serveCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "serve",
		Short: "Start the connector service with scheduler and API",
		Long:  "Start the connector service in daemon mode with automatic scheduling and management API",
		Run: func(cmd *cobra.Command, args []string) {
			runServe()
		},
	}
}

// listCmd returns the list command
func listCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List all configured connectors",
		Long:  "Display all connectors defined in the configuration file",
		Run: func(cmd *cobra.Command, args []string) {
			runList()
		},
	}
}

// statusCmd returns the status command
func statusCmd() *cobra.Command{
	var connectorID string

	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show status and sync history for a connector",
		Long:  "Display the current status and last sync report for a connector",
		Run: func(cmd *cobra.Command, args []string) {
			runStatus(connectorID)
		},
	}

	cmd.Flags().StringVarP(&connectorID, "connector", "c", "", "connector ID (required)")
	cmd.MarkFlagRequired("connector")

	return cmd
}

// runSync executes a manual sync
func runSync(connectorID string) {
	// Load configuration
	cfg, err := config.LoadConfig(cfgFile, log)
	if err != nil {
		log.Fatal("Failed to load config", zap.Error(err))
	}

	// Update logger with config settings
	log, err = logger.NewLogger(logger.LogConfig{
		Level:      cfg.Logging.Level,
		Format:     cfg.Logging.Format,
		OutputPath: cfg.Logging.OutputPath,
	})
	if err != nil {
		log.Fatal("Failed to initialize logger", zap.Error(err))
	}

	// Find connector
	connectorCfg, err := cfg.GetConnectorByID(connectorID)
	if err != nil {
		log.Fatal("Connector not found", zap.String("connector_id", connectorID))
	}

	// Initialize components
	memoryClient := client.NewMemoryClient(client.MemoryClientConfig{
		APIURL:     cfg.MemoryAPI.URL,
		APIKey:     cfg.MemoryAPI.APIKey,
		Timeout:    time.Duration(cfg.MemoryAPI.Timeout) * time.Second,
		MaxRetries: cfg.MemoryAPI.MaxRetries,
		RetryDelay: time.Duration(cfg.MemoryAPI.RetryDelay) * time.Second,
	}, log)

	lightragClient := client.NewLightRAGClient(client.LightRAGClientConfig{
		APIURL:     cfg.LightRAG.URL,
		Timeout:    time.Duration(cfg.LightRAG.Timeout) * time.Second,
		MaxRetries: cfg.LightRAG.MaxRetries,
		RetryDelay: time.Duration(cfg.LightRAG.RetryDelay) * time.Second,
	}, log)

	trans, err := transformer.NewTransformer(connectorCfg.Transform.Strategy, log)
	if err != nil {
		log.Fatal("Failed to create transformer", zap.Error(err))
	}

	stateManager, err := state.NewStateManager(state.Config{
		Type: cfg.Storage.Type,
		Path: cfg.Storage.Path,
	}, log)
	if err != nil {
		log.Fatal("Failed to create state manager", zap.Error(err))
	}
	defer stateManager.Close()

	orch := orchestrator.NewOrchestrator(memoryClient, lightragClient, trans, stateManager, log)

	// Execute sync
	log.Info("Starting manual sync", zap.String("connector_id", connectorID))

	report, err := orch.SyncConnector(context.Background(), connectorCfg)
	if err != nil {
		log.Fatal("Sync failed", zap.Error(err))
	}

	// Output report (as per user's answer: both text and JSON)
	if jsonOutput {
		data, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(data))
	} else {
		fmt.Printf("\n=== Sync Report ===\n")
		fmt.Printf("Connector ID: %s\n", report.ConnectorID)
		fmt.Printf("Status: %s\n", report.Status)
		fmt.Printf("Duration: %s\n", report.Duration)
		fmt.Printf("Fetched: %d\n", report.TotalFetched)
		fmt.Printf("Processed: %d\n", report.TotalProcessed)
		fmt.Printf("Skipped: %d\n", report.TotalSkipped)
		fmt.Printf("Failed: %d\n", report.TotalFailed)
		fmt.Printf("Success Rate: %.2f%%\n", report.CalculateSuccessRate())

		if len(report.MemoriesFailed) > 0 {
			fmt.Printf("\nFailed Items:\n")
			for _, failed := range report.MemoriesFailed {
				fmt.Printf("  - %s: %s\n", failed.MemoryID, failed.ErrorMessage)
			}
		}
	}

	if report.IsFailed() {
		os.Exit(1)
	}
}

// runServe starts the service in daemon mode
func runServe() {
	// Load configuration
	cfg, err := config.LoadConfig(cfgFile, log)
	if err != nil {
		log.Fatal("Failed to load config", zap.Error(err))
	}

	// Update logger
	log, err = logger.NewLogger(logger.LogConfig{
		Level:      cfg.Logging.Level,
		Format:     cfg.Logging.Format,
		OutputPath: cfg.Logging.OutputPath,
	})
	if err != nil {
		log.Fatal("Failed to initialize logger", zap.Error(err))
	}

	log.Info("Starting Memory Connector service",
		zap.String("version", "0.1.0"),
		zap.Int("connectors", len(cfg.Connectors)),
	)

	// Initialize components (similar to sync but for all connectors)
	// TODO: Implement full service mode with HTTP API and scheduler
	log.Fatal("Service mode not yet implemented - use 'sync' command for manual syncs")
}

// runList lists all connectors
func runList() {
	cfg, err := config.LoadConfig(cfgFile, log)
	if err != nil {
		log.Fatal("Failed to load config", zap.Error(err))
	}

	if jsonOutput {
		data, _ := json.MarshalIndent(cfg.Connectors, "", "  ")
		fmt.Println(string(data))
	} else {
		fmt.Printf("\n=== Configured Connectors ===\n\n")
		for _, conn := range cfg.Connectors {
			fmt.Printf("ID: %s\n", conn.ID)
			fmt.Printf("  Enabled: %v\n", conn.Enabled)
			fmt.Printf("  Context: %s\n", conn.ContextID)
			fmt.Printf("  Schedule: %s\n", conn.GetScheduleDescription())
			fmt.Printf("  Transform: %s\n", conn.Transform.Strategy)
			fmt.Println()
		}
	}
}

// runStatus shows connector status
func runStatus(connectorID string) {
	cfg, err := config.LoadConfig(cfgFile, log)
	if err != nil {
		log.Fatal("Failed to load config", zap.Error(err))
	}

	stateManager, err := state.NewStateManager(state.Config{
		Type: cfg.Storage.Type,
		Path: cfg.Storage.Path,
	}, log)
	if err != nil {
		log.Fatal("Failed to create state manager", zap.Error(err))
	}
	defer stateManager.Close()

	syncState, err := stateManager.GetState(context.Background(), connectorID)
	if err != nil {
		log.Fatal("Failed to get state", zap.Error(err))
	}

	if jsonOutput {
		data, _ := json.MarshalIndent(syncState, "", "  ")
		fmt.Println(string(data))
	} else {
		fmt.Printf("\n=== Connector Status ===\n")
		fmt.Printf("Connector ID: %s\n", syncState.ConnectorID)
		fmt.Printf("Context ID: %s\n", syncState.ContextID)
		fmt.Printf("Total Syncs: %d\n", syncState.TotalSyncCount)
		fmt.Printf("Last Sync: %s\n", syncState.LastSyncTime.Format(time.RFC3339))
		fmt.Printf("Processed Memories: %d\n", len(syncState.ProcessedIDs))
		fmt.Printf("Failed Items (DLQ): %d\n", len(syncState.FailedItems))

		if syncState.LastSyncReport != nil {
			fmt.Printf("\nLast Sync Report:\n")
			fmt.Printf("  Status: %s\n", syncState.LastSyncReport.Status)
			fmt.Printf("  Duration: %s\n", syncState.LastSyncReport.Duration)
			fmt.Printf("  Success Rate: %.2f%%\n", syncState.LastSyncReport.CalculateSuccessRate())
		}
	}
}
