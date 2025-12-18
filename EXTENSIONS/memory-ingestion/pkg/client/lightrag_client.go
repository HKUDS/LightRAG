package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"go.uber.org/zap"
)

// LightRAGClient is a client for the LightRAG API
type LightRAGClient struct {
	apiURL     string
	httpClient *http.Client
	logger     *zap.Logger
	maxRetries int
	retryDelay time.Duration
}

// LightRAGClientConfig holds configuration for the LightRAG API client
type LightRAGClientConfig struct {
	APIURL     string
	Timeout    time.Duration
	MaxRetries int
	RetryDelay time.Duration
}

// DocumentRequest represents a document submission to LightRAG
type DocumentRequest struct {
	Text     string            `json:"text"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// DocumentResponse represents the response from LightRAG
type DocumentResponse struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
	DocID   string `json:"doc_id,omitempty"`
}

// NewLightRAGClient creates a new LightRAG API client
func NewLightRAGClient(config LightRAGClientConfig, logger *zap.Logger) *LightRAGClient {
	if config.Timeout == 0 {
		config.Timeout = 60 * time.Second // Longer timeout for document processing
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}
	if config.RetryDelay == 0 {
		config.RetryDelay = 2 * time.Second
	}

	return &LightRAGClient{
		apiURL: config.APIURL,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
		logger:     logger,
		maxRetries: config.MaxRetries,
		retryDelay: config.RetryDelay,
	}
}

// InsertDocument inserts a document into LightRAG
func (c *LightRAGClient) InsertDocument(ctx context.Context, text string, metadata map[string]string) (*DocumentResponse, error) {
	url := fmt.Sprintf("%s/documents/text", c.apiURL)

	docReq := DocumentRequest{
		Text:     text,
		Metadata: metadata,
	}

	c.logger.Debug("Inserting document",
		zap.String("url", url),
		zap.Int("text_length", len(text)),
		zap.Any("metadata", metadata),
	)

	var docResp DocumentResponse
	err := c.doRequestWithRetry(ctx, "POST", url, docReq, &docResp)
	if err != nil {
		return nil, fmt.Errorf("failed to insert document: %w", err)
	}

	c.logger.Info("Successfully inserted document",
		zap.String("status", docResp.Status),
		zap.String("doc_id", docResp.DocID),
	)

	return &docResp, nil
}

// HealthCheck checks if the LightRAG API is available
func (c *LightRAGClient) HealthCheck(ctx context.Context) error {
	url := fmt.Sprintf("%s/health", c.apiURL)

	c.logger.Debug("Performing health check", zap.String("url", url))

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("health check returned status %d: %s", resp.StatusCode, string(body))
	}

	c.logger.Info("LightRAG API is healthy")
	return nil
}

// doRequestWithRetry performs an HTTP request with retry logic
func (c *LightRAGClient) doRequestWithRetry(ctx context.Context, method, url string, requestBody interface{}, result interface{}) error {
	var lastErr error

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			c.logger.Warn("Retrying request",
				zap.String("url", url),
				zap.Int("attempt", attempt),
				zap.Int("max_retries", c.maxRetries),
			)
			time.Sleep(c.retryDelay * time.Duration(attempt))
		}

		// Marshal request body
		var bodyReader io.Reader
		if requestBody != nil {
			bodyBytes, err := json.Marshal(requestBody)
			if err != nil {
				return fmt.Errorf("failed to marshal request body: %w", err)
			}
			bodyReader = bytes.NewReader(bodyBytes)
		}

		req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
		if err != nil {
			return fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			c.logger.Warn("Request failed",
				zap.String("url", url),
				zap.Error(err),
			)
			continue
		}

		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			body, _ := io.ReadAll(resp.Body)
			lastErr = fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))

			// Don't retry on 4xx errors (client errors)
			if resp.StatusCode >= 400 && resp.StatusCode < 500 {
				return lastErr
			}

			c.logger.Warn("Non-success status code",
				zap.Int("status_code", resp.StatusCode),
				zap.String("body", string(body)),
			)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			lastErr = fmt.Errorf("failed to read response body: %w", err)
			continue
		}

		if result != nil {
			err = json.Unmarshal(body, result)
			if err != nil {
				return fmt.Errorf("failed to unmarshal response: %w", err)
			}
		}

		return nil
	}

	return fmt.Errorf("request failed after %d retries: %w", c.maxRetries, lastErr)
}
