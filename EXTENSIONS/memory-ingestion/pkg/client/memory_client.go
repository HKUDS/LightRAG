package client

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/kamir/memory-connector/pkg/models"
	"go.uber.org/zap"
)

// MemoryClient is a client for the Memory API
type MemoryClient struct {
	apiURL     string
	apiKey     string
	httpClient *http.Client
	logger     *zap.Logger
	maxRetries int
	retryDelay time.Duration
}

// MemoryClientConfig holds configuration for the Memory API client
type MemoryClientConfig struct {
	APIURL     string
	APIKey     string
	Timeout    time.Duration
	MaxRetries int
	RetryDelay time.Duration
}

// NewMemoryClient creates a new Memory API client
func NewMemoryClient(config MemoryClientConfig, logger *zap.Logger) *MemoryClient {
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}
	if config.RetryDelay == 0 {
		config.RetryDelay = 2 * time.Second
	}

	return &MemoryClient{
		apiURL: config.APIURL,
		apiKey: config.APIKey,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
		logger:     logger,
		maxRetries: config.MaxRetries,
		retryDelay: config.RetryDelay,
	}
}

// GetMemories fetches memories from the Memory API
func (c *MemoryClient) GetMemories(ctx context.Context, ctxID string, limit int, rangeParam string) (*models.MemoryList, error) {
	// Build URL with query parameters
	baseURL := fmt.Sprintf("%s/memory/%s", c.apiURL, ctxID)
	params := url.Values{}
	params.Add("limit", fmt.Sprintf("%d", limit))
	params.Add("range", rangeParam)

	fullURL := fmt.Sprintf("%s?%s", baseURL, params.Encode())

	c.logger.Debug("Fetching memories",
		zap.String("url", fullURL),
		zap.String("context_id", ctxID),
		zap.Int("limit", limit),
		zap.String("range", rangeParam),
	)

	var memoryList models.MemoryList
	err := c.doRequestWithRetry(ctx, "GET", fullURL, &memoryList)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch memories: %w", err)
	}

	c.logger.Info("Successfully fetched memories",
		zap.String("context_id", ctxID),
		zap.Int("count", memoryList.Count),
	)

	return &memoryList, nil
}

// GetMemoryAudio fetches audio data for a specific memory
func (c *MemoryClient) GetMemoryAudio(ctx context.Context, ctxID, memoryID string) ([]byte, error) {
	url := fmt.Sprintf("%s/memory/%s/%s/audio", c.apiURL, ctxID, memoryID)

	c.logger.Debug("Fetching memory audio",
		zap.String("context_id", ctxID),
		zap.String("memory_id", memoryID),
	)

	return c.doRawRequestWithRetry(ctx, "GET", url)
}

// GetMemoryImage fetches image data for a specific memory
func (c *MemoryClient) GetMemoryImage(ctx context.Context, ctxID, memoryID string) ([]byte, error) {
	url := fmt.Sprintf("%s/memory/%s/%s/image", c.apiURL, ctxID, memoryID)

	c.logger.Debug("Fetching memory image",
		zap.String("context_id", ctxID),
		zap.String("memory_id", memoryID),
	)

	return c.doRawRequestWithRetry(ctx, "GET", url)
}

// doRequestWithRetry performs an HTTP request with retry logic and JSON unmarshaling
func (c *MemoryClient) doRequestWithRetry(ctx context.Context, method, url string, result interface{}) error {
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

		req, err := http.NewRequestWithContext(ctx, method, url, nil)
		if err != nil {
			return fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("X-API-KEY", c.apiKey)
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

		err = json.Unmarshal(body, result)
		if err != nil {
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}

		return nil
	}

	return fmt.Errorf("request failed after %d retries: %w", c.maxRetries, lastErr)
}

// doRawRequestWithRetry performs an HTTP request with retry logic and returns raw bytes
func (c *MemoryClient) doRawRequestWithRetry(ctx context.Context, method, url string) ([]byte, error) {
	var lastErr error

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			c.logger.Warn("Retrying request",
				zap.String("url", url),
				zap.Int("attempt", attempt),
			)
			time.Sleep(c.retryDelay * time.Duration(attempt))
		}

		req, err := http.NewRequestWithContext(ctx, method, url, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("X-API-KEY", c.apiKey)

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			body, _ := io.ReadAll(resp.Body)
			lastErr = fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))

			// Don't retry on 4xx errors
			if resp.StatusCode >= 400 && resp.StatusCode < 500 {
				return nil, lastErr
			}
			continue
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			lastErr = fmt.Errorf("failed to read response body: %w", err)
			continue
		}

		return body, nil
	}

	return nil, fmt.Errorf("request failed after %d retries: %w", c.maxRetries, lastErr)
}
