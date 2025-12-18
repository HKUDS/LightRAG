# Multi-stage build for Memory Connector
# As per user's answer: Go 1.21+

# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make

WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o memory-connector cmd/memory-connector/main.go

# Runtime stage
FROM alpine:latest

# Install ca-certificates for HTTPS requests
RUN apk --no-cache add ca-certificates tzdata

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/memory-connector /app/memory-connector

# Copy default config
COPY configs/config.yaml /app/configs/config.yaml

# Create data directory
RUN mkdir -p /app/data

# Expose API port
EXPOSE 8080

# Run as non-root user
RUN addgroup -g 1000 appuser && \
    adduser -D -u 1000 -G appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

# Set entrypoint
ENTRYPOINT ["/app/memory-connector"]

# Default command
CMD ["serve"]
