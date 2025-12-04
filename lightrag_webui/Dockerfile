# Multi-stage build for LightRAG WebUI
# Stage 1: Build
FROM oven/bun:latest AS builder

WORKDIR /app

# Copy package files
COPY package.json bun.lock ./

# Install dependencies with bun
RUN bun install

# Copy source code
COPY . .

# Build the application
RUN bun run build

# Stage 2: Serve
FROM node:20-alpine3.20

WORKDIR /app

# Install serve to run the application  
RUN npm install -g serve

# The vite config outputs to dist
COPY --from=builder /app/dist /app/dist

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
  CMD wget --quiet --tries=1 --spider http://localhost:3000 || exit 1

# Start the application
CMD ["serve", "-s", "dist", "-l", "3000"]
