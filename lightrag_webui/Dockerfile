# Use the official Bun image as the base image
FROM oven/bun:latest

# Set the working directory
WORKDIR /app

# Copy project files to the working directory
COPY . .

# Install project dependencies
RUN bun install --frozen-lockfile

# Build the project
RUN bun run build --emptyOutDir

# Expose the application's port
EXPOSE 5173

# Start the server
CMD ["bun", "run", "dev", "--host", "0.0.0.0"]