FROM node:20-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy package files first (for better Docker layer caching)
COPY package.json package-lock.json* ./

# Install dependencies with verbose logging
RUN npm ci --only=production --verbose || npm install --verbose

# Copy source code (excluding node_modules)
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Start the application
CMD ["node", "main.js"]