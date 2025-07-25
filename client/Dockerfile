FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies including OpenSSL and curl for health checks
RUN apt-get update && apt-get install -y \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for authentication config, SSL certificates, and MCP servers
RUN mkdir -p keys ssl mcp_servers/company_tagging/categories

# Copy the application code
COPY . .

# Copy the company tagging MCP server
COPY mcp_servers/ /app/mcp_servers/

# Copy SSL-related scripts and make them executable
COPY generate_ssl_certificate.sh startup_ssl.sh ./
RUN chmod +x generate_ssl_certificate.sh startup_ssl.sh

# Ensure proper permissions for directories (run as root to avoid permission issues)
RUN chmod 755 keys ssl mcp_servers

# Expose both HTTP and HTTPS ports
EXPOSE 8501
EXPOSE 8503

# Fixed health check - check both HTTP and HTTPS endpoints
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f -k https://localhost:8503/_stcore/health || curl -f http://localhost:8501/_stcore/health || exit 1

# Run as root to avoid SSL permission issues
CMD ["./startup_ssl.sh"]