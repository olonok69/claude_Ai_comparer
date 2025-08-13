#!/bin/bash

# Generate self-signed certificate for MCP HTTPS server
# This script creates a certificate valid for 365 days

echo "🔐 Generating self-signed certificate for MCP HTTPS server..."

# Create certs directory if it doesn't exist
mkdir -p certs
cd certs

# Generate private key
echo "📝 Generating private key..."
openssl genrsa -out server.key 2048

# Generate certificate signing request
echo "📋 Creating certificate signing request..."
openssl req -new -key server.key -out server.csr \
  -subj "/C=US/ST=State/L=City/O=MCP Server/CN=localhost"

# Generate self-signed certificate (valid for 365 days)
echo "🎯 Generating self-signed certificate..."
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

# Optional: Create a combined PEM file
echo "📦 Creating combined PEM file..."
cat server.crt server.key > server.pem

# Set appropriate permissions
chmod 600 server.key
chmod 644 server.crt
chmod 600 server.pem

# Clean up CSR file (no longer needed)
rm server.csr

echo "✅ Certificate generation complete!"
echo ""
echo "📁 Files created in ./certs directory:"
echo "   - server.key  : Private key (keep this secure!)"
echo "   - server.crt  : Certificate"
echo "   - server.pem  : Combined certificate and key"
echo ""
echo "🚀 You can now run the HTTPS server with:"
echo "   python serversse_https.py"
echo ""
echo "⚠️  Note: This is a self-signed certificate."
echo "   Browsers will show a security warning that you'll need to accept."

cd ..