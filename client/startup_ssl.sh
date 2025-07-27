#!/bin/bash

# Startup script for Streamlit with SSL support
# This script handles both HTTP and HTTPS modes

echo "🚀 Google Search MCP Client - Starting Application..."

# Check if SSL is enabled
if [ "$SSL_ENABLED" = "true" ]; then
    echo "🔒 SSL mode enabled"
    
    # Generate certificates if they don't exist
    if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/private.key" ]; then
        echo "📝 Generating SSL certificates..."
        
        # Create ssl directory with proper permissions
        mkdir -p ssl
        chmod 755 ssl
        
        # Generate certificates using the shell script
        if [ -f "generate_ssl_certificate.sh" ]; then
            chmod +x generate_ssl_certificate.sh
            ./generate_ssl_certificate.sh
            
            echo "✅ SSL certificates generated successfully"
        else
            echo "❌ Certificate generation script not found"
            echo "🔄 Falling back to HTTP mode"
            SSL_ENABLED="false"
        fi
    else
        echo "✅ SSL certificates found"
    fi
    
    # Start with HTTPS if certificates exist
    if [ "$SSL_ENABLED" = "true" ] && [ -f "ssl/cert.pem" ] && [ -f "ssl/private.key" ]; then
        echo "🔒 Starting Streamlit with HTTPS on port 8503..."
        echo "📱 Access URL: https://localhost:8503"
        echo "⚠️  Browser will show security warning for self-signed certificate"
        echo "   Click 'Advanced' -> 'Proceed to localhost (unsafe)' to continue"
        echo ""
        
        exec streamlit run app.py \
            --server.port=8503 \
            --server.address=0.0.0.0 \
            --server.enableCORS=false \
            --server.enableXsrfProtection=false \
            --server.sslCertFile=ssl/cert.pem \
            --server.sslKeyFile=ssl/private.key
    fi
fi

# Default to HTTP mode
echo "🌐 Starting Streamlit with HTTP on port 8501..."
echo "📱 Access URL: http://localhost:8501"
echo ""

exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0