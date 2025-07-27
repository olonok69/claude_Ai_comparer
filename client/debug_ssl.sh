#!/bin/bash

# Debug script for SSL configuration

echo "🔍 SSL Debug Information"
echo "========================"

# Check environment variables
echo "📊 Environment Variables:"
echo "SSL_ENABLED: ${SSL_ENABLED:-not set}"
echo ""

# Check if SSL directory exists
echo "📁 SSL Directory Status:"
if [ -d "ssl" ]; then
    echo "✅ ssl/ directory exists"
    ls -la ssl/
else
    echo "❌ ssl/ directory does not exist"
fi
echo ""

# Check SSL files
echo "📄 SSL Files Status:"
if [ -f "ssl/cert.pem" ]; then
    echo "✅ Certificate file exists: ssl/cert.pem"
    echo "📝 Certificate details:"
    openssl x509 -in ssl/cert.pem -text -noout | head -20
else
    echo "❌ Certificate file missing: ssl/cert.pem"
fi

if [ -f "ssl/private.key" ]; then
    echo "✅ Private key file exists: ssl/private.key"
    echo "🔑 Key permissions: $(ls -l ssl/private.key | awk '{print $1}')"
else
    echo "❌ Private key file missing: ssl/private.key"
fi
echo ""

# Check certificate validity
if [ -f "ssl/cert.pem" ]; then
    echo "📅 Certificate Validity:"
    openssl x509 -in ssl/cert.pem -dates -noout
    echo ""
fi

# Test certificate and key match
if [ -f "ssl/cert.pem" ] && [ -f "ssl/private.key" ]; then
    echo "🔗 Certificate and Key Compatibility:"
    cert_md5=$(openssl x509 -noout -modulus -in ssl/cert.pem | openssl md5)
    key_md5=$(openssl rsa -noout -modulus -in ssl/private.key | openssl md5)
    
    if [ "$cert_md5" = "$key_md5" ]; then
        echo "✅ Certificate and private key match"
    else
        echo "❌ Certificate and private key do not match!"
    fi
    echo ""
fi

# Show what Streamlit command would be used
echo "🚀 Streamlit Command Preview:"
if [ "$SSL_ENABLED" = "true" ] && [ -f "ssl/cert.pem" ] && [ -f "ssl/private.key" ]; then
    echo "HTTPS Mode (Port 8502):"
    echo "streamlit run app.py \\"
    echo "  --server.port=8502 \\"
    echo "  --server.address=0.0.0.0 \\"
    echo "  --server.enableCORS=false \\"
    echo "  --server.enableXsrfProtection=false \\"
    echo "  --server.sslCertFile=ssl/cert.pem \\"
    echo "  --server.sslKeyFile=ssl/private.key"
else
    echo "HTTP Mode (Port 8501):"
    echo "streamlit run app.py \\"
    echo "  --server.port=8501 \\"
    echo "  --server.address=0.0.0.0"
fi
echo ""

# Test SSL connection (if certificates exist)
if [ "$SSL_ENABLED" = "true" ] && [ -f "ssl/cert.pem" ]; then
    echo "🔌 SSL Connection Test:"
    echo "You can test the certificate with:"
    echo "openssl s_client -connect localhost:8502 -verify_return_error"
    echo ""
fi

echo "✅ SSL Debug Complete"