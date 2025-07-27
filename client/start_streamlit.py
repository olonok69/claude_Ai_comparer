#!/usr/bin/env python3
"""
Startup script for Streamlit app with optional SSL support
"""

import os
import sys
import subprocess
from pathlib import Path

def check_ssl_certificates():
    """Check if SSL certificates exist."""
    ssl_dir = Path("ssl")
    cert_file = ssl_dir / "cert.pem"
    key_file = ssl_dir / "private.key"
    
    return cert_file.exists() and key_file.exists()

def generate_certificates():
    """Generate SSL certificates."""
    print("🔒 SSL certificates not found. Generating new ones...")
    
    # Try to run the Python certificate generator
    try:
        from generate_ssl_certificate import generate_self_signed_cert
        generate_self_signed_cert()
        return True
    except ImportError:
        # Fallback to shell script if Python script not available
        if os.name != 'nt' and Path("generate_ssl_certificate.sh").exists():
            try:
                subprocess.run(["bash", "generate_ssl_certificate.sh"], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
        else:
            print("❌ Unable to generate SSL certificates automatically.")
            print("📝 Please run: python generate_ssl_certificate.py")
            return False

def start_streamlit():
    """Start Streamlit with appropriate configuration."""
    ssl_enabled = os.getenv("SSL_ENABLED", "false").lower() == "true"
    
    if ssl_enabled:
        print("🔒 SSL mode enabled")
        
        # Check if certificates exist
        if not check_ssl_certificates():
            if not generate_certificates():
                print("❌ Failed to generate SSL certificates. Falling back to HTTP mode.")
                ssl_enabled = False
        
        if ssl_enabled:
            print("🚀 Starting Streamlit with HTTPS on port 8502...")
            cmd = [
                "streamlit", "run", "app.py",
                "--server.port=8502",
                "--server.address=0.0.0.0",
                "--server.enableCORS=false",
                "--server.enableXsrfProtection=false",
                "--server.sslCertFile=ssl/cert.pem",
                "--server.sslKeyFile=ssl/private.key"
            ]
        else:
            print("🌐 Starting Streamlit with HTTP on port 8501...")
            cmd = [
                "streamlit", "run", "app.py",
                "--server.port=8501",
                "--server.address=0.0.0.0"
            ]
    else:
        print("🌐 Starting Streamlit with HTTP on port 8501...")
        cmd = [
            "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ]
    
    # Print access URLs
    if ssl_enabled:
        print("📱 Access URLs:")
        print("   🔒 HTTPS: https://localhost:8502")
        print("   🔒 HTTPS: https://127.0.0.1:8502")
        print("   🔒 HTTPS: https://0.0.0.0:8502")
        print("")
        print("⚠️  Browser will show security warning for self-signed certificate")
        print("   Click 'Advanced' -> 'Proceed to localhost (unsafe)' to continue")
    else:
        print("📱 Access URLs:")
        print("   🌐 HTTP: http://localhost:8501")
        print("   🌐 HTTP: http://127.0.0.1:8501")
        print("   🌐 HTTP: http://0.0.0.0:8501")
    
    print("\n" + "="*60)
    
    # Start Streamlit
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_streamlit()