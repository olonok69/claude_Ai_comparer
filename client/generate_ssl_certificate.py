#!/usr/bin/env python3
"""
Generate self-signed SSL certificate for Streamlit app using Python cryptography library
This is a cross-platform alternative to the shell script version.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
except ImportError:
    print("❌ Error: cryptography library not found")
    print("📦 Installing cryptography...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography"])
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

def generate_self_signed_cert():
    """Generate a self-signed SSL certificate for localhost."""
    
    print("🔒 Generating self-signed SSL certificate for Streamlit app...")
    
    # Create ssl directory if it doesn't exist
    ssl_dir = Path("ssl")
    ssl_dir.mkdir(exist_ok=True)
    
    # Generate private key
    print("📝 Generating private key...")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Generate certificate
    print("📝 Generating self-signed certificate...")
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "UK"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "London"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CSM MCP Organization"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "IT Department"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("127.0.0.1"),
            x509.DNSName("0.0.0.0"),
        ]),
        critical=False,
    ).sign(private_key, hashes.SHA256())
    
    # Write private key to file
    private_key_path = ssl_dir / "private.key"
    with open(private_key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # Write certificate to file
    cert_path = ssl_dir / "cert.pem"
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    # Set appropriate permissions (Unix-like systems only)
    if os.name != 'nt':  # Not Windows
        os.chmod(private_key_path, 0o600)
        os.chmod(cert_path, 0o644)
    
    print("✅ SSL certificate generated successfully!")
    print(f"📄 Certificate: {cert_path}")
    print(f"🔑 Private key: {private_key_path}")
    print("")
    print("⚠️  Note: This is a self-signed certificate.")
    print("   Browsers will show a security warning that you'll need to accept.")
    print("")
    print("🚀 Certificate valid for 365 days")
    print(f"   Valid from: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   Valid to: {(datetime.utcnow() + timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')} UTC")

def main():
    """Main function to generate SSL certificate."""
    try:
        generate_self_signed_cert()
        return 0
    except Exception as e:
        print(f"❌ Error generating SSL certificate: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())