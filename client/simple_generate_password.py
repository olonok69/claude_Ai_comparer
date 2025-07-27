#!/usr/bin/env python3
"""
Simple password hashing utility for streamlit-authenticator
Run this script to generate hashed passwords for your config.yaml file

Usage:
    python simple_generate_passwords.py
"""

import bcrypt
import yaml
import os

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    # Convert password to bytes
    password_bytes = password.encode('utf-8')
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Return as string
    return hashed.decode('utf-8')

def main():
    """Generate hashed passwords and create config file."""
    
    # Define the passwords you want to hash
    users = {
        'admin': {
            'password': 'very_Secure_p@ssword_123!',
            'name': 'Administrator',
            'email': 'admin@csm.com'
        },
        'juan': {
            'password': 'Larisa1000@',
            'name': 'Juan Perez', 
            'email': 'olonok@gmail.com'
        },
        'giovanni_romero': {
            'password': 'MrRomero2024!',
            'name': 'Giovanni Romero',
            'email': 'g.romero@closerstillmedia.com'
        },
        'demo_user': {
            'password': 'strong_password_123!',
            'name': 'Demo User',
            'email': 'demo@example.com'
        }
    }
    
    print("🔐 Generating hashed passwords for streamlit-authenticator...")
    print("=" * 60)
    
    # Create config structure
    config = {
        'credentials': {
            'usernames': {}
        },
        'cookie': {
            'expiry_days': 30,
            'key': 'csm_mcp_auth_key_2024_secure_random_string_change_in_production',
            'name': 'csm_mcp_auth_cookie'
        },
        'preauthorized': {
            'emails': []
        }
    }
    
    # Process each user
    for username, user_info in users.items():
        plain_password = user_info['password']
        hashed_password = hash_password(plain_password)
        
        # Add to config
        config['credentials']['usernames'][username] = {
            'email': user_info['email'],
            'name': user_info['name'],
            'password': hashed_password
        }
        
        # Add to preauthorized emails
        config['preauthorized']['emails'].append(user_info['email'])
        
        print(f"Username: {username}")
        print(f"Password: {plain_password}")
        print(f"Name: {user_info['name']}")
        print(f"Email: {user_info['email']}")
        print(f"Hashed: {hashed_password[:50]}...")
        print("-" * 40)
    
    # Create keys directory if it doesn't exist
    os.makedirs('keys', exist_ok=True)
    
    # Save configuration to file
    config_path = 'keys/config.yaml'
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Configuration saved to {config_path}")
    print("\n📋 Login Credentials:")
    print("=" * 30)
    
    for username, user_info in users.items():
        print(f"Username: {username}")
        print(f"Password: {user_info['password']}")
        print(f"Name: {user_info['name']}")
        print(f"Email: {user_info['email']}")
        print("-" * 20)
    
    print("\n⚠️  IMPORTANT SECURITY NOTES:")
    print("1. 🔑 Change the cookie key in production!")
    print("2. 🔒 Use strong passwords in production!")
    print("3. 📁 Add keys/ directory to .gitignore!")
    print("4. 🛡️  Keep this configuration file secure!")
    print("5. 🔄 Consider regular password rotation!")
    
    print(f"\n🚀 Ready to run: streamlit run app.py")

if __name__ == "__main__":
    main()