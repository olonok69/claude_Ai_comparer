#!/usr/bin/env python3
"""
Password hashing utility for streamlit-authenticator
Run this script to generate hashed passwords for your config.yaml file

Usage:
    python generate_passwords.py
"""

import streamlit_authenticator as stauth
import yaml

def generate_hashed_passwords():
    """Generate hashed passwords for the authentication system."""
    
    # Define the passwords you want to hash
    passwords_to_hash = {
        'admin': 'very_Secure_p@ssword_123!',
        'juan': 'Larisa1000@', 
        'giovanni_romero': 'MrRomero2024!',
        'demo_user': 'StrongPassword123!'
    }
    
    print("🔐 Generating hashed passwords for streamlit-authenticator...")
    print("=" * 60)
    
    hashed_passwords = {}
    
    for username, plain_password in passwords_to_hash.items():
        hashed_pw = stauth.Hasher([plain_password]).generate()[0]
        hashed_passwords[username] = hashed_pw
        print(f"Username: {username}")
        print(f"Plain Password: {plain_password}")
        print(f"Hashed Password: {hashed_pw}")
        print("-" * 40)
    
    # Generate complete config structure
    config = {
        'credentials': {
            'usernames': {
                'admin': {
                    'email': 'admin@csm.com',
                    'name': 'Administrator',
                    'password': hashed_passwords['admin']
                },
                'juan': {
                    'email': 'olonok@gmail.com',
                    'name': 'Juan Perez',
                    'password': hashed_passwords['juan']
                },
                'giovanni_romero': {
                    'email': 'g.romero@closerstillmedia.com',
                    'name': 'Giovanni Romero',
                    'password': hashed_passwords['giovanni_romero']
                },
                'demo_user': {
                    'email': 'demo@example.com',
                    'name': 'Demo User',
                    'password': hashed_passwords['demo_user']
                }
            }
        },
        'cookie': {
            'expiry_days': 30,
            'key': 'csm_mcp_auth_key_2024_secure_random_string',
            'name': 'csm_mcp_auth_cookie'
        },
        'preauthorized': {
            'emails': [
                'admin@csm.com',
                'olonok@gmail.com',
                'g.romero@closerstillmedia.com',
                'demo@example.com'
            ]
        }
    }
    
    # Save to file
    import os
    os.makedirs('keys', exist_ok=True)
    
    with open('keys/config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
    print("\n✅ Configuration saved to keys/config.yaml")
    print("\n📋 Login Credentials:")
    print("=" * 30)
    for username, password in passwords_to_hash.items():
        user_info = config['credentials']['usernames'][username]
        print(f"Username: {username}")
        print(f"Password: {password}")
        print(f"Name: {user_info['name']}")
        print(f"Email: {user_info['email']}")
        print("-" * 20)
    
    print("\n⚠️  SECURITY NOTES:")
    print("1. Change the cookie key in production!")
    print("2. Use strong passwords in production!")
    print("3. Consider using environment variables for sensitive data!")
    print("4. The keys/ directory should be added to .gitignore!")


if __name__ == "__main__":
    generate_hashed_passwords()