#!/usr/bin/env python3
"""
Script to set up the icons directory and provide instructions for adding CSM.png
"""

import os
import sys
from pathlib import Path

def setup_icons_directory():
    """Create the icons directory and provide setup instructions."""
    
    print("🎨 Setting up icons directory for CSM MCP Client...")
    print("=" * 50)
    
    # Create icons directory
    icons_dir = Path("icons")
    icons_dir.mkdir(exist_ok=True)
    
    print(f"✅ Created/verified icons directory: {icons_dir.absolute()}")
    
    # Check if CSM.png already exists
    csm_icon_path = icons_dir / "CSM.png"
    
    if csm_icon_path.exists():
        print(f"✅ CSM.png already exists at: {csm_icon_path.absolute()}")
        print(f"📏 File size: {csm_icon_path.stat().st_size} bytes")
    else:
        print(f"📁 CSM.png not found. Please add your CSM logo to: {csm_icon_path.absolute()}")
        print("")
        print("📋 Icon Requirements:")
        print("   • Format: PNG with transparency support")
        print("   • Recommended size: 256x256 pixels or 512x512 pixels")
        print("   • File name: CSM.png (case sensitive)")
        print("   • Location: icons/CSM.png")
        print("")
        print("🎨 Icon Design Tips:")
        print("   • Use a square aspect ratio for best results")
        print("   • Ensure good contrast for sidebar visibility")
        print("   • Consider both light and dark themes")
        print("   • PNG format supports transparency")
        print("")
        print("📥 How to add your icon:")
        print("   1. Save your CSM logo as 'CSM.png'")
        print("   2. Copy it to the 'icons' directory")
        print("   3. Restart your Streamlit application")
        print("   4. The icon will appear in the sidebar automatically")
    
    # Check for playground.png (used for page icon)
    playground_icon_path = icons_dir / "playground.png"
    
    if playground_icon_path.exists():
        print(f"✅ playground.png exists at: {playground_icon_path.absolute()}")
    else:
        print(f"📁 playground.png not found. This is used for the browser tab icon.")
        print(f"   You can add it to: {playground_icon_path.absolute()}")
    
    print("")
    print("🚀 Setup complete!")
    print("")
    print("🔧 Next steps:")
    print("   1. Add your CSM.png logo to the icons directory")
    print("   2. Run your Streamlit application: streamlit run app.py")
    print("   3. The CSM logo will appear in the sidebar")
    
    return icons_dir

def create_sample_readme():
    """Create a README file in the icons directory."""
    icons_dir = Path("icons")
    readme_path = icons_dir / "README.md"
    
    readme_content = """# Icons Directory

This directory contains icons used by the CSM MCP Client application.

## Required Icons

### CSM.png
- **Purpose**: Main CSM logo displayed in the sidebar
- **Format**: PNG with transparency
- **Recommended size**: 256x256 or 512x512 pixels
- **Location**: `icons/CSM.png`

### playground.png (Optional)
- **Purpose**: Browser tab/favicon icon  
- **Format**: PNG
- **Recommended size**: 32x32 or 64x64 pixels
- **Location**: `icons/playground.png`

## Adding Your Icons

1. Save your CSM logo as `CSM.png` 
2. Copy to this directory: `icons/CSM.png`
3. Restart the Streamlit application
4. Icon will appear automatically in the sidebar

## Icon Guidelines

- Use square aspect ratios for best results
- Ensure good contrast for visibility
- Consider both light and dark theme compatibility
- PNG format recommended for transparency support

## Troubleshooting

If icons don't appear:
- Check file name case sensitivity (`CSM.png` not `csm.png`)
- Verify file format is PNG
- Restart the Streamlit application
- Check browser cache (Ctrl+F5 to refresh)
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Created README.md in icons directory: {readme_path.absolute()}")

def main():
    """Main function to setup icons directory."""
    try:
        # Change to the client directory if we're not already there
        if os.path.exists('client'):
            os.chdir('client')
            print("📁 Changed to client directory")
        
        icons_dir = setup_icons_directory()
        create_sample_readme()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error setting up icons directory: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())