#!/usr/bin/env python3
"""
Generate icon files for EffektGuard Home Assistant integration.

This script creates properly sized icon and logo images according to
Home Assistant and HACS standards.

Requirements:
    - Pillow (PIL): pip install Pillow
    - Source logo: EffektGuard-logo.png (1024x1024 recommended)

Usage:
    python3 scripts/generate_icons.py
"""

from PIL import Image
import os
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent
SOURCE_LOGO = REPO_ROOT / "EffektGuard-logo.png"
ICON_DIR = REPO_ROOT / "custom_components" / "effektguard" / "icons"

# Icon specifications (Home Assistant standards)
ICON_SIZES = {
    "icon.png": 256,
    "icon@2x.png": 512,
    "logo.png": 256,
    "logo@2x.png": 512,
}


def main():
    """Generate all required icon sizes from source logo."""
    
    # Verify source logo exists
    if not SOURCE_LOGO.exists():
        print(f"❌ Error: Source logo not found at {SOURCE_LOGO}")
        print("   Please ensure EffektGuard-logo.png exists in the repository root.")
        return 1
    
    # Load source image
    print(f"📖 Loading source logo: {SOURCE_LOGO}")
    try:
        img = Image.open(SOURCE_LOGO)
        print(f"   Size: {img.size}, Mode: {img.mode}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return 1
    
    # Create icon directory
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Icon directory: {ICON_DIR}")
    
    # Generate each icon size
    print("\n🎨 Generating icons...")
    for filename, size in ICON_SIZES.items():
        output_path = ICON_DIR / filename
        
        # Resize with high-quality Lanczos resampling
        resized = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Save with optimization
        resized.save(output_path, optimize=True)
        
        # Get file size for reporting
        file_size = output_path.stat().st_size / 1024  # KB
        
        print(f"   ✅ {filename:<16} ({size}x{size}) - {file_size:.1f} KB")
    
    print("\n✨ All icons generated successfully!")
    print(f"\nLocation: {ICON_DIR}")
    print("\nNext steps:")
    print("1. Verify icons look correct")
    print("2. Restart Home Assistant")
    print("3. Check integration displays logo properly")
    
    return 0


if __name__ == "__main__":
    exit(main())
