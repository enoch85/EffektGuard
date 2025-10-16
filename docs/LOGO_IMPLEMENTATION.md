# Logo Implementation for EffektGuard

## Overview

EffektGuard now includes custom branding with a professional logo displayed throughout Home Assistant's UI.

## Logo Specifications

According to Home Assistant and HACS standards, we've implemented:

### Icon Files

Located in: `custom_components/effektguard/icons/`

- **icon.png** (256x256 pixels) - Standard resolution square icon
- **icon@2x.png** (512x512 pixels) - High DPI version of icon
- **logo.png** (256x256 pixels) - Standard resolution logo
- **logo@2x.png** (512x512 pixels) - High DPI version of logo

### Technical Requirements Met

✅ **File format**: PNG (lossless compression)  
✅ **Aspect ratio**: 1:1 (square) for icons  
✅ **Dimensions**: 256x256 (standard) and 512x512 (hDPI)  
✅ **Optimization**: Images are properly compressed for web use  
✅ **Location**: Stored in `custom_components/effektguard/icons/`  
✅ **Manifest**: Removed MDI icon reference to use custom icon  

## Where the Logo Appears

The logo will be displayed in:

1. **Integration card** in HACS
2. **Integration settings** page
3. **Device & Integration** lists
4. **Configuration flow** dialogs
5. **Entity cards** (when applicable)

## Home Assistant Integration

### Manifest Changes

The `manifest.json` file has been updated to remove the Material Design Icon reference:

```json
{
  "domain": "effektguard",
  "name": "EffektGuard",
  // "icon": "mdi:heat-pump" <- REMOVED to use custom icon
  ...
}
```

When no `icon` field is present in the manifest, Home Assistant automatically looks for custom icons in the `icons/` directory within the integration folder.

### File Discovery

Home Assistant will automatically serve the icons using this priority:

1. If requesting `icon.png`: Serves `custom_components/effektguard/icons/icon.png`
2. If requesting `icon@2x.png`: Serves `custom_components/effektguard/icons/icon@2x.png`
3. If requesting `logo.png`: Serves `custom_components/effektguard/icons/logo.png` (falls back to `icon.png` if missing)
4. If requesting `logo@2x.png`: Serves `custom_components/effektguard/icons/logo@2x.png` (falls back to `icon@2x.png` if missing)

## Future: Home Assistant Brands Repository

For wider distribution and better integration, we can submit our logo to the official Home Assistant Brands repository:

**Repository**: https://github.com/home-assistant/brands

### Submission Process

1. Fork the brands repository
2. Add our icons to `custom_integrations/effektguard/`:
   - `icon.png`
   - `icon@2x.png`
   - `logo.png` (optional if same as icon)
   - `logo@2x.png` (optional if same as icon)
3. Submit a pull request

### Benefits of Brands Repository

- **CDN hosting**: Served via `https://brands.home-assistant.io/effektguard/icon.png`
- **Centralized caching**: 7-day browser cache, 24-hour Cloudflare cache
- **Fallback support**: Placeholder images if logo is missing
- **Official recognition**: Listed alongside core integrations

## Source Logo

Original logo: `/workspaces/EffektGuard/EffektGuard-logo.png` (1024x1024)

All derived icons are high-quality downscaled versions using Lanczos resampling for optimal quality.

## Testing

To verify the logo appears correctly:

1. **Restart Home Assistant** after updating the integration
2. Navigate to **Settings → Devices & Services**
3. The EffektGuard integration should display the custom logo
4. Check **HACS** integration card for logo display

## References

- [Home Assistant Integration Manifest Documentation](https://developers.home-assistant.io/docs/creating_integration_manifest)
- [Home Assistant Brands Repository](https://github.com/home-assistant/brands)
- [HACS Publishing Documentation](https://hacs.xyz/docs/publish/integration/)

## Maintenance

When updating the logo:

1. Replace `EffektGuard-logo.png` with new version (1024x1024 recommended)
2. Run the image processing script to regenerate icon sizes:

```python
from PIL import Image
import os

img = Image.open('EffektGuard-logo.png')
icon_dir = 'custom_components/effektguard/icons'
os.makedirs(icon_dir, exist_ok=True)

# Create all required sizes
img.resize((256, 256), Image.Resampling.LANCZOS).save(f'{icon_dir}/icon.png', optimize=True)
img.resize((512, 512), Image.Resampling.LANCZOS).save(f'{icon_dir}/icon@2x.png', optimize=True)
img.resize((256, 256), Image.Resampling.LANCZOS).save(f'{icon_dir}/logo.png', optimize=True)
img.resize((512, 512), Image.Resampling.LANCZOS).save(f'{icon_dir}/logo@2x.png', optimize=True)
```

3. Increment version in `manifest.json`
4. Create new release

---

**Status**: ✅ Logo implementation complete and ready for testing
