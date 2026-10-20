# Logo Implementation for EffektGuard

## Where the images live

| Folder | Files | Used by |
|---|---|---|
| `custom_components/effektguard/brand/` | `icon.png`, `icon@2x.png` | Home Assistant's UI (integration cards, Devices & Services, config flow) |
| `icons/` (repository root) | `icon.png`, `icon@2x.png`, `logo.png`, `logo@2x.png` | The README and GitHub |

## How Home Assistant finds them

Home Assistant's `brands` integration serves a custom integration's images from a `brand/` folder in
the integration's directory, when there is one (`Integration.has_branding`). A missing image falls
back to the next one that exists - `logo.png` to `icon.png`, `icon@2x.png` to `icon.png`, the dark
variants to the light ones (`homeassistant/components/brands/const.py`, `IMAGE_FALLBACKS`). So
`brand/` holds only the two icon files: the logo requests fall back to them.

The `brand/` folder was added in #56 ("Add local brand assets"). An install on an older Home
Assistant without the `brands` integration ignores the folder.

## Specifications

- PNG, square
- `icon.png` 256 × 256 pixels, `icon@2x.png` 512 × 512 pixels

## Updating the images

1. Replace the files in `custom_components/effektguard/brand/` (and in `icons/` for the README)
2. Keep the sizes above
3. Release as usual (`docs/agent/RELEASING.md`)

## References

- [Home Assistant Brands Repository](https://github.com/home-assistant/brands)
- [HACS Publishing Documentation](https://hacs.xyz/docs/publish/integration/)
