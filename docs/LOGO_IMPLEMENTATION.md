# Logo and icons

## Where the files are

| Folder | Files | Used by |
|---|---|---|
| `custom_components/effektguard/brand/` | `icon.png` 256 × 256, `icon@2x.png` 512 × 512 | Home Assistant's UI (integration cards, Devices & services, config flow) |
| `icons/` (repository root) | the same two, plus `logo.png` 256 × 256 and `logo@2x.png` 512 × 512 | `README.md` (`icons/logo.png`) |

All PNG, square.

## How Home Assistant finds them

Home Assistant 2026.3 and later ships a `brands` integration. The frontend requests
`/api/brands/integration/effektguard/<image>`; the view first looks in the custom integration's own
`brand/` folder (`Integration.has_branding` in `homeassistant/loader.py`), then the brands CDN. A
missing file falls back along `IMAGE_FALLBACKS` in `homeassistant/components/brands/const.py`:

- `logo.png` → `icon.png`
- `icon@2x.png` → `icon.png`
- `logo@2x.png` → `logo.png` → `icon.png`
- `dark_icon.png` → `icon.png`
- `dark_logo.png` → `dark_icon.png` → `logo.png` → `icon.png`
- `dark_icon@2x.png` → `icon@2x.png` → `icon.png`
- `dark_logo@2x.png` → `dark_icon@2x.png` → `logo@2x.png` → `logo.png` → `icon.png`

So `brand/` only needs the two icon files. The accepted names are `icon.png`, `logo.png`,
`icon@2x.png`, `logo@2x.png` and their `dark_` variants (`ALLOWED_IMAGES`).

An install older than 2026.3 (`hacs.json` allows 2025.11.0) has no `brands` integration and
ignores the folder: its frontend loads `https://brands.home-assistant.io/effektguard/<image>`,
built from the [home-assistant/brands](https://github.com/home-assistant/brands) repository, where
`custom_integrations/effektguard/` holds the same `icon.png` and `icon@2x.png`. A new icon goes
there too.

## Changing the images

1. Replace the files in `custom_components/effektguard/brand/` and `icons/`, keeping the sizes.
2. Release as usual ([RELEASE_PROCESS.md](RELEASE_PROCESS.md)).
