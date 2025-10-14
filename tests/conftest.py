"""Pytest configuration for EffektGuard tests."""

import sys
import warnings
from pathlib import Path

# Add custom_components to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

# Filter warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="josepy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="acme")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="homeassistant")
