"""NIBE heat pump model profiles.

Supported models:
- F730: 6kW ASHP (widely used, replaced by F735 in current lineup)
- F750: 8kW ASHP (widely used, not in current NIBE lineup)
- F2040: 12-16kW ASHP (widely used, not in current NIBE lineup)
- S1155: 3-12kW GSHP mid-range (VERIFIED - current NIBE model)
"""

from .f730 import NibeF730Profile
from .f750 import NibeF750Profile
from .f2040 import NibeF2040Profile
from .s1155 import NibeS1155Profile

__all__ = [
    "NibeF730Profile",
    "NibeF750Profile",
    "NibeF2040Profile",
    "NibeS1155Profile",
]
