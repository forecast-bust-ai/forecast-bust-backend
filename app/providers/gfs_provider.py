"""
GFS (Global Forecast System) Deterministic High-Resolution Provider.
"""

from typing import Any, Dict, List
from app.providers.gefs_provider import GEFSProvider


class GFSProvider(GEFSProvider):
    """NOAA GFS Deterministic 0.25° NWP Provider."""

    @property
    def provider_id(self) -> str:
        return "gfs"

    @property
    def name(self) -> str:
        return "NOAA GFS (Global Forecast System - Deterministic)"

    @property
    def description(self) -> str:
        return "High-resolution 0.25° deterministic numerical weather prediction model"

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["provider_id"] = self.provider_id
        status["name"] = self.name
        status["members"] = 1
        status["resolution_deg"] = 0.25
        return status
