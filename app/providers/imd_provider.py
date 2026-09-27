"""
IMD (India Meteorological Department) / NCUM Provider.
"""

from typing import Any, Dict
from app.providers.gefs_provider import GEFSProvider


class IMDProvider(GEFSProvider):
    """India Meteorological Department NCUM 4km Regional High-Resolution Provider."""

    @property
    def provider_id(self) -> str:
        return "imd"

    @property
    def name(self) -> str:
        return "IMD NCUM (National Centre Unified Model)"

    @property
    def description(self) -> str:
        return "Ministry of Earth Sciences / NCMRWF Unified Model at 4km regional resolution"

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["provider_id"] = self.provider_id
        status["name"] = self.name
        status["members"] = 1
        status["resolution_deg"] = 0.04
        return status
