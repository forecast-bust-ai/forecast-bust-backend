"""
ERA5 (ECMWF Reanalysis v5) Provider.
Serves as the high-accuracy meteorological ground-truth verification reference.
"""

from typing import Any, Dict
from app.providers.gefs_provider import GEFSProvider


class ERA5Provider(GEFSProvider):
    """ECMWF ERA5 Reanalysis Atmospheric Ground Truth Reference Provider."""

    @property
    def provider_id(self) -> str:
        return "era5"

    @property
    def name(self) -> str:
        return "ERA5 Global Atmospheric Reanalysis (Ground Truth)"

    @property
    def description(self) -> str:
        return "ECMWF fifth-generation reanalysis assimilated with satellite and in-situ observations"

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["provider_id"] = self.provider_id
        status["name"] = self.name
        status["resolution_deg"] = 0.25
        return status
