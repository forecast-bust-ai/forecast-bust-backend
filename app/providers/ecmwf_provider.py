"""
ECMWF (European Centre for Medium-Range Weather Forecasts) Provider.
"""

from typing import Any, Dict
from app.providers.gefs_provider import GEFSProvider


class ECMWFProvider(GEFSProvider):
    """ECMWF Integrated Forecasting System (IFS) / OpenData Provider."""

    @property
    def provider_id(self) -> str:
        return "ecmwf"

    @property
    def name(self) -> str:
        return "ECMWF IFS (Integrated Forecasting System)"

    @property
    def description(self) -> str:
        return "European Centre 0.4° high-resolution global medium-range forecast model"

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["provider_id"] = self.provider_id
        status["name"] = self.name
        status["members"] = 51
        status["resolution_deg"] = 0.4
        return status
