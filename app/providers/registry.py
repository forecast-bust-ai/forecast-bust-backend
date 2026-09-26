"""
Provider Registry for Meteorological Data Feeds.
Allows dynamic selection and querying of GEFS, GFS, ECMWF, IMD, and ERA5 providers.
"""

from typing import Dict, List, Optional
from app.providers.base import ForecastProvider
from app.providers.gefs_provider import GEFSProvider
from app.providers.gfs_provider import GFSProvider
from app.providers.ecmwf_provider import ECMWFProvider
from app.providers.imd_provider import IMDProvider
from app.providers.era5_provider import ERA5Provider


class ProviderRegistry:
    """Singleton registry holding registered forecast data providers."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProviderRegistry, cls).__new__(cls)
            cls._instance._init_providers()
        return cls._instance

    def _init_providers(self):
        self._providers: Dict[str, ForecastProvider] = {
            "gefs": GEFSProvider(),
            "gfs": GFSProvider(),
            "ecmwf": ECMWFProvider(),
            "imd": IMDProvider(),
            "era5": ERA5Provider(),
        }

    def get_provider(self, provider_id: str = "gefs") -> ForecastProvider:
        """Get provider instance by ID, defaulting to 'gefs'."""
        normalized = (provider_id or "gefs").lower().strip()
        if normalized not in self._providers:
            # Fallback to GEFS
            return self._providers["gefs"]
        return self._providers[normalized]

    def list_providers(self) -> List[Dict]:
        """List metadata and health status for all registered providers."""
        return [p.get_status() for p in self._providers.values()]


def get_provider_registry() -> ProviderRegistry:
    return ProviderRegistry()
