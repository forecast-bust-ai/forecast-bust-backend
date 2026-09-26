"""
Abstract Base Provider for Meteorological Forecast Data Sources.
Defines contracts for GFS, GEFS, ECMWF, IMD, and ERA5.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ForecastProvider(ABC):
    """Abstract interface for all numerical weather prediction and observation providers."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier (e.g. 'gefs', 'gfs', 'ecmwf', 'imd', 'era5')."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Provider description and resolution metadata."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if the backend data feed or local dataset is active and ready."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Returns provider status, initialization timestamp, resolution, and availability."""
        pass

    @abstractmethod
    def get_available_lead_times(self) -> List[int]:
        """Returns list of available forecast lead times in hours."""
        pass

    @abstractmethod
    def get_layer_data(
        self,
        layer_id: str,
        lead_time_hours: int = 24,
        level_hpa: int = 850,
        threshold_mm: float = 25.0,
        stride: int = 2,
    ) -> Dict[str, Any]:
        """Extract spatial gridded values, bounding box, statistics, and metadata for a layer."""
        pass

    @abstractmethod
    def get_vector_field(
        self,
        vector_id: str,
        lead_time_hours: int = 24,
        level_hpa: int = 850,
        stride: int = 3,
    ) -> Dict[str, Any]:
        """Return u and v wind/flow vector components for animated particle or streamline visualization."""
        pass

    @abstractmethod
    def get_point_profile(
        self,
        latitude: float,
        longitude: float,
        lead_time_hours: int = 24,
    ) -> Dict[str, Any]:
        """Return full vertical atmospheric sounding & diagnostic profile at a specific point."""
        pass

    @abstractmethod
    def get_monsoon_regime(self, lead_time_hours: int = 24) -> Dict[str, Any]:
        """Compute regime classification, Webster-Yang index, and monsoon diagnostic indicators."""
        pass

    @abstractmethod
    def get_ensemble_analysis(
        self,
        lead_time_hours: int = 24,
        threshold_mm: float = 25.0,
        stride: int = 2,
    ) -> Dict[str, Any]:
        """Compute ensemble mean, spread, quantiles (P10-P90), and exceedance probabilities."""
        pass

    @abstractmethod
    def get_confidence_map(
        self,
        lead_time_hours: int = 24,
        stride: int = 2,
    ) -> Dict[str, Any]:
        """Compute spatial forecast confidence grid, error-prone areas, and summary statistics."""
        pass

    @abstractmethod
    def get_bust_probability_map(
        self,
        lead_time_hours: int = 24,
        stride: int = 2,
    ) -> Dict[str, Any]:
        """Compute calibrated forecast bust probability grid across domain."""
        pass

    @abstractmethod
    def get_expected_error_map(
        self,
        lead_time_hours: int = 24,
        stride: int = 2,
    ) -> Dict[str, Any]:
        """Compute expected continuous forecast error field (mm / units)."""
        pass

    @abstractmethod
    def get_model_disagreement_map(
        self,
        lead_time_hours: int = 24,
        stride: int = 2,
    ) -> Dict[str, Any]:
        """Compute multi-model disagreement index (MDI) and inter-model spread."""
        pass

    @abstractmethod
    def get_explanation(
        self,
        latitude: float,
        longitude: float,
        lead_time_hours: int = 24,
    ) -> Dict[str, Any]:
        """Return SHAP feature attributions and transparent meteorological explanations for why confidence is low."""
        pass

