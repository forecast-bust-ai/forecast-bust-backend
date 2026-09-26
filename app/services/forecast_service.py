"""
Forecast Service Module.
Loads gridded forecast layers from NetCDF datasets for visualization.
"""

from pathlib import Path
import numpy as np
import xarray as xr

from app.config import DATASET_PATH
from app.schemas.forecast import ForecastMapResponse, ForecastPoint


class ForecastService:
    """Service for loading and querying gridded forecast layers."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ForecastService, cls).__new__(cls)
        return cls._instance

    def get_forecast_map(self, lead_time_hours: int = 24, stride: int = 4) -> ForecastMapResponse:
        """Get gridded GFS forecast precipitation layer."""
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")

        ds = xr.open_dataset(DATASET_PATH, decode_timedelta=False)

        available_lts = [int(v) for v in ds["lead_time"].values]
        if lead_time_hours in available_lts:
            lt_sel = lead_time_hours
        else:
            lt_sel = min(available_lts, key=lambda x: abs(x - lead_time_hours))

        ds_lt = ds.sel(lead_time=lt_sel)

        init_time = str(ds.attrs.get("initialization_time", ""))
        valid_time = str(ds_lt["valid_time"].values) if "valid_time" in ds_lt.coords else ""

        lats = ds_lt["latitude"].values[::stride]
        lons = ds_lt["longitude"].values[::stride]
        fcst_p = ds_lt["forecast_precipitation"].values[::stride, ::stride]

        has_verif = "verification_precipitation" in ds_lt.data_vars
        verif_p = ds_lt["verification_precipitation"].values[::stride, ::stride] if has_verif else None

        has_abs_err = "absolute_error" in ds_lt.data_vars
        abs_err = ds_lt["absolute_error"].values[::stride, ::stride] if has_abs_err else None

        has_bust = "bust" in ds_lt.data_vars
        bust_vals = ds_lt["bust"].values[::stride, ::stride] if has_bust else None

        points = []
        for i in range(len(lats)):
            for j in range(len(lons)):
                pt = ForecastPoint(
                    latitude=float(lats[i]),
                    longitude=float(lons[j]),
                    forecast_precipitation_mm=float(np.round(fcst_p[i, j], 2)),
                    verification_precipitation_mm=float(np.round(verif_p[i, j], 2)) if verif_p is not None else None,
                    absolute_error_mm=float(np.round(abs_err[i, j], 2)) if abs_err is not None else None,
                    is_bust_actual=bool(bust_vals[i, j] == 1) if bust_vals is not None else None,
                )
                points.append(pt)

        return ForecastMapResponse(
            lead_time_hours=lt_sel,
            total_points=len(points),
            initialization_time=init_time,
            valid_time=valid_time,
            points=points,
        )


def get_forecast_service() -> ForecastService:
    return ForecastService()
