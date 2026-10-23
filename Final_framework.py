"""Orbit-to-detectability pipeline for NH3 monitoring over the Netherlands.

Propagates candidate orbits, generates synthetic footprints, samples environmental
and emission fields, computes detectability, and exports diagnostic results.
"""
import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "ORBIT_PIPELINE_DATA_ROOT",
        "./data"
    )
)

HOURLY_TC_DIR = DATA_ROOT / "thermal_contrast"

ECMWF_TCC_DIR = (
    DATA_ROOT
    / "ecmwf_tcc"
    / "{yyyy}"
)

DIURNAL_FACTOR_PATH = (
    DATA_ROOT
    / "temporal_profiles"
    / "hourfactor_sza_gridded_2019.nc"
)

SEASONAL_FACTOR_PATH = (
    DATA_ROOT
    / "temporal_profiles"
    / "daily_seasonal_factors_agriculture-manure_2019.nc"
)

EMISSIONS_PATH = (
    DATA_ROOT
    / "emissions"
    / "Base-emissions-CAMS-REG_v8_0_2022.nc"
)

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle
import time
import warnings

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from tudatpy.astro import element_conversion
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.interface import spice
from tudatpy.util import result2array

# Load standard SPICE kernels once before creating the propagation environment.
spice.load_standard_kernels()




# Time convention:
# - Orbit-case start and end times are specified as timezone-aware UTC datetimes.
# - Tudat propagation epochs use SPICE ephemeris time (ET/TDB), in seconds
#   since J2000.
# - Propagated epochs are converted back to approximate UTC before matching
#   hourly environmental and emission datasets.


def utc_datetime_to_et(dt_utc: datetime) -> float:
    """
    Convert a timezone-aware UTC datetime to SPICE ephemeris time (ET/TDB)
    in seconds since J2000.
    """
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware UTC")
    s = dt_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return spice.convert_date_string_to_ephemeris_time(s)


def et_to_utc_timestamp(epoch_et: float) -> pd.Timestamp:
    """
    Convert ET/TDB seconds since J2000 to an (approximate) UTC timestamp.

    In some builds, spice.get_approximate_utc_from_tdb() may return:
      - a UTC string  (ideal), OR
      - a numeric value (seconds), depending on wrapper/version.

    This function handles both safely.
    """
    out = spice.get_approximate_utc_from_tdb(float(epoch_et))

    # Case 1: "out" is already a string like "2008 APR 28 00:00:00.000"
    if isinstance(out, str):
        return pd.to_datetime(out, utc=True)

    # # Some TudatPy/SPICE wrapper versions return a numeric approximation instead
    # of a formatted UTC string. In that case, interpret the returned value using
    # the J2000 UTC reference used throughout this pipeline.

    if isinstance(out, (float, int, np.floating, np.integer)):
        return pd.Timestamp(J2000_UTC + timedelta(seconds=float(out)))

    # If it's neither, fail
    raise TypeError(f"Unexpected type from get_approximate_utc_from_tdb: {type(out)}; value={out}")

#Define string names for bodies to be created from default.
bodies_to_create = ["Sun","Earth", "Moon", "Mars", "Venus"]

#Using Earth J2000 as global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

#Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation
    )

# Add the spacecraft before assigning aerodynamic and radiation-pressure interfaces.
body_settings.add_empty_settings("Sat-A")

#Creating aerodynamic coefficient interface settings



# Simplified spacecraft model used consistently for all candidate orbits:
# - 3U CubeSat dimensions: 0.3 x 0.1 x 0.1 m;
# - constant mass, drag coefficient and radiation-pressure coefficient;
# - no attitude propagation;
# - orientation-averaged reference areas;
# - cannonball solar-radiation-pressure model.
# These properties are intended for comparative orbit analysis rather than detailed spacecraft-specific aerodynamic or attitude modelling.


reference_area_drag = (4*0.3*0.1 + 2*0.1*0.1)/4    # Orientation-averaged projected area [m^2].
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area_drag, [drag_coefficient, 0.0, 0.0]
)
#Adding the aerodynamic interface to the body settings
body_settings.get("Sat-A").aerodynamic_coefficient_settings = aero_coefficient_settings    # Aerodynamic acceleration uses the atmosphere model contained in TudatPy's default Earth body settings.


# Create radiation pressure settings
reference_area_radiation = (4*0.3*0.1+2*0.1*0.1)/4  

radiation_pressure_coefficient = 1.2
occulting_bodies_dict = dict()
occulting_bodies_dict["Sun"] = ["Earth"]
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )

# Add the radiation pressure interface to the body settings
body_settings.get("Sat-A").radiation_pressure_target_settings = vehicle_target_settings


#Creating system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get("Sat-A").mass = 2.2 #kg


#Propagation setup

# Define bodies that are propagated
bodies_to_propagate = ["Sat-A"]

# Define central bodies of propagation
central_bodies = ["Earth"]

# Define accelerations acting on Sat-A by Sun and Earth.

# Perturbed propagation model used identically for every candidate orbit.
# Earth gravity is represented to degree and order 5. Atmospheric drag and
# cannonball solar radiation pressure are included, together with third-body
# gravity from the Sun, Moon, Mars and Venus. The purpose is consistent
# comparative propagation, not high-precision orbit determination.


spacecraft_acceleration_settings = dict(
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Earth=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
        propagation_setup.acceleration.aerodynamic()
    ],
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Mars=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Venus=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)

# Create global accelerations settings dictionary.
acceleration_settings = {"Sat-A": spacecraft_acceleration_settings}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

#Converting TudatPy output to "track" objects
EARTH_RADIUS_M = bodies.get("Earth").shape_model.average_radius
J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# # Spherical navigation (for swath/polygons)
def bearing_rad(lat1, lon1, lat2, lon2):
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def tudat_to_track(dependent_variables_array):
    # time: dep_vars[:,0] is ET/TDB seconds since J2000 (Tudat/SPICE convention)
    t_et = dependent_variables_array[:, 0].astype(float)

    # Convert each epoch to a UTC timestamp using SPICE
    time_utc = pd.DatetimeIndex([et_to_utc_timestamp(et) for et in t_et])

    # lat / lon
    lat_deg = np.rad2deg(dependent_variables_array[:, 1])
    lon_deg = np.rad2deg(dependent_variables_array[:, 2])
    lon_deg = (lon_deg + 540.0) % 360.0 - 180.0

    alt_km = dependent_variables_array[:, 3] / 1000.0

    # bearing
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)

    brng = np.zeros_like(lat_r)
    if len(lat_r) >= 2:
        brng[1:] = bearing_rad(lat_r[:-1], lon_r[:-1], lat_r[1:], lon_r[1:])
        brng[0] = brng[1]

    track = pd.DataFrame({
        "time_utc": time_utc,
        "lat": lat_deg,
        "lon": lon_deg,
        "alt_km": alt_km,
        "brng_rad": brng,
        "epoch_et": t_et, #Recommended for debugging, if required later
    })
    track["model_hour"] = track["time_utc"].dt.floor("h")  
    return track

def windows_from_coarse_track(track_coarse, pad_seconds=600, max_gap_seconds=1800):
    """
    Convert coarse in-bbox points into merged fine-propagation windows.
    pad_seconds: expand each side of each window
    max_gap_seconds: merge coarse hits that are close in time
    """
    if track_coarse.empty:
        return []

    times = pd.to_datetime(track_coarse["time_utc"], utc=True).sort_values().to_list()

    windows = []
    current_start = times[0]
    current_end = times[0]

    for tt in times[1:]:
        gap = (tt - current_end).total_seconds()
        if gap <= max_gap_seconds:
            current_end = tt
        else:
            windows.append((current_start, current_end))
            current_start = tt
            current_end = tt

    windows.append((current_start, current_end))

    padded = []
    pad = pd.Timedelta(seconds=pad_seconds)
    for ws, we in windows:
        padded.append((ws - pad, we + pad))

    return padded

def merge_overlapping_windows(windows):
    """
    Merge overlapping or directly adjacent windows.
    """
    if not windows:
        return []

    windows = sorted(windows, key=lambda x: x[0])
    merged = [windows[0]]

    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged

# -------------------------
# GLOBAL drift settings (Option A) Used for SGP4 vs Tudat validation. Not applied in the main pipeline
# -------------------------
EXPORT_GLOBAL_TRACK = False
GLOBAL_STEP_SECONDS = 60.0        # output cadence for global drift plot



# ============================================================
# THERMAL CONTRAST SETTINGS
# ============================================================

USE_TC_WEIGHT = True

HOURLY_TC_DIR = DATA_ROOT / "thermal_contrast"

HOURLY_TC_PATTERN = "hourly_t500_tc_{yyyymmdd}.nc"

TC_VARNAME = "thermal_contrast_K"

# TC weighting settings
# tc_bad: below this absolute TC, retrieval usefulness is low
# tc_good: above this absolute TC, retrieval usefulness is high
TC_BAD_K = 1.0
TC_GOOD_K = 6.0


# ============================================================
# CLOUD COVER SETTINGS
# ============================================================

USE_CLOUD_FILTER = True

ECMWF_TCC_DIR = DATA_ROOT / "ecmwf_tcc" / "{yyyy}"


ECMWF_TCC_PATTERN = "tcc_{yyyymmdd}_1h.nc"
TCC_VARNAME = "tcc"

# Binary clear-sky threshold
# Clear observation if total cloud cover <= this value
TCC_THRESHOLD = 0.25

# If needed later, you can set this to np.timedelta64(-1, "h")
ECMWF_TCC_TIME_OFFSET = np.timedelta64(0, "h")


# ============================================================
# DIURNAL NH3 EMISSION FACTOR SETTINGS
# ============================================================

USE_DIURNAL_EMISSIONS = True

DIURNAL_FACTOR_PATH = DATA_ROOT / "temporal_profiles" / "hourfactor_sza_gridded_2019.nc"

DIURNAL_VARNAME = "hourfactor"

# ============================================================
# SEASONAL NH3 EMISSION FACTOR SETTINGS
# ============================================================

USE_SEASONAL_EMISSIONS = True

SEASONAL_FACTOR_PATH = (
    DATA_ROOT
    / "temporal_profiles"
    / "daily_seasonal_factors_agriculture-manure_2019.nc"
)
SEASONAL_VARNAME = "time_factors"

#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------
#IMPORTING values and FOOTPRINT FUNCTIONS FROM TYLER's CODE:
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
# SZA caculation function
def solar_zenith_angle_deg(time_utc, lat_deg, lon_deg):
    """
    Approximate solar zenith angle (SZA) in degrees using a NOAA-style algorithm.

    Inputs
    ------
    time_utc : pandas.Timestamp or datetime (UTC)
    lat_deg, lon_deg : scalar values or arrays

    Returns
    -------
    sza_deg : float or np.ndarray
    """
    t = pd.to_datetime(time_utc, utc=True)
    if isinstance(t, pd.Timestamp):
        t = pd.DatetimeIndex([t])

    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.asarray(lon_deg, dtype=float)

    # Day of year and fractional hour
    doy = t.dayofyear.to_numpy(dtype=float)
    hour = t.hour.to_numpy(dtype=float)
    minute = t.minute.to_numpy(dtype=float)
    second = t.second.to_numpy(dtype=float)
    frac_hour = hour + minute / 60.0 + second / 3600.0

    # Fractional year
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (frac_hour - 12.0) / 24.0)

    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    # Solar declination
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    # True solar time
    time_offset = eqtime + 4.0 * lon
    tst = (frac_hour * 60.0 + time_offset) % 1440.0

    # Hour angle
    ha_deg = tst / 4.0 - 180.0
    ha = np.deg2rad(ha_deg)

    # Solar zenith angle
    cos_sza = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cos_sza = np.clip(cos_sza, -1.0, 1.0)

    sza = np.rad2deg(np.arccos(cos_sza))

    # Return scalar if scalar input
    if sza.size == 1:
        return float(sza[0])

    return sza


def local_solar_hour_decimal(time_utc, lon_deg):
    """
    Compute approximate true local solar time in decimal hours.

    Uses:
    true solar time = UTC time + equation of time correction + longitude correction

    Output:
    Decimal hour in range [0, 24).
    Example: 10.5 means 10:30 local solar time.

    This is consistent with the solar-time logic used in solar_zenith_angle_deg().
    """

    t_raw = pd.to_datetime(time_utc, utc=True)

    scalar_input = isinstance(t_raw, pd.Timestamp)

    if scalar_input:
        t = pd.DatetimeIndex([t_raw])
    else:
        t = pd.DatetimeIndex(t_raw)

    lon = np.asarray(lon_deg, dtype=float)

    if lon.ndim == 0:
        lon = np.array([float(lon)])

    # UTC decimal hour
    utc_hour = (
        t.hour.astype(float)
        + t.minute.astype(float) / 60.0
        + t.second.astype(float) / 3600.0
    )

    # Day of year
    doy = t.dayofyear.astype(float)

    # Fractional year, same convention as your SZA function
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (utc_hour - 12.0) / 24.0)

    # Equation of time in minutes
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )

    # True solar time in minutes
    true_solar_minutes = (utc_hour * 60.0 + eqtime + 4.0 * lon) % 1440.0

    local_hour = true_solar_minutes / 60.0

    if scalar_input:
        return float(local_hour[0])

    return local_hour


# Build IFOV-based scanline polygons using spherical-Earth geometry.


def build_scanline_polys_ifov(i_scan, track_row, brng_rad):
    lat0 = np.deg2rad(track_row["lat"])
    lon0 = np.deg2rad(track_row["lon"])
    b = brng_rad
    b_left = b - np.pi / 2.0
    b_right = b + np.pi / 2.0

    # Altitude
    H = float(track_row["alt_km"]) * 1000.0 if USE_DYNAMIC_ALTITUDE else 824000.0
    R = EARTH_RADIUS_M

    # -------------------------
    # Determine IFOV and theta grid
    # -------------------------
    if GEOM_MODE == "fixed_swath":
        # Old behavior: swath width fixed, derive nadir spacing
        nadir_spacing_m = (SWATH_WIDTH_KM * 1000.0) / N_ACROSS
        ifov_x = nadir_spacing_m / H
        ifov_y = ifov_x if SQUARE_AT_NADIR else ifov_x

        theta_max = np.arctan2((SWATH_WIDTH_KM * 1000.0) / 2.0, H)
        theta = np.linspace(-theta_max, theta_max, N_ACROSS)

    elif GEOM_MODE == "fixed_nadir":
        # New behavior: nadir pixel size fixed, swath width becomes variable
        nadir_spacing_m = float(NADIR_PIXEL_M)
        ifov_x = nadir_spacing_m / H
        ifov_y = ifov_x if SQUARE_AT_NADIR else ifov_x

        # Pixel centers equally spaced in scan angle by IFOV (small-angle)
        j = np.arange(N_ACROSS, dtype=float)
        theta = (j - (N_ACROSS - 1) / 2.0) * ifov_x

        # safety clamp to horizon if someone picks crazy N_ACROSS / NADIR_PIXEL_M / H
        theta_h = np.arcsin(R / (R + H))
        theta = np.clip(theta, -theta_h, theta_h)

        # For bookkeeping / metadata
        # (This is "effective swath width at this altitude" at the surface)
        # We'll compute it later from centers.

    else:
        raise ValueError(f"Unknown GEOM_MODE: {GEOM_MODE}")

    # -------------------------
    # Cross-track center positions + widths on spherical Earth
    # -------------------------
    if USE_EARTH_CURVATURE:
        psi_c, slant = ground_central_angle_from_look(theta, H, R_m=R)
        x_center = R * psi_c  # signed arc distance (m), left(-) to right(+)

        # Across-track width on the surface from IFOV bounds
        psi_l, _ = ground_central_angle_from_look(theta - 0.5 * ifov_x, H, R_m=R)
        psi_r, _ = ground_central_angle_from_look(theta + 0.5 * ifov_x, H, R_m=R)
        w_x = R * np.abs(psi_r - psi_l)

        # Along-track length: keep your first-pass secant scaling (good enough for now),
        # but use slant-range magnification rather than H.
        # Near edges, slant increases, so this grows naturally.
        w_y = slant * ifov_y

    else:
        # Flat approximations (your old behavior)
        x_center = H * np.tan(theta)
        sec = 1.0 / np.cos(theta)
        w_x = H * ifov_x * sec
        w_y = H * ifov_y * sec

    # Compute pixel center lat/lon from cross-track displacement along great-circle
    lat_c = np.empty(N_ACROSS, dtype=float)
    lon_c = np.empty(N_ACROSS, dtype=float)

    for j in range(N_ACROSS):
        dist = abs(x_center[j])
        if x_center[j] < 0:
            lat_c[j], lon_c[j] = destination_point(lat0, lon0, b_left, dist)
        else:
            lat_c[j], lon_c[j] = destination_point(lat0, lon0, b_right, dist)

    # Corners: build from variable half sizes
    half_w = w_x / 2.0
    half_l = w_y / 2.0

    out = []
    t = track_row["time_utc"]
    mh = track_row["model_hour"]

    for j in range(N_ACROSS):
        # forward/back midpoints along-track
        lat_f, lon_f = destination_point(lat_c[j], lon_c[j], b,         half_l[j])
        lat_b, lon_b = destination_point(lat_c[j], lon_c[j], b + np.pi, half_l[j])

        # corners from forward/back midpoints
        lat_ul, lon_ul = destination_point(lat_f, lon_f, b_left,  half_w[j])
        lat_ur, lon_ur = destination_point(lat_f, lon_f, b_right, half_w[j])
        lat_ll, lon_ll = destination_point(lat_b, lon_b, b_left,  half_w[j])
        lat_lr, lon_lr = destination_point(lat_b, lon_b, b_right, half_w[j])

        # Viewing geometry
        theta_rad = theta[j]
        theta_deg = np.rad2deg(theta_rad)

        # SZA calc approxmimation
        sza_deg = solar_zenith_angle_deg(
            track_row["time_utc"],
            np.rad2deg(lat_c[j]),
            np.rad2deg(lon_c[j])
        )

        out.append(
            {
                "time_utc": t,
                "model_hour": mh,
                "i_scan": int(i_scan),
                "j_across": int(j),
                "theta_rad": float(theta_rad),
                "theta_deg": float(theta_deg),
                "sec_theta": float(1.0 / np.cos(theta_rad)),  # slant path factor
                "sza_deg": float(sza_deg),
                "w_x_m": float(w_x[j]),
                "w_y_m": float(w_y[j]),
                "lat_c": float(np.rad2deg(lat_c[j])),
                "lon_c": float(np.rad2deg(lon_c[j])),
                "lat_ul": float(np.rad2deg(lat_ul)), "lon_ul": float(np.rad2deg(lon_ul)),
                "lat_ur": float(np.rad2deg(lat_ur)), "lon_ur": float(np.rad2deg(lon_ur)),
                "lat_lr": float(np.rad2deg(lat_lr)), "lon_lr": float(np.rad2deg(lon_lr)),
                "lat_ll": float(np.rad2deg(lat_ll)), "lon_ll": float(np.rad2deg(lon_ll)),
            }
        )

    return pd.DataFrame(out)


def ground_central_angle_from_look(theta_rad, H_m, R_m=EARTH_RADIUS_M):
    """
    Spherical Earth: given off-nadir look angle theta at the satellite,
    return central angle psi between subsatellite point and ground intercept.

    Geometry is in a 2D plane containing Earth center and the look direction.

    Returns
    -------
    psi_rad : float or np.ndarray
        Central angle (radians). Valid for |theta| <= asin(R/(R+H)).
    slant_range_m : float or np.ndarray
        Distance from satellite to intercept (meters).
    """
    Rs = R_m + H_m
    theta = np.asarray(theta_rad, dtype=float)

    # Horizon check
    theta_h = np.arcsin(R_m / Rs)
    if np.any(np.abs(theta) > theta_h + 1e-12):
        # Outside Earth-view; clamp (or raise). Clamping is safer for production.
        theta = np.clip(theta, -theta_h, theta_h)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Solve line-circle intersection in 2D
    # t = Rs*cos(theta) - sqrt(R^2 - (Rs*sin(theta))^2)
    under = R_m**2 - (Rs * sin_t)**2
    under = np.maximum(under, 0.0)
    t = Rs * cos_t - np.sqrt(under)

    # Intercept coordinates (center at origin, satellite at (Rs,0))
    x = Rs - t * cos_t
    y = t * sin_t

    psi = np.arctan2(y, x)
    return psi, t

def destination_point(lat1, lon1, brng, dist_m, R=EARTH_RADIUS_M):
    ang = dist_m / R
    sin_lat1 = np.sin(lat1)
    cos_lat1 = np.cos(lat1)
    sin_ang = np.sin(ang)
    cos_ang = np.cos(ang)

    lat2 = np.arcsin(sin_lat1 * cos_ang + cos_lat1 * sin_ang * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * sin_ang * cos_lat1,
        cos_ang - sin_lat1 * np.sin(lat2),
    )
    lon2 = (lon2 + np.pi) % (2 * np.pi) - np.pi
    return lat2, lon2



# Segment intersects bbox (lon/lat plane; safe for NL, no dateline issues)
def segment_intersects_bbox(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            u = qi / pi
            if pi < 0:
                if u > u2:
                    return False
                if u > u1:
                    u1 = u
            else:
                if u < u1:
                    return False
                if u < u2:
                    u2 = u
    return True


def check_groundtrack_over_netherlands(track):
    """
    Check whether the satellite subsatellite ground track crosses the
    Netherlands analysis domain.

    This checks the ground-track centreline, not merely swath overlap.
    """

    if track is None or track.empty or len(track) < 2:
        print("\n[NL GROUNDTRACK CHECK] Track is empty or too short.")
        return False, pd.DataFrame()

    track_check = (
        track[["time_utc", "lon", "lat", "alt_km"]]
        .copy()
        .sort_values("time_utc")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Check whether saved track points lie directly inside NL
    # --------------------------------------------------------
    point_inside = (
        (track_check["lon"] >= GRID_WEST)
        & (track_check["lon"] <= GRID_EAST)
        & (track_check["lat"] >= GRID_SOUTH)
        & (track_check["lat"] <= GRID_NORTH)
    )

    direct_points = track_check.loc[point_inside].copy()

    # --------------------------------------------------------
    # Check whether a track segment crosses NL between outputs
    # --------------------------------------------------------
    crossing_records = []

    for i in range(len(track_check) - 1):
        row0 = track_check.iloc[i]
        row1 = track_check.iloc[i + 1]

        lon0 = float(row0["lon"])
        lat0 = float(row0["lat"])
        lon1 = float(row1["lon"])
        lat1 = float(row1["lat"])

        # Avoid falsely connecting points across the date line.
        if abs(lon1 - lon0) > 180.0:
            continue

        crosses_nl = segment_intersects_bbox(
            lon0,
            lat0,
            lon1,
            lat1,
            GRID_WEST,
            GRID_EAST,
            GRID_SOUTH,
            GRID_NORTH
        )

        if crosses_nl:
            crossing_records.append({
                "segment_start_time": row0["time_utc"],
                "segment_end_time": row1["time_utc"],
                "start_lon": lon0,
                "start_lat": lat0,
                "end_lon": lon1,
                "end_lat": lat1,
            })

    crossing_segments = pd.DataFrame(crossing_records)

    centreline_crosses_nl = (
        len(direct_points) > 0
        or len(crossing_segments) > 0
    )

    print("\n" + "=" * 80)
    print("NETHERLANDS GROUND-TRACK CHECK")
    print("=" * 80)
    print(
        f"Netherlands domain: "
        f"{GRID_WEST:.2f} to {GRID_EAST:.2f} deg E, "
        f"{GRID_SOUTH:.2f} to {GRID_NORTH:.2f} deg N"
    )
    print(f"Track points directly inside domain: {len(direct_points)}")
    print(f"Track segments intersecting domain: {len(crossing_segments)}")

    if centreline_crosses_nl:
        print("RESULT: YES, the subsatellite ground track crosses the Netherlands domain.")
    else:
        print("RESULT: NO, the subsatellite ground track does not cross the Netherlands domain.")

    if len(direct_points) > 0:
        print("\nFirst direct ground-track point inside the domain:")
        print(
            direct_points[
                ["time_utc", "lon", "lat", "alt_km"]
            ].head(1).to_string(index=False)
        )
    elif len(crossing_segments) > 0:
        print("\nFirst segment crossing the domain:")
        print(crossing_segments.head(1).to_string(index=False))

    print("=" * 80)

    return centreline_crosses_nl, crossing_segments


# LOTOS-EUROS grid mapping
def in_le_grid(lon_deg, lat_deg):
    lon = np.asarray(lon_deg)
    lat = np.asarray(lat_deg)
    return (lon >= GRID_WEST) & (lon < GRID_EAST) & (lat >= GRID_SOUTH) & (lat < GRID_NORTH)

def lonlat_to_ij(lon_deg, lat_deg):
    lon = np.asarray(lon_deg)
    lat = np.asarray(lat_deg)

    ix = np.floor((lon - GRID_WEST) / GRID_DLON).astype(int)
    iy = np.floor((lat - GRID_SOUTH) / GRID_DLAT).astype(int)

    mask = in_le_grid(lon, lat)
    ix = np.where(mask, ix, -1)
    iy = np.where(mask, iy, -1)
    return ix, iy


# Writing to CSO-like format
def write_cso_like_geometry_nc(df_pixels: pd.DataFrame, out_nc: Path, orbit_tag: str):
    """
    Write a CSO-like NetCDF containing only pixel geometry:
      - latitude(pixel), longitude(pixel)
      - time(pixel) as seconds since reference time (first pixel time)
      - latitude_bounds(pixel, corner), longitude_bounds(pixel, corner) with corner=4
    """
    if df_pixels.empty:
        raise ValueError("df_pixels is empty; nothing to write.")

    df = df_pixels.copy()

    # Fill vals
    FILL_FLOAT = -999.5 #np.float32(-9999.9)
    FILL_INT = int(-999)

    # Ensure UTC-aware datetime
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)

    # Sort for stability
    df = df.sort_values("time_utc").reset_index(drop=True)

    # Reference time for this file
    ref_time = df["time_utc"].iloc[0]
    time_s = (df["time_utc"] - ref_time).dt.total_seconds().to_numpy(dtype=np.float64)

    n = len(df)
    pixel_index = np.arange(n, dtype=np.int32)

    lon = df["lon_c"].to_numpy(dtype=np.float64)
    lat = df["lat_c"].to_numpy(dtype=np.float64)

    # Corner order must match your convention (here: UL, UR, LR, LL)
    lon_b = np.vstack([
        df["lon_ul"].to_numpy(dtype=np.float64),
        df["lon_ur"].to_numpy(dtype=np.float64),
        df["lon_lr"].to_numpy(dtype=np.float64),
        df["lon_ll"].to_numpy(dtype=np.float64),
    ]).T  # (pixel, corner)

    lat_b = np.vstack([
        df["lat_ul"].to_numpy(dtype=np.float64),
        df["lat_ur"].to_numpy(dtype=np.float64),
        df["lat_lr"].to_numpy(dtype=np.float64),
        df["lat_ll"].to_numpy(dtype=np.float64),
    ]).T  # (pixel, corner)


    ds = xr.Dataset(
        data_vars=dict(
        longitude=(("pixel",), lon),
        latitude=(("pixel",), lat),
        time=(("pixel",), time_s),
        theta=(("pixel",), df["theta_rad"].to_numpy(dtype=np.float64)),
        sec_theta=(("pixel",), df["sec_theta"].to_numpy(dtype=np.float64)),
        sza=(("pixel",), df["sza_deg"].to_numpy(dtype=np.float64)),
        longitude_bounds=(("pixel", "corner"), lon_b),
        latitude_bounds=(("pixel", "corner"), lat_b),
        ),
        coords=dict(
            pixel=("pixel", pixel_index),
            corner=("corner", np.arange(4, dtype=np.int32)),
        ),
        attrs=dict(
            format="1.0",
            Conventions="CF-1.7",
            author="TNO",
            institution="TNO",
            orbit=str(orbit_tag),
            reference_time=str(ref_time.to_pydatetime()).replace("+00:00", ""),
        )
    )

    # Variable attributes (CSO-like)
    ds["pixel"].attrs.update(dict(_FillValue=FILL_INT, units="1", long_name="pixel index"))
    ds["longitude"].attrs.update(dict(_FillValue=FILL_FLOAT, units="degrees_east", long_name="Longitude"))
    ds["latitude"].attrs.update(dict(_FillValue=FILL_FLOAT, units="degrees_north", long_name="Latitude"))
    ds["time"].attrs.update(dict(
        _FillValue=FILL_FLOAT,
        long_name="offset from reference start time of measurement",
        units=f"seconds since {ref_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
    ))
    ds["theta"].attrs.update(dict(_FillValue=FILL_FLOAT,long_name="Off-nadir viewing angle at satellite",units="radians"))
    ds["sec_theta"].attrs.update(dict(_FillValue=FILL_FLOAT,long_name="Slant path length factor (1/cos(theta))",units="1"))
    ds["sza"].attrs.update(dict(_FillValue=FILL_FLOAT,units="degrees",long_name="Solar zenith angle at pixel center",comment="Auxiliary metadata; Computed from time_utc and pixel center lat/lon."))
    ds["longitude_bounds"].attrs.update(dict(_FillValue=FILL_FLOAT, units="degrees_east", long_name="Longitude bounds"))
    ds["latitude_bounds"].attrs.update(dict(_FillValue=FILL_FLOAT, units="degrees_north", long_name="Latitude bounds"))


    if "thermal_contrast_K" in df.columns:
        ds["thermal_contrast_K"] = (
            ("pixel",),
            df["thermal_contrast_K"].to_numpy(dtype=np.float64)
        )
        ds["thermal_contrast_K"].attrs.update(dict(
            _FillValue=FILL_FLOAT,
            units="K",
            long_name="Thermal contrast, SKT minus T500"
        ))

    if "w_tc" in df.columns:
        ds["w_tc"] = (
            ("pixel",),
            df["w_tc"].to_numpy(dtype=np.float64)
        )
        ds["w_tc"].attrs.update(dict(
            _FillValue=FILL_FLOAT,
            units="1",
            long_name="Thermal contrast quality weight"
        ))

    if "tcc" in df.columns:
        ds["tcc"] = (
            ("pixel",),
            df["tcc"].to_numpy(dtype=np.float64)
        )
        ds["tcc"].attrs.update(dict(
            _FillValue=FILL_FLOAT,
            units="1",
            long_name="ECMWF total cloud cover",
            comment="Total cloud cover fraction sampled at pixel center"
        ))

    if "w_cloud" in df.columns:
        ds["w_cloud"] = (
            ("pixel",),
            df["w_cloud"].to_numpy(dtype=np.float64)
        )
        ds["w_cloud"].attrs.update(dict(
            _FillValue=FILL_FLOAT,
            units="1",
            long_name="Binary cloud-cover quality weight",
            comment=f"1 if TCC <= {TCC_THRESHOLD}, else 0"
        ))


    if "hourfactor" in df.columns:
        ds["hourfactor"] = (
            ("pixel",),
            df["hourfactor"].to_numpy(dtype=np.float64)
        )
        ds["hourfactor"].attrs.update(dict(
            _FillValue=FILL_FLOAT,
            units="1",
            long_name="SZA/solar-based hourly NH3 emission factor",
            comment="Emission scaling factor normalized so daily mean equals 1 per grid cell"
        ))

    if "seasonal_factor" in df.columns:
        ds["seasonal_factor"] = (
            ("pixel",),
            df["seasonal_factor"].to_numpy(dtype=np.float64)
        )
        ds["seasonal_factor"].attrs.update(dict(
            _FillValue=FILL_FLOAT,
            units="1",
            long_name="Agriculture-manure seasonal NH3 emission factor",
            comment="Seasonal scaling factor normalized to yearly mean approximately 1 per grid cell"
        ))

    if "combined_emission_factor" in df.columns:
        ds["combined_emission_factor"] = (
            ("pixel",),
            df["combined_emission_factor"].to_numpy(dtype=np.float64)
        )
        ds["combined_emission_factor"].attrs.update(dict(
            _FillValue=FILL_FLOAT,
            units="1",
            long_name="Combined hourly and seasonal NH3 emission factor",
            comment="hourfactor multiplied by seasonal_factor"
        ))

    # compression + chunking
    enc = {
        "longitude": dict(zlib=True, complevel=4),
        "latitude": dict(zlib=True, complevel=4),
        "time": dict(zlib=True, complevel=4),
        "theta": dict(zlib=True, complevel=4),
        "sec_theta": dict(zlib=True, complevel=4),
        "sza": dict(zlib=True, complevel=4),
        "longitude_bounds": dict(zlib=True, complevel=4),
        "latitude_bounds": dict(zlib=True, complevel=4),
        "pixel": dict(zlib=True, complevel=4),
    }

    for col in ["thermal_contrast_K", "w_tc", "tcc", "w_cloud", "hourfactor", "seasonal_factor","combined_emission_factor",]:
        if col in ds:
            enc[col] = dict(zlib=True, complevel=4)

    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc, encoding=enc)
    return out_nc


'''
OBSERVABILITY CODE MODIFIED TO FIT THE PIPELINE
'''
# ----------------------------------------------
# ----------------------------------------------


# ============================================================
# OBSERVABILITY / DETECTABILITY HELPERS
# Assumes the following already exist in the orbit pipeline:
#   - GRID_WEST, GRID_SOUTH, GRID_DLON, GRID_DLAT, GRID_NX, GRID_NY
#   - lonlat_to_ij(lon_deg, lat_deg)
# ============================================================

# -----------------------------
# 1) Physics helpers
# -----------------------------
# Physical constants (replace binas)
XM_AIR = 28.964e-3        # kg/mol
XM_N     =   14.00670e-3
XM_H     =    1.00790e-3

XM_NH3 = XM_N + XM_H * 3  # kg/mol       # kg/mol
GRAV = 9.80665       # m/s^2
AVOG = 6.02214076e23    # molecules/mol

def calc_alpha(
    mass_air=XM_AIR,
    mass_nh3=XM_NH3,
    pressure_hpa=1000.0,
    wind_kmh=5.0   # A Fixed higher windspeed scenario assumtion of 18kph is used in production calls 
):

    """
    Calculate the enhancement scaling coefficient.

    The function-level default is retained for standalone use. Production
    detectability calls explicitly pass 18 km/h, equivalent to 5 m/s.
    """
    wind_ms = wind_kmh / 3.6
    pressure_pa = pressure_hpa * 1e2

    alpha = 1e9 * (mass_air / mass_nh3) * GRAV / (pressure_pa * wind_ms)
    alpha = alpha / (3600.0 * 1000.0)

    return alpha

def calc_enhancement_ppb(alpha, emission_flux_kg_km2_h, W_km):
    """
    ΔX [ppb] = alpha * E [kg km^-2 h^-1] * W [km]
    """
    return alpha * emission_flux_kg_km2_h * W_km

def ppb_to_molec_cm2_factor(pressure_hpa=1000.0):
    p_pa = pressure_hpa * 1e2
    aircol_kg_m2 = p_pa / GRAV
    aircol_mol_m2 = aircol_kg_m2 / XM_AIR
    aircol_molec_cm2 = (aircol_mol_m2 * AVOG) / 1e4   # m² -> cm²
    return 1e-9 * aircol_molec_cm2   # 1 ppb = 1e-9 mol/mol


def calc_sigma_ppb(sigma_I_molec_cm2, sigma_X_molec_cm2, pressure_hpa=1000.0):
    """
    Combine instrument noise and scene variability noise (both in molec/cm^2),
    return total sigma in ppb.
    """
    fac = ppb_to_molec_cm2_factor(pressure_hpa)
    sigma_tot_molec_cm2 = np.sqrt(sigma_I_molec_cm2**2 + sigma_X_molec_cm2**2)
    return sigma_tot_molec_cm2 / fac


# -----------------------------
# 2) LOTOS-EUROS cell area
# -----------------------------
def le_cell_area_km2(iy, ix=None, R_km=6371.0):
    """
    Approximate area of one LE grid cell on a sphere.
    Depends only on latitude band (iy), not longitude.
    Works for scalars or arrays.
    """
    iy = np.asarray(iy, dtype=float)

    lat0 = np.deg2rad(GRID_SOUTH + iy * GRID_DLAT)
    lat1 = np.deg2rad(GRID_SOUTH + (iy + 1) * GRID_DLAT)
    dlon = np.deg2rad(GRID_DLON)

    area = (R_km**2) * dlon * (np.sin(lat1) - np.sin(lat0))
    return np.abs(area)

def spherical_rect_area_km2(lon_lo, lon_hi, lat_lo, lat_hi, R_km=6371.0):           # To perform weighted overlappong of CAMS cell area to LE cells. (Attempt to regrid better)
    """
    Area of a lon/lat rectangle on a sphere [km^2].
    Inputs in degrees.
    """
    lon1 = np.deg2rad(lon_lo)
    lon2 = np.deg2rad(lon_hi)
    lat1 = np.deg2rad(lat_lo)
    lat2 = np.deg2rad(lat_hi)

    return np.abs((R_km**2) * (lon2 - lon1) * (np.sin(lat2) - np.sin(lat1)))


def build_full_le_grid(emis_cell, value_col, fill_value=0.0):
    """
    Convert sparse emis_cell table (one row per populated LE cell)
    into a full 2D LE grid array of shape (GRID_NY, GRID_NX).

    Parameters
    ----------
    emis_cell : pd.DataFrame
        Must contain columns: iy, ix, and value_col
    value_col : str
        Column to place in grid (e.g. 'nh3_kg_km2_yr', 'nh3_kg_m2_yr')
    fill_value : float
        Value for cells with no emissions (usually 0.0)

    Returns
    -------
    grid : np.ndarray
        Full 2D array [iy, ix]
    """
    grid = np.full((GRID_NY, GRID_NX), fill_value, dtype=float)

    for _, row in emis_cell.iterrows():
        iy = int(row["iy"])
        ix = int(row["ix"])
        grid[iy, ix] = row[value_col]

    return grid


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------


#Simple day/night binary SZA quality weightage

def sza_quality_weight(
    sza_deg,
    daylight_cutoff_deg=100.0   # Binary daylight cutoff for the SZA quality weight.
):
    """
    Simple day/night SZA weighting.

    If SZA <= daylight_cutoff_deg:
        weight = 1.0  -> usable daylight observation

    If SZA > daylight_cutoff_deg:
        weight = 0.0  -> night-side observation, ignored
    """

    sza = np.asarray(sza_deg, dtype=float)

    w = np.where(sza <= daylight_cutoff_deg, 1.0, 0.0)

    return w


def thermal_contrast_quality_weight(
    tc_K,
    tc_good=TC_GOOD_K,
    tc_bad=TC_BAD_K
):
    """
    Convert thermal contrast into a retrieval quality weight.

    Uses absolute thermal contrast:
        TC = SKT - T500

    Large |TC| means stronger surface-atmosphere thermal contrast.
    Small |TC| means weak thermal contrast.

    Returns values between 0 and 1.
    """

    tc_abs = np.abs(np.asarray(tc_K, dtype=float))

    w = np.zeros_like(tc_abs, dtype=float)

    # Fully good
    w[tc_abs >= tc_good] = 1.0

    # Linear transition
    mid = (tc_abs > tc_bad) & (tc_abs < tc_good)
    w[mid] = (tc_abs[mid] - tc_bad) / (tc_good - tc_bad)

    # Bad or missing
    w = np.where(np.isfinite(tc_abs), w, 0.0)

    return np.clip(w, 0.0, 1.0)


def cloud_quality_weight(
    tcc,
    threshold=TCC_THRESHOLD
):
    """
    Binary cloud-cover quality weight.

    TCC is total cloud cover fraction:
        0 = clear
        1 = fully cloudy

    Clear-sky criterion:
        tcc <= threshold  -> usable observation
        tcc > threshold   -> rejected observation
    """

    arr = np.asarray(tcc, dtype=float)

    w = np.where(
        np.isfinite(arr) & (arr <= threshold),
        1.0,
        0.0
    )

    return w

_tcc_cache = {}    #cloud cover

def open_ecmwf_tcc_for_day(yyyymmdd: str) -> xr.Dataset:
    """
    Open ECMWF hourly total cloud cover file for one day.
    Uses caching so each daily file is opened only once.
    """

    if yyyymmdd in _tcc_cache:
        return _tcc_cache[yyyymmdd]

    yyyy = yyyymmdd[:4]
    fdir = Path(str(ECMWF_TCC_DIR).format(yyyy=yyyy))
    fp = fdir / ECMWF_TCC_PATTERN.format(yyyymmdd=yyyymmdd)

    if not fp.exists():
        raise FileNotFoundError(f"Missing ECMWF TCC file for {yyyymmdd}: {fp}")

    ds = xr.open_dataset(fp)

    # Normalize coordinate names if needed
    rename = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)

    for c in ["time", "latitude", "longitude"]:
        if c not in ds.coords:
            raise ValueError(
                f"{fp.name}: missing coordinate '{c}'. "
                f"Available coords: {list(ds.coords)}"
            )

    if TCC_VARNAME not in ds:
        raise ValueError(
            f"{fp.name}: variable '{TCC_VARNAME}' not found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    # Apply optional time offset and force hourly timestamps
    t = ds["time"].values.astype("datetime64[ns]")
    t = (t + ECMWF_TCC_TIME_OFFSET).astype("datetime64[h]").astype("datetime64[ns]")
    ds = ds.assign_coords(time=("time", t)).sortby("time")

    print("\nTCC FILE:", fp)
    print(ds["time"].values[:5])
    print("...")
    print(ds["time"].values[-5:])

    _tcc_cache[yyyymmdd] = ds
    return ds


_tc_cache = {}

def open_hourly_tc_for_day(yyyymmdd: str) -> xr.Dataset:
    """
    Open precomputed hourly T500 thermal contrast file for one day.
    Uses caching so each daily file is opened only once.
    """

    if yyyymmdd in _tc_cache:
        return _tc_cache[yyyymmdd]

    fp = HOURLY_TC_DIR / HOURLY_TC_PATTERN.format(yyyymmdd=yyyymmdd)

    if not fp.exists():
        raise FileNotFoundError(f"Missing hourly TC file for {yyyymmdd}: {fp}")

    ds = xr.open_dataset(fp)
    print("\nTC FILE:", fp)
    print(ds["time"].values[:5])
    print("...")
    print(ds["time"].values[-5:])

    if TC_VARNAME not in ds:
        raise ValueError(
            f"{fp.name}: variable '{TC_VARNAME}' not found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    _tc_cache[yyyymmdd] = ds
    return ds


_diurnal_ds_cache = None

def open_diurnal_factor_file() -> xr.Dataset:
    """
    Open the gridded hourly NH3 emission factor file.

    The file contains:
        hourfactor(longitude, latitude, time)

    The factors are normalized so that the mean hourly factor per day
    equals 1 for each grid cell.
    """
    global _diurnal_ds_cache

    if _diurnal_ds_cache is not None:
        return _diurnal_ds_cache

    if not DIURNAL_FACTOR_PATH.exists():
        raise FileNotFoundError(f"Missing diurnal factor file: {DIURNAL_FACTOR_PATH}")

    ds = xr.open_dataset(DIURNAL_FACTOR_PATH)

    if DIURNAL_VARNAME not in ds:
        raise ValueError(
            f"Variable '{DIURNAL_VARNAME}' not found in {DIURNAL_FACTOR_PATH}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    print("\n========== DIURNAL FACTOR FILE CHECK ==========")
    print(ds)
    print("Using variable:", DIURNAL_VARNAME)

    _diurnal_ds_cache = ds
    return ds


def build_orbit_independent_diurnal_reference():
    """
    Build a completely orbit-independent annual diurnal NH3 emission
    reference curve directly from the full gridded hourfactor dataset.

    The reference is calculated using:
        - every available time in the 2019 diurnal factor file;
        - every diurnal-grid cell inside the Netherlands analysis domain;
        - true local solar time at each longitude;
        - cosine-latitude area weighting;
        - one-hour local-solar-time bins.

    No orbit tracks, observation times, footprints, or sampled pixels
    are used.

    Returns
    -------
    reference_df : pandas.DataFrame
        Columns:
            local_hour_bin
            local_hour_center
            annual_mean_hourfactor
            weighted_sample_count
            time_samples
    """
    ds = open_diurnal_factor_file()
    hf_da = ds[DIURNAL_VARNAME]

    required_coordinates = {
        "time",
        "latitude",
        "longitude",
    }

    missing_coordinates = required_coordinates.difference(
        hf_da.coords
    )

    if missing_coordinates:
        raise ValueError(
            "The diurnal hourfactor variable is missing the "
            f"coordinates: {sorted(missing_coordinates)}"
        )

    # ------------------------------------------------------------
    # Restrict the source dataset to the Netherlands analysis domain
    # ------------------------------------------------------------
    latitude_values = hf_da["latitude"].values.astype(float)
    longitude_values = hf_da["longitude"].values.astype(float)

    latitude_indices = np.where(
        (latitude_values >= GRID_SOUTH)
        & (latitude_values < GRID_NORTH)
    )[0]

    longitude_indices = np.where(
        (longitude_values >= GRID_WEST)
        & (longitude_values < GRID_EAST)
    )[0]

    if len(latitude_indices) == 0:
        raise ValueError(
            "No latitude cells from the diurnal factor file fall "
            "inside the Netherlands analysis domain."
        )

    if len(longitude_indices) == 0:
        raise ValueError(
            "No longitude cells from the diurnal factor file fall "
            "inside the Netherlands analysis domain."
        )

    hf_nl = hf_da.isel(
        latitude=latitude_indices,
        longitude=longitude_indices
    )

    nl_latitudes = (
        hf_nl["latitude"].values.astype(float)
    )

    nl_longitudes = (
        hf_nl["longitude"].values.astype(float)
    )

    time_values = pd.to_datetime(
        hf_nl["time"].values,
        utc=True
    )

    # Latitude weighting accounts for the decreasing physical area
    # of regular longitude-latitude cells towards the poles.
    latitude_weights = np.cos(
        np.deg2rad(nl_latitudes)
    )

    latitude_weights = np.where(
        np.isfinite(latitude_weights)
        & (latitude_weights > 0),
        latitude_weights,
        0.0
    )

    if latitude_weights.sum() <= 0:
        raise ValueError(
            "The latitude-area weights are invalid."
        )

    # One-hour local-solar-time bins:
    # [0,1), [1,2), ..., [23,24).
    weighted_factor_sum = np.zeros(24, dtype=float)
    total_spatial_weight = np.zeros(24, dtype=float)
    raw_sample_count = np.zeros(24, dtype=np.int64)
    time_sample_count = np.zeros(24, dtype=np.int64)

    print(
        "\n========== ORBIT-INDEPENDENT DIURNAL REFERENCE =========="
    )
    print("Diurnal source file:", DIURNAL_FACTOR_PATH)
    print(
        "Netherlands latitude range in source subset:",
        float(np.min(nl_latitudes)),
        "to",
        float(np.max(nl_latitudes))
    )
    print(
        "Netherlands longitude range in source subset:",
        float(np.min(nl_longitudes)),
        "to",
        float(np.max(nl_longitudes))
    )
    print("Number of latitude cells:", len(nl_latitudes))
    print("Number of longitude cells:", len(nl_longitudes))
    print("Number of hourly time samples:", len(time_values))

    # ------------------------------------------------------------
    # Process one source timestamp at a time
    # ------------------------------------------------------------
    # This avoids loading the entire longitude x latitude x time
    # Netherlands subset into memory at once.
    for time_index, timestamp in enumerate(time_values):
        # Force a consistent field order, independent of the original
        # NetCDF dimension order.
        field = (
            hf_nl
            .isel(time=time_index)
            .transpose("longitude", "latitude")
            .values
            .astype(float)
        )

        # field shape:
        #     longitude x latitude
        if field.shape != (
            len(nl_longitudes),
            len(nl_latitudes)
        ):
            raise ValueError(
                "Unexpected diurnal field shape after transposition: "
                f"{field.shape}"
            )

        # Calculate true local solar time separately at every
        # source-grid longitude for this UTC timestamp.
        repeated_time = pd.DatetimeIndex(
            [timestamp] * len(nl_longitudes)
        )

        local_solar_hours = local_solar_hour_decimal(
            repeated_time,
            nl_longitudes
        )

        local_hour_bins = (
            np.floor(local_solar_hours).astype(int) % 24
        )

        # --------------------------------------------------------
        # Accumulate the spatially area-weighted field
        # --------------------------------------------------------
        for longitude_index, local_hour_bin in enumerate(
            local_hour_bins
        ):
            longitude_slice = field[longitude_index, :]

            valid = (
                np.isfinite(longitude_slice)
                & np.isfinite(latitude_weights)
                & (latitude_weights > 0)
            )

            if not np.any(valid):
                continue

            valid_factors = longitude_slice[valid]
            valid_weights = latitude_weights[valid]

            weighted_factor_sum[local_hour_bin] += np.sum(
                valid_factors * valid_weights
            )

            total_spatial_weight[local_hour_bin] += np.sum(
                valid_weights
            )

            raw_sample_count[local_hour_bin] += int(
                np.sum(valid)
            )

            time_sample_count[local_hour_bin] += 1

        if (
            (time_index + 1) % 1000 == 0
            or time_index == len(time_values) - 1
        ):
            print(
                "Processed diurnal source times:",
                f"{time_index + 1:,}/{len(time_values):,}"
            )

    annual_mean_hourfactor = np.divide(
        weighted_factor_sum,
        total_spatial_weight,
        out=np.full(24, np.nan, dtype=float),
        where=total_spatial_weight > 0
    )

    reference_df = pd.DataFrame({
        "local_hour_bin": np.arange(24, dtype=int),
        "local_hour_center": (
            np.arange(24, dtype=float) + 0.5
        ),
        "annual_mean_hourfactor": annual_mean_hourfactor,
        "weighted_sample_count": total_spatial_weight,
        "raw_grid_time_sample_count": raw_sample_count,
        "longitude_time_contributions": time_sample_count,
    })

    print("\nOrbit-independent annual diurnal reference:")
    print(reference_df.to_string(index=False))

    print(
        "Mean of the 24-hour reference curve:",
        np.nanmean(
            reference_df["annual_mean_hourfactor"]
        )
    )

    return reference_df


_seasonal_ds_cache = None

def open_seasonal_factor_file() -> xr.Dataset:
    """
    Open the gridded seasonal NH3 emission factor file.

    The file contains:
        time_factors(time, latitude, longitude)

    These factors are normalized so that the yearly mean is approximately 1
    for each grid cell. They redistribute annual emissions through the year.
    """
    global _seasonal_ds_cache

    if _seasonal_ds_cache is not None:
        return _seasonal_ds_cache

    if not SEASONAL_FACTOR_PATH.exists():
        raise FileNotFoundError(f"Missing seasonal factor file: {SEASONAL_FACTOR_PATH}")

    ds = xr.open_dataset(SEASONAL_FACTOR_PATH)

    if SEASONAL_VARNAME not in ds:
        raise ValueError(
            f"Variable '{SEASONAL_VARNAME}' not found in {SEASONAL_FACTOR_PATH}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    # Normalize coordinate names if needed
    rename = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)

    for c in ["time", "latitude", "longitude"]:
        if c not in ds.coords:
            raise ValueError(
                f"{SEASONAL_FACTOR_PATH.name}: missing coordinate '{c}'. "
                f"Available coords: {list(ds.coords)}"
            )

    # Important:
    # Seasonal file times are at half-hour, e.g. 2019-01-01 00:30.
    # Your footprint model_hour is floored to exact hour, e.g. 2019-01-01 00:00.
    # Therefore we floor the seasonal timestamps to the hour for exact matching.
    t = ds["time"].values.astype("datetime64[ns]")
    t = t.astype("datetime64[h]").astype("datetime64[ns]")
    ds = ds.assign_coords(time=("time", t)).sortby("time")

    print("\n========== SEASONAL FACTOR FILE CHECK ==========")
    print(ds)
    print("Using variable:", SEASONAL_VARNAME)
    print("Seasonal time range:", ds["time"].values[0], "to", ds["time"].values[-1])

    _seasonal_ds_cache = ds
    return ds


def add_diurnal_factor_to_polys(polys_nl: pd.DataFrame) -> pd.DataFrame:
    """
    Add SZA/solar-based hourly NH3 emission factor to each footprint pixel.

    This factor modifies emission strength, not retrieval quality.

    The file hourfactor_sza_gridded_2019.nc has:
        hourfactor(longitude, latitude, time)

    Since the current simulation is now in 2019, the pixel model_hour can be
    matched directly to the 2019 file time coordinate.
    """
    if polys_nl.empty:
        return polys_nl

    if not USE_DIURNAL_EMISSIONS:
        df = polys_nl.copy()
        df["hourfactor"] = 1.0
        return df

    ds = open_diurnal_factor_file()
    hf_da = ds[DIURNAL_VARNAME]

    # Important: file dimensions are (longitude, latitude, time)
    hf_times = hf_da["time"].values.astype("datetime64[ns]")
    hf_lat = hf_da["latitude"].values
    hf_lon = hf_da["longitude"].values

    df = polys_nl.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["model_hour"] = df["time_utc"].dt.floor("h")

    df["hourfactor"] = np.nan

    for hour, df_hour in df.groupby("model_hour"):
        idx = df_hour.index.to_numpy()

        hour_ts = pd.Timestamp(hour)

        if hour_ts.tzinfo is not None:
            hour_ts_naive = hour_ts.tz_convert("UTC").tz_localize(None)
        else:
            hour_ts_naive = hour_ts

        hour_np = np.datetime64(hour_ts_naive, "ns")

        hit = np.where(hf_times == hour_np)[0]

        if hit.size == 0:
            print(f"[WARN] No diurnal factor time match for {hour_np}; leaving NaN.")
            continue

        # field shape is (longitude, latitude)
        field = hf_da.isel(time=int(hit[0])).values

        pix_lat = df.loc[idx, "lat_c"].to_numpy(dtype=float)
        pix_lon = df.loc[idx, "lon_c"].to_numpy(dtype=float)

        lat_idx = np.array([
            np.argmin(np.abs(hf_lat - lat0))
            for lat0 in pix_lat
        ])

        lon_idx = np.array([
            np.argmin(np.abs(hf_lon - lon0))
            for lon0 in pix_lon
        ])

        # Because field dims are (longitude, latitude), indexing is [lon_idx, lat_idx]
        vals = field[lon_idx, lat_idx]

        df.loc[idx, "hourfactor"] = vals

    # Missing values should not silently kill the run
    #missing_fraction = df["hourfactor"].isna().mean()

    # Preserve which pixel values were originally missing before filling them.
    df["hourfactor_was_missing"] = df["hourfactor"].isna()

    missing_fraction = df["hourfactor_was_missing"].mean()

    if missing_fraction > 0:
        print(
            f"[WARN] Missing hourfactor for "
            f"{missing_fraction:.3f} of pixels. Filling with 1.0."
        )
        df["hourfactor"] = df["hourfactor"].fillna(1.0)


    # if missing_fraction > 0:
    #     print(f"[WARN] Missing hourfactor for {missing_fraction:.3f} of pixels. Filling with 1.0.")
    #     df["hourfactor"] = df["hourfactor"].fillna(1.0)

    print("\n========== DIURNAL HOURFACTOR CHECK ==========")
    print("Rows:", len(df))
    print(df["hourfactor"].describe())
    print("Mean sampled hourfactor:", df["hourfactor"].mean())
    print("Min/max sampled time:", df["time_utc"].min(), df["time_utc"].max())

    return df


def add_seasonal_factor_to_polys(polys_nl: pd.DataFrame) -> pd.DataFrame:
    """
    Add gridded seasonal NH3 emission factor to each footprint pixel.

    Adds:
        seasonal_factor

    This factor represents annual/seasonal emission variability and is
    normalized to yearly mean approximately 1 per grid cell.
    """

    if polys_nl.empty:
        return polys_nl

    if not USE_SEASONAL_EMISSIONS:
        df = polys_nl.copy()
        df["seasonal_factor"] = 1.0
        return df

    ds = open_seasonal_factor_file()
    sf_da = ds[SEASONAL_VARNAME]

    # Expected dimensions: time, latitude, longitude
    sf_times = sf_da["time"].values.astype("datetime64[ns]")
    sf_lat = sf_da["latitude"].values
    sf_lon = sf_da["longitude"].values

    df = polys_nl.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["model_hour"] = df["time_utc"].dt.floor("h")

    df["seasonal_factor"] = np.nan

    for hour, df_hour in df.groupby("model_hour"):
        idx = df_hour.index.to_numpy()

        hour_ts = pd.Timestamp(hour)

        if hour_ts.tzinfo is not None:
            hour_ts_naive = hour_ts.tz_convert("UTC").tz_localize(None)
        else:
            hour_ts_naive = hour_ts

        hour_np = np.datetime64(hour_ts_naive, "ns")

        hit = np.where(sf_times == hour_np)[0]

        if hit.size == 0:
            print(f"[WARN] No seasonal factor time match for {hour_np}; leaving NaN.")
            continue

        # field shape expected: latitude, longitude
        field = sf_da.isel(time=int(hit[0])).values

        pix_lat = df.loc[idx, "lat_c"].to_numpy(dtype=float)
        pix_lon = df.loc[idx, "lon_c"].to_numpy(dtype=float)

        lat_idx = np.array([
            np.argmin(np.abs(sf_lat - lat0))
            for lat0 in pix_lat
        ])

        lon_idx = np.array([
            np.argmin(np.abs(sf_lon - lon0))
            for lon0 in pix_lon
        ])

        vals = field[lat_idx, lon_idx]

        df.loc[idx, "seasonal_factor"] = vals

    #missing_fraction = df["seasonal_factor"].isna().mean()

    # Preserve which pixel values were originally missing before filling them.
    df["seasonal_factor_was_missing"] = df["seasonal_factor"].isna()

    missing_fraction = df["seasonal_factor_was_missing"].mean()

    if missing_fraction > 0:
        print(
            f"[WARN] Missing seasonal_factor for "
            f"{missing_fraction:.3f} of pixels. Filling with 1.0."
        )
        df["seasonal_factor"] = df["seasonal_factor"].fillna(1.0)

    # if missing_fraction > 0:
    #     print(f"[WARN] Missing seasonal_factor for {missing_fraction:.3f} of pixels. Filling with 1.0.")
    #     df["seasonal_factor"] = df["seasonal_factor"].fillna(1.0)

    print("\n========== SEASONAL FACTOR CHECK ==========")
    print("Rows:", len(df))
    print(df["seasonal_factor"].describe())
    print("Mean sampled seasonal_factor:", df["seasonal_factor"].mean())
    print("Min/max sampled time:", df["time_utc"].min(), df["time_utc"].max())

    return df

#Helper to attach TC values to polys_nl

def add_tc_fields_to_polys(polys_nl: pd.DataFrame) -> pd.DataFrame:
    """
    Add hourly T500-based thermal contrast to each footprint pixel.

    Tries:
    1. Same-day hourly TC file based on model_hour date.
    2. Previous-day hourly TC file.

    The previous-day fallback is needed because:
        hourly_t500_tc_20200101.nc may contain times from
        2020-01-01 09:00 to 2020-01-02 06:00.
    So a pass at 2020-01-02 05:00 should look in the 20200101 file.
    """

    if polys_nl.empty:
        return polys_nl

    df = polys_nl.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)

    # Use hourly model time, consistent with the rest of your pipeline
    df["model_hour"] = df["time_utc"].dt.floor("h")
    df["tc_date"] = df["model_hour"].dt.strftime("%Y%m%d")

    df["thermal_contrast_K"] = np.nan

    # Diagnostic reason assigned to every pixel.
    df["tc_sampling_status"] = "unassigned"

    for yyyymmdd, df_day in df.groupby("tc_date"):

        # Try opening same-day file once for this date group
        try:
            ds_tc = open_hourly_tc_for_day(yyyymmdd)
            tc_da = ds_tc[TC_VARNAME]
            tc_times = tc_da["time"].values.astype("datetime64[ns]")
            tc_lat = tc_da["latitude"].values
            tc_lon = tc_da["longitude"].values
        except FileNotFoundError:
            print(f"[WARN] No same-day hourly TC file for {yyyymmdd}. Will try previous-day files per hour.")
            ds_tc = None
            tc_da = None
            tc_times = None
            tc_lat = None
            tc_lon = None

        for hour, df_hour in df_day.groupby("model_hour"):

            idx = df_hour.index.to_numpy()

            # Convert pandas Timestamp to timezone-naive numpy datetime64[ns]
            hour_ts = pd.Timestamp(hour)

            if hour_ts.tzinfo is not None:
                hour_ts_naive = hour_ts.tz_convert("UTC").tz_localize(None)
            else:
                hour_ts_naive = hour_ts

            hour_np = np.datetime64(hour_ts_naive, "ns")

            field = None
            tc_lat_use = None
            tc_lon_use = None

            # ------------------------------------------------------------
            # 1. Try same-day TC file
            # ------------------------------------------------------------
            if tc_da is not None:
                hit = np.where(tc_times == hour_np)[0]

                if hit.size > 0:
                    field = tc_da.isel(time=int(hit[0])).values
                    tc_lat_use = tc_lat
                    tc_lon_use = tc_lon

            # ------------------------------------------------------------
            # 2. If same-day failed, try previous-day TC file
            # ------------------------------------------------------------
            if field is None:

                prev_date = (hour_ts_naive - pd.Timedelta(days=1)).strftime("%Y%m%d")

                try:
                    ds_tc_prev = open_hourly_tc_for_day(prev_date)
                    tc_da_prev = ds_tc_prev[TC_VARNAME]
                    tc_times_prev = tc_da_prev["time"].values.astype("datetime64[ns]")

                    hit_prev = np.where(tc_times_prev == hour_np)[0]

                    if hit_prev.size > 0:
                        field = tc_da_prev.isel(time=int(hit_prev[0])).values
                        tc_lat_use = tc_da_prev["latitude"].values
                        tc_lon_use = tc_da_prev["longitude"].values
                    # else:
                    #     print(f"[WARN] No TC time match for {hour_np}; leaving NaN.")
                    #     continue

                    else:
                        print(
                            f"[WARN] No TC time match for {hour_np}; leaving NaN."
                        )
                        df.loc[idx, "tc_sampling_status"] = (
                            "missing_hourly_time_match"
                        )
                        continue


                # except FileNotFoundError:
                #     print(f"[WARN] No TC file available for same day or previous day for {hour_np}; leaving NaN.")
                #     continue

                except FileNotFoundError:
                    print(
                        f"[WARN] No TC file available for same day or "
                        f"previous day for {hour_np}; leaving NaN."
                    )
                    df.loc[idx, "tc_sampling_status"] = (
                        "missing_same_and_previous_day_file"
                    )
                    continue

            # ------------------------------------------------------------
            # Spatial nearest-neighbour TC sampling
            # ------------------------------------------------------------
            pix_lat = df.loc[idx, "lat_c"].to_numpy(dtype=float)
            pix_lon = df.loc[idx, "lon_c"].to_numpy(dtype=float)

            lat_idx = np.array([
                np.argmin(np.abs(tc_lat_use - lat0))
                for lat0 in pix_lat
            ])

            lon_idx = np.array([
                np.argmin(np.abs(tc_lon_use - lon0))
                for lon0 in pix_lon
            ])

            # tc_vals = field[lat_idx, lon_idx]

            # df.loc[idx, "thermal_contrast_K"] = tc_vals

            tc_vals = field[lat_idx, lon_idx]
            df.loc[idx, "thermal_contrast_K"] = tc_vals

            valid_tc = np.isfinite(tc_vals)

            df.loc[idx[valid_tc], "tc_sampling_status"] = "valid"
            df.loc[idx[~valid_tc], "tc_sampling_status"] = (
                "source_field_nan"
            )
    unassigned_tc = (
        df["thermal_contrast_K"].isna()
        & (df["tc_sampling_status"] == "unassigned")
    )

    df.loc[unassigned_tc, "tc_sampling_status"] = (
        "unclassified_missing"
    )
    # ------------------------------------------------------------
    # Convert TC to quality weight
    # ------------------------------------------------------------
    if USE_TC_WEIGHT:
        df["w_tc"] = thermal_contrast_quality_weight(
            df["thermal_contrast_K"].values
        )
    else:
        df["w_tc"] = 1.0

    print("\n========== THERMAL CONTRAST CHECK ==========")
    print("Rows:", len(df))
    print("TC stats:")
    print(df["thermal_contrast_K"].describe())
    print("w_tc stats:")
    print(df["w_tc"].describe())

    valid_tc_fraction = np.isfinite(df["thermal_contrast_K"]).mean()
    print(f"Fraction of pixels with valid TC: {valid_tc_fraction:.3f}")
    print(f"Fraction of pixels missing TC: {1.0 - valid_tc_fraction:.3f}")

    df = df.drop(columns=["tc_date"], errors="ignore")

    return df


# TCC is total cloud cover; TC is thermal contrast.

def add_tcc_fields_to_polys(polys_nl: pd.DataFrame) -> pd.DataFrame:
    """
    Add ECMWF total cloud cover to each footprint pixel.

    TCC is sampled at:
        pixel model_hour
        pixel lat_c/lon_c

    Adds:
        tcc
        w_cloud
    """

    if polys_nl.empty:
        return polys_nl

    df = polys_nl.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["model_hour"] = df["time_utc"].dt.floor("h")
    df["tcc_date"] = df["model_hour"].dt.strftime("%Y%m%d")

    df["tcc"] = np.nan

    # Diagnostic reason assigned to every pixel.
    df["tcc_sampling_status"] = "unassigned"

    for yyyymmdd, df_day in df.groupby("tcc_date"):

        try:
            ds_tcc = open_ecmwf_tcc_for_day(yyyymmdd)
            tcc_da = ds_tcc[TCC_VARNAME]
            tcc_times = tcc_da["time"].values.astype("datetime64[ns]")
            tcc_lat = tcc_da["latitude"].values
            tcc_lon = tcc_da["longitude"].values

        # except FileNotFoundError:
        #     print(f"[WARN] No ECMWF TCC file for {yyyymmdd}; leaving NaN.")
        #     continue


        except FileNotFoundError:
            print(
                f"[WARN] No ECMWF TCC file for "
                f"{yyyymmdd}; leaving NaN."
            )

            df.loc[
                df_day.index,
                "tcc_sampling_status"
            ] = "missing_daily_file"

            continue

        for hour, df_hour in df_day.groupby("model_hour"):

            idx = df_hour.index.to_numpy()

            hour_ts = pd.Timestamp(hour)

            if hour_ts.tzinfo is not None:
                hour_ts_naive = hour_ts.tz_convert("UTC").tz_localize(None)
            else:
                hour_ts_naive = hour_ts

            hour_np = np.datetime64(hour_ts_naive, "ns")

            hit = np.where(tcc_times == hour_np)[0]

            # if hit.size == 0:
            #     print(f"[WARN] No TCC time match for {hour_np}; leaving NaN.")
            #     continue

            if hit.size == 0:
                print(
                    f"[WARN] No TCC time match for "
                    f"{hour_np}; leaving NaN."
                )

                df.loc[
                    idx,
                    "tcc_sampling_status"
                ] = "missing_hourly_time_match"

                continue


            field = tcc_da.isel(time=int(hit[0])).values

            pix_lat = df.loc[idx, "lat_c"].to_numpy(dtype=float)
            pix_lon = df.loc[idx, "lon_c"].to_numpy(dtype=float)

            lat_idx = np.array([
                np.argmin(np.abs(tcc_lat - lat0))
                for lat0 in pix_lat
            ])

            lon_idx = np.array([
                np.argmin(np.abs(tcc_lon - lon0))
                for lon0 in pix_lon
            ])

            # TCC file is expected to be shaped as latitude, longitude
            # tcc_vals = field[lat_idx, lon_idx]

            # df.loc[idx, "tcc"] = tcc_vals

            tcc_vals = field[lat_idx, lon_idx]
            df.loc[idx, "tcc"] = tcc_vals

            valid_tcc = np.isfinite(tcc_vals)

            df.loc[idx[valid_tcc], "tcc_sampling_status"] = "valid"
            df.loc[idx[~valid_tcc], "tcc_sampling_status"] = (
                "source_field_nan"
            )

    unassigned_tcc = (
        df["tcc"].isna()
        & (df["tcc_sampling_status"] == "unassigned")
    )

    df.loc[unassigned_tcc, "tcc_sampling_status"] = (
        "unclassified_missing"
    )


    if USE_CLOUD_FILTER:
        df["w_cloud"] = cloud_quality_weight(df["tcc"].values)
    else:
        df["w_cloud"] = 1.0

    print("\n========== CLOUD COVER CHECK ==========")
    print("Rows:", len(df))
    print("TCC stats:")
    print(df["tcc"].describe())
    print("w_cloud stats:")
    print(df["w_cloud"].describe())

    valid_tcc_fraction = np.isfinite(df["tcc"]).mean()
    print(f"Fraction of pixels with valid TCC: {valid_tcc_fraction:.3f}")
    print(f"Fraction of pixels missing TCC: {1.0 - valid_tcc_fraction:.3f}")
    print(f"Clear-sky threshold: TCC <= {TCC_THRESHOLD}")

    df = df.drop(columns=["tcc_date"], errors="ignore")

    return df

def add_relative_to_peak_hourfactor(
    polys_nl: pd.DataFrame,
    plot_dir: Path,
    orbit_name: str
):
    """
    For each observed pixel, compute how close the sampled hourfactor is
    to the daily maximum hourfactor for the same nearest diurnal grid cell.

    Adds columns:
        daily_peak_hourfactor
        relative_to_peak_hourfactor

    Produces:
        1. Histogram of relative_to_peak_hourfactor
        2. CSV summary by local solar hour
        3. Summary metrics returned as dictionary

    Interpretation:
        relative_to_peak_hourfactor = 1.0 means the observation occurred
        at the daily peak hourfactor for that grid cell/day.

        relative_to_peak_hourfactor = 0.5 means the observation occurred
        when hourfactor was half of the local daily maximum.
    """

    if polys_nl.empty:
        print("[PEAK HOURFACTOR] polys_nl is empty. Skipping.")
        return polys_nl, {}

    if "hourfactor" not in polys_nl.columns:
        print("[PEAK HOURFACTOR] Missing hourfactor. Skipping.")
        return polys_nl, {}

    if "local_solar_hour" not in polys_nl.columns:
        print("[PEAK HOURFACTOR] Missing local_solar_hour. Skipping.")
        return polys_nl, {}

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = polys_nl.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["model_hour"] = df["time_utc"].dt.floor("h")
    df["date_utc"] = df["model_hour"].dt.floor("D")

    # Open the same diurnal factor file already used in the pipeline
    ds = open_diurnal_factor_file()
    hf_da = ds[DIURNAL_VARNAME]

    hf_times = hf_da["time"].values.astype("datetime64[ns]")
    hf_lat = hf_da["latitude"].values
    hf_lon = hf_da["longitude"].values

    # The diurnal file has dimensions (longitude, latitude, time)
    # according to your add_diurnal_factor_to_polys() function.
    hf_values = hf_da.values

    df["daily_peak_hourfactor"] = np.nan

    # ------------------------------------------------------------
    # Loop over dates to compute daily peak hourfactor per pixel
    # ------------------------------------------------------------
    for day, df_day in df.groupby("date_utc"):
        idx = df_day.index.to_numpy()

        day_ts = pd.Timestamp(day)

        if day_ts.tzinfo is not None:
            day_start = day_ts.tz_convert("UTC").tz_localize(None)
        else:
            day_start = day_ts

        day_end = day_start + pd.Timedelta(days=1)

        day_start_np = np.datetime64(day_start, "ns")
        day_end_np = np.datetime64(day_end, "ns")

        day_time_mask = (hf_times >= day_start_np) & (hf_times < day_end_np)

        if not np.any(day_time_mask):
            print(f"[WARN] No diurnal factor times found for date {day_start_np}.")
            continue

        # Subset one day: expected shape remains (longitude, latitude, time_in_day)
        hf_day = hf_values[:, :, day_time_mask]

        pix_lat = df.loc[idx, "lat_c"].to_numpy(dtype=float)
        pix_lon = df.loc[idx, "lon_c"].to_numpy(dtype=float)

        lat_idx = np.array([
            np.argmin(np.abs(hf_lat - lat0))
            for lat0 in pix_lat
        ])

        lon_idx = np.array([
            np.argmin(np.abs(hf_lon - lon0))
            for lon0 in pix_lon
        ])

        # For each pixel, get the 24-hour series at the nearest diurnal grid cell.
        # Because hf_day is shaped (longitude, latitude, time), use [lon_idx, lat_idx, :]
        daily_peak_vals = np.array([
            np.nanmax(hf_day[lo, la, :])
            for lo, la in zip(lon_idx, lat_idx)
        ])

        df.loc[idx, "daily_peak_hourfactor"] = daily_peak_vals

    # ------------------------------------------------------------
    # Compute relative-to-peak metric
    # ------------------------------------------------------------
    df["relative_to_peak_hourfactor"] = np.where(
        np.isfinite(df["daily_peak_hourfactor"]) & (df["daily_peak_hourfactor"] > 0),
        df["hourfactor"] / df["daily_peak_hourfactor"],
        np.nan
    )

    # Numerical safety
    df["relative_to_peak_hourfactor"] = df["relative_to_peak_hourfactor"].clip(lower=0.0)

    # ------------------------------------------------------------
    # Save pixel-level diagnostic summary
    # ------------------------------------------------------------
    cols_to_save = [
        "time_utc",
        "model_hour",
        "lat_c",
        "lon_c",
        "iy",
        "ix",
        "pass_id",
        "i_scan",
        "local_solar_hour",
        "hourfactor",
        "daily_peak_hourfactor",
        "relative_to_peak_hourfactor",
    ]

    available_cols = [c for c in cols_to_save if c in df.columns]

    out_pixel_csv = plot_dir / f"{orbit_name}_relative_to_peak_hourfactor_pixels.csv"
    df[available_cols].to_csv(out_pixel_csv, index=False)

    # ------------------------------------------------------------
    # Summary by local solar hour
    # ------------------------------------------------------------
    df["hour_bin"] = np.floor(df["local_solar_hour"]).astype(int)

    hourly = (
        df.groupby("hour_bin", as_index=False)
        .agg(
            n_pixels=("relative_to_peak_hourfactor", "size"),
            mean_hourfactor=("hourfactor", "mean"),
            mean_daily_peak_hourfactor=("daily_peak_hourfactor", "mean"),
            mean_relative_to_peak=("relative_to_peak_hourfactor", "mean"),
            median_relative_to_peak=("relative_to_peak_hourfactor", "median"),
        )
    )

    out_hourly_csv = plot_dir / f"{orbit_name}_relative_to_peak_by_local_hour.csv"
    hourly.to_csv(out_hourly_csv, index=False)

    # ------------------------------------------------------------
    # Plot 1: histogram of relative-to-peak values
    # ------------------------------------------------------------
    valid_rel = df["relative_to_peak_hourfactor"].to_numpy(dtype=float)
    valid_rel = valid_rel[np.isfinite(valid_rel)]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        valid_rel,
        bins=np.linspace(0.0, 1.5, 31),
        edgecolor="black",
        alpha=0.75
    )

    ax.axvline(
        np.nanmean(valid_rel),
        color="forestgreen",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {np.nanmean(valid_rel):.2f}"
    )

    ax.axvline(
        np.nanmedian(valid_rel),
        color="darkorange",
        linestyle="--",
        linewidth=2,
        label=f"Median = {np.nanmedian(valid_rel):.2f}"
    )

    ax.set_xlabel("Observed hourfactor / daily peak hourfactor")
    ax.set_ylabel("Number of observed pixels")
    ax.set_title(f"Relative-to-peak hourfactor distribution: {orbit_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_hist_png = plot_dir / f"{orbit_name}_relative_to_peak_hourfactor_histogram.png"
    fig.savefig(out_hist_png, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------
    # Plot 2: relative-to-peak versus local solar hour
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(
        hourly["hour_bin"],
        hourly["n_pixels"],
        color="lightgray",
        edgecolor="black",
        alpha=0.7,
        label="Observed pixels"
    )

    ax1.set_xlabel("Local solar time [hour]")
    ax1.set_ylabel("Observed pixel count")
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_xlim(-0.5, 23.5)

    ax2 = ax1.twinx()

    ax2.plot(
        hourly["hour_bin"],
        hourly["mean_relative_to_peak"],
        marker="o",
        color="forestgreen",
        linewidth=2,
        label="Mean relative-to-peak hourfactor"
    )

    ax2.set_ylabel("Observed hourfactor / daily peak hourfactor")
    ax2.set_ylim(0.0, max(1.1, np.nanmax(hourly["mean_relative_to_peak"]) * 1.1))

    ax1.set_title(f"Relative-to-peak hourfactor by local time: {orbit_name}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()

    out_hourly_png = plot_dir / f"{orbit_name}_relative_to_peak_by_local_hour.png"
    fig.savefig(out_hourly_png, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    metrics = {
        "mean_relative_to_peak_hourfactor": float(np.nanmean(valid_rel)) if len(valid_rel) > 0 else np.nan,
        "median_relative_to_peak_hourfactor": float(np.nanmedian(valid_rel)) if len(valid_rel) > 0 else np.nan,
        "p10_relative_to_peak_hourfactor": float(np.nanpercentile(valid_rel, 10)) if len(valid_rel) > 0 else np.nan,
        "p90_relative_to_peak_hourfactor": float(np.nanpercentile(valid_rel, 90)) if len(valid_rel) > 0 else np.nan,

        "frac_obs_relative_to_peak_ge_0p50": float((valid_rel >= 0.50).mean()) if len(valid_rel) > 0 else np.nan,
        "frac_obs_relative_to_peak_ge_0p75": float((valid_rel >= 0.75).mean()) if len(valid_rel) > 0 else np.nan,
        "frac_obs_relative_to_peak_ge_0p90": float((valid_rel >= 0.90).mean()) if len(valid_rel) > 0 else np.nan,

        "mean_daily_peak_hourfactor_sampled_cells": float(np.nanmean(df["daily_peak_hourfactor"].values)),
        "mean_observed_hourfactor_for_peak_diag": float(np.nanmean(df["hourfactor"].values)),
    }

    print("\n========== RELATIVE-TO-PEAK HOURFACTOR DIAGNOSTICS ==========")
    print("Orbit:", orbit_name)
    print("Saved pixel CSV:", out_pixel_csv)
    print("Saved hourly CSV:", out_hourly_csv)
    print("Saved histogram plot:", out_hist_png)
    print("Saved hourly plot:", out_hourly_png)
    print("Mean relative-to-peak hourfactor:", metrics["mean_relative_to_peak_hourfactor"])
    print("Median relative-to-peak hourfactor:", metrics["median_relative_to_peak_hourfactor"])
    print("P90 relative-to-peak hourfactor:", metrics["p90_relative_to_peak_hourfactor"])
    print("Fraction observations >= 50% of daily peak:", metrics["frac_obs_relative_to_peak_ge_0p50"])
    print("Fraction observations >= 75% of daily peak:", metrics["frac_obs_relative_to_peak_ge_0p75"])
    print("Fraction observations >= 90% of daily peak:", metrics["frac_obs_relative_to_peak_ge_0p90"])

    return df, metrics


def calculate_orbit_data_completeness(
    orbit_name: str,
    polys_nl: pd.DataFrame,
    emis_cell: pd.DataFrame
):
    """
    Calculate orbit-level environmental-data completeness diagnostics.

    Environmental missing fractions are pixel weighted:
        missing pixel rows / all pixel rows observed inside the analysis domain.

    The non-emitting-cell fraction is calculated over the complete
    LOTOS-EUROS analysis grid:
        cells without positive NH3 emissions / all LE grid cells.

    Returns
    -------
    diagnostics : dict
        Orbit-level completeness metrics.
    """

    total_grid_cells = int(GRID_NX * GRID_NY)

    # --------------------------------------------------------
    # Fraction of non-emitting LE grid cells
    # --------------------------------------------------------
    if (
        emis_cell is not None
        and not emis_cell.empty
        and {"iy", "ix", "nh3_kg_yr"}.issubset(emis_cell.columns)
    ):
        emissions = emis_cell[["iy", "ix", "nh3_kg_yr"]].copy()

        emissions["nh3_kg_yr"] = pd.to_numeric(
            emissions["nh3_kg_yr"],
            errors="coerce"
        )

        # Protect against accidental duplicate grid-cell rows.
        emissions = (
            emissions.groupby(["iy", "ix"], as_index=False)
            .agg(nh3_kg_yr=("nh3_kg_yr", "sum"))
        )

        positive_emission_cells = int(
            (emissions["nh3_kg_yr"] > 0).sum()
        )

        non_emitting_cells = max(
            total_grid_cells - positive_emission_cells,
            0
        )

        fraction_non_emitting_cells = (
            non_emitting_cells / total_grid_cells
            if total_grid_cells > 0
            else np.nan
        )
    else:
        positive_emission_cells = 0
        non_emitting_cells = total_grid_cells
        fraction_non_emitting_cells = np.nan

    # --------------------------------------------------------
    # Missing environmental factors among observed pixels
    # --------------------------------------------------------
    total_pixel_rows = int(len(polys_nl))

    if total_pixel_rows == 0:
        fraction_missing_tc = np.nan
        fraction_missing_tcc = np.nan
        fraction_missing_diurnal = np.nan
        fraction_missing_seasonal = np.nan

    else:
        # Thermal contrast remains NaN when sampling failed.
        if "thermal_contrast_K" in polys_nl.columns:
            tc_values = pd.to_numeric(
                polys_nl["thermal_contrast_K"],
                errors="coerce"
            )
            fraction_missing_tc = float(tc_values.isna().mean())
        else:
            fraction_missing_tc = 1.0

        # TCC remains NaN when sampling failed.
        if "tcc" in polys_nl.columns:
            tcc_values = pd.to_numeric(
                polys_nl["tcc"],
                errors="coerce"
            )
            fraction_missing_tcc = float(tcc_values.isna().mean())
        else:
            fraction_missing_tcc = 1.0

        # Diurnal missing status was preserved before fillna(1.0).
        if "hourfactor_was_missing" in polys_nl.columns:
            missing_diurnal = (
                polys_nl["hourfactor_was_missing"]
                .fillna(True)
                .astype(bool)
            )
            fraction_missing_diurnal = float(
                missing_diurnal.mean()
            )
        elif "hourfactor" in polys_nl.columns:
            # Fallback for older data that were not yet filled.
            hourfactor_values = pd.to_numeric(
                polys_nl["hourfactor"],
                errors="coerce"
            )
            fraction_missing_diurnal = float(
                hourfactor_values.isna().mean()
            )
        else:
            fraction_missing_diurnal = 1.0

        # Seasonal missing status was preserved before fillna(1.0).
        if "seasonal_factor_was_missing" in polys_nl.columns:
            missing_seasonal = (
                polys_nl["seasonal_factor_was_missing"]
                .fillna(True)
                .astype(bool)
            )
            fraction_missing_seasonal = float(
                missing_seasonal.mean()
            )
        elif "seasonal_factor" in polys_nl.columns:
            # Fallback for older data that were not yet filled.
            seasonal_values = pd.to_numeric(
                polys_nl["seasonal_factor"],
                errors="coerce"
            )
            fraction_missing_seasonal = float(
                seasonal_values.isna().mean()
            )
        else:
            fraction_missing_seasonal = 1.0

    diagnostics = {
        "orbit_name": orbit_name,
        "total_grid_cells": total_grid_cells,
        "positive_emission_cells": positive_emission_cells,
        "non_emitting_cells": non_emitting_cells,
        "fraction_non_emitting_cells": (
            fraction_non_emitting_cells
        ),
        "total_observed_pixel_rows": total_pixel_rows,
        "fraction_missing_tc": fraction_missing_tc,
        "fraction_missing_tcc": fraction_missing_tcc,
        "fraction_missing_diurnal_factor": (
            fraction_missing_diurnal
        ),
        "fraction_missing_seasonal_factor": (
            fraction_missing_seasonal
        ),
    }

    return diagnostics


def generate_environmental_missingness_diagnostics(
    orbit_name: str,
    polys_nl: pd.DataFrame,
    output_dir: Path
):
    """
    Diagnose missing TC and TCC values by:
        - sampling failure reason;
        - month;
        - UTC hour;
        - UTC date;
        - regional pass;
        - LOTOS-EUROS grid cell.

    Reports are printed to the terminal and saved as CSV files.
    """

    if polys_nl.empty:
        print(
            f"[MISSINGNESS] {orbit_name}: polys_nl is empty. "
            "Diagnostics skipped."
        )
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = polys_nl.copy()

    df["time_utc"] = pd.to_datetime(
        df["time_utc"],
        utc=True,
        errors="coerce"
    )

    df["date_utc"] = df["time_utc"].dt.strftime("%Y-%m-%d")
    df["month"] = df["time_utc"].dt.month
    df["utc_hour"] = df["time_utc"].dt.hour

    df["tc_missing"] = (
        pd.to_numeric(
            df.get(
                "thermal_contrast_K",
                pd.Series(np.nan, index=df.index)
            ),
            errors="coerce"
        ).isna()
    )

    df["tcc_missing"] = (
        pd.to_numeric(
            df.get(
                "tcc",
                pd.Series(np.nan, index=df.index)
            ),
            errors="coerce"
        ).isna()
    )

    if "tc_sampling_status" not in df.columns:
        df["tc_sampling_status"] = np.where(
            df["tc_missing"],
            "reason_not_recorded",
            "valid"
        )

    if "tcc_sampling_status" not in df.columns:
        df["tcc_sampling_status"] = np.where(
            df["tcc_missing"],
            "reason_not_recorded",
            "valid"
        )

    total_pixels = len(df)

    def build_group_summary(
        data,
        group_columns,
        output_name
    ):
        summary = (
            data.groupby(
                group_columns,
                dropna=False,
                as_index=False
            )
            .agg(
                total_pixels=("time_utc", "size"),
                missing_tc_pixels=("tc_missing", "sum"),
                missing_tcc_pixels=("tcc_missing", "sum"),
            )
        )

        summary["fraction_missing_tc"] = (
            summary["missing_tc_pixels"]
            / summary["total_pixels"].replace(0, np.nan)
        )

        summary["fraction_missing_tcc"] = (
            summary["missing_tcc_pixels"]
            / summary["total_pixels"].replace(0, np.nan)
        )

        summary.to_csv(
            output_dir / output_name,
            index=False
        )

        return summary

    # --------------------------------------------------------
    # Missingness by month
    # --------------------------------------------------------
    by_month = build_group_summary(
        data=df,
        group_columns=["month"],
        output_name="missing_tc_tcc_by_month.csv"
    )

    # --------------------------------------------------------
    # Missingness by UTC hour
    # --------------------------------------------------------
    by_hour = build_group_summary(
        data=df,
        group_columns=["utc_hour"],
        output_name="missing_tc_tcc_by_utc_hour.csv"
    )

    # --------------------------------------------------------
    # Missingness by UTC date
    # --------------------------------------------------------
    by_date = build_group_summary(
        data=df,
        group_columns=["date_utc"],
        output_name="missing_tc_tcc_by_date.csv"
    )

    # --------------------------------------------------------
    # Missingness by pass
    # --------------------------------------------------------
    by_pass = build_group_summary(
        data=df,
        group_columns=["pass_id"],
        output_name="missing_tc_tcc_by_pass.csv"
    )

    # --------------------------------------------------------
    # Missingness by LE grid cell
    # --------------------------------------------------------
    by_cell = build_group_summary(
        data=df,
        group_columns=["iy", "ix"],
        output_name="missing_tc_tcc_by_grid_cell.csv"
    )

    # Add cell-centre coordinates for spatial inspection.
    by_cell["cell_lon_c"] = (
        GRID_WEST
        + (by_cell["ix"] + 0.5) * GRID_DLON
    )

    by_cell["cell_lat_c"] = (
        GRID_SOUTH
        + (by_cell["iy"] + 0.5) * GRID_DLAT
    )

    by_cell.to_csv(
        output_dir / "missing_tc_tcc_by_grid_cell.csv",
        index=False
    )

    # --------------------------------------------------------
    # Sampling-failure reasons
    # --------------------------------------------------------
    tc_reasons = (
        df.groupby(
            "tc_sampling_status",
            dropna=False,
            as_index=False
        )
        .agg(pixel_count=("time_utc", "size"))
    )

    tc_reasons["fraction_of_all_pixels"] = (
        tc_reasons["pixel_count"] / total_pixels
    )

    tc_reasons.to_csv(
        output_dir / "tc_sampling_status_summary.csv",
        index=False
    )

    tcc_reasons = (
        df.groupby(
            "tcc_sampling_status",
            dropna=False,
            as_index=False
        )
        .agg(pixel_count=("time_utc", "size"))
    )

    tcc_reasons["fraction_of_all_pixels"] = (
        tcc_reasons["pixel_count"] / total_pixels
    )

    tcc_reasons.to_csv(
        output_dir / "tcc_sampling_status_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save a manageable sample of missing pixel rows
    # --------------------------------------------------------
    diagnostic_columns = [
        "time_utc",
        "date_utc",
        "month",
        "utc_hour",
        "pass_id",
        "i_scan",
        "j_across",
        "lat_c",
        "lon_c",
        "iy",
        "ix",
        "thermal_contrast_K",
        "tc_sampling_status",
        "tcc",
        "tcc_sampling_status",
    ]

    diagnostic_columns = [
        column
        for column in diagnostic_columns
        if column in df.columns
    ]

    missing_rows = df.loc[
        df["tc_missing"] | df["tcc_missing"],
        diagnostic_columns
    ].copy()

    # Avoid creating an extremely large CSV while retaining a
    # reproducible sample for manual inspection.
    maximum_saved_rows = 100_000

    if len(missing_rows) > maximum_saved_rows:
        missing_rows_to_save = missing_rows.sample(
            n=maximum_saved_rows,
            random_state=42
        )
    else:
        missing_rows_to_save = missing_rows

    missing_rows_to_save.to_csv(
        output_dir / "missing_tc_tcc_pixel_sample.csv",
        index=False
    )

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"DETAILED TC/TCC MISSINGNESS DIAGNOSTICS: {orbit_name}")
    print("=" * 90)

    print(f"Total observed pixels: {total_pixels:,}")

    print(
        "Missing TC pixels: "
        f"{int(df['tc_missing'].sum()):,} "
        f"({100.0 * df['tc_missing'].mean():.3f}%)"
    )

    print(
        "Missing TCC pixels: "
        f"{int(df['tcc_missing'].sum()):,} "
        f"({100.0 * df['tcc_missing'].mean():.3f}%)"
    )

    print("\nTC sampling status:")
    print(
        tc_reasons.sort_values(
            "pixel_count",
            ascending=False
        ).to_string(index=False)
    )

    print("\nTCC sampling status:")
    print(
        tcc_reasons.sort_values(
            "pixel_count",
            ascending=False
        ).to_string(index=False)
    )

    print("\nMissing TC/TCC by month:")
    print(
        by_month[
            [
                "month",
                "total_pixels",
                "fraction_missing_tc",
                "fraction_missing_tcc",
            ]
        ].to_string(index=False)
    )

    print("\nMissing TC/TCC by UTC hour:")
    print(
        by_hour[
            [
                "utc_hour",
                "total_pixels",
                "fraction_missing_tc",
                "fraction_missing_tcc",
            ]
        ].to_string(index=False)
    )

    print("\nTen dates with the largest TCC missing fractions:")
    print(
        by_date.sort_values(
            [
                "fraction_missing_tcc",
                "missing_tcc_pixels"
            ],
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTen dates with the largest TC missing fractions:")
    print(
        by_date.sort_values(
            [
                "fraction_missing_tc",
                "missing_tc_pixels"
            ],
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTen passes with the largest TCC missing fractions:")
    print(
        by_pass.sort_values(
            [
                "fraction_missing_tcc",
                "missing_tcc_pixels"
            ],
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTen passes with the largest TC missing fractions:")
    print(
        by_pass.sort_values(
            [
                "fraction_missing_tc",
                "missing_tc_pixels"
            ],
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved detailed diagnostics to:")
    print(output_dir)
    print("=" * 90)

    return {
        "fraction_missing_tc": float(
            df["tc_missing"].mean()
        ),
        "fraction_missing_tcc": float(
            df["tcc_missing"].mean()
        ),
        "tc_sampling_status": tc_reasons,
        "tcc_sampling_status": tcc_reasons,
        "by_month": by_month,
        "by_utc_hour": by_hour,
        "by_date": by_date,
        "by_pass": by_pass,
        "by_cell": by_cell,
    }







'''
Observation-count definitions:
N_raw:
    Number of synthetic pixel rows.
N_eff:
    Sum of retrieval-quality weights:
        w_sza * w_tc * w_cloud.
N_sig_eff:
    Sum of signal-effective weights:
        w_sza * w_tc * w_cloud
        * (hourfactor * seasonal_factor)^2.
These are weighted sample-equivalent quantities; N_eff and N_sig_eff need
not be integers. The squared emission factor follows the assumed SNR-squared
accumulation used in the detectability formulation.
'''

def build_sampling_summary_by_pass(polys_nl: pd.DataFrame):

    if polys_nl.empty:
        raise ValueError("polys_nl is empty; cannot build sampling summary.")

    df = polys_nl.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)



    # ------------------------------------------------------------
    # Pixel-level SZA weight
    # ------------------------------------------------------------
    if "sza_deg" in df.columns:
        df["w_sza_pixel"] = sza_quality_weight(df["sza_deg"].values)
    else:
        df["sza_deg"] = np.nan
        df["w_sza_pixel"] = 1.0

    # ------------------------------------------------------------
    # Pixel-level thermal contrast weight
    # ------------------------------------------------------------
    if "thermal_contrast_K" in df.columns:
        df["w_tc_pixel"] = thermal_contrast_quality_weight(
            df["thermal_contrast_K"].values
        )
    else:
        df["thermal_contrast_K"] = np.nan
        df["w_tc_pixel"] = 1.0


    #   Missing TC/TCC values receive zero retrieval weight



    # ------------------------------------------------------------
    # Pixel-level cloud-cover weight
    # ------------------------------------------------------------
    if "tcc" in df.columns:
        df["tcc"] = pd.to_numeric(df["tcc"], errors="coerce")
        df["w_cloud_pixel"] = cloud_quality_weight(df["tcc"].values)
    else:
        df["tcc"] = np.nan
        df["w_cloud_pixel"] = 1.0

    # ------------------------------------------------------------
    # Pixel-level diurnal NH3 emission factor

    # Replace missing diurnal factors with the neutral value 1.0 so they do not modify emissions. Missingness remains recorded in the corresponding flag.
    # ------------------------------------------------------------
    if "hourfactor" in df.columns:
        df["hourfactor"] = pd.to_numeric(df["hourfactor"], errors="coerce")
        df["hourfactor"] = df["hourfactor"].fillna(1.0)
    else:
        df["hourfactor"] = 1.0

    # ------------------------------------------------------------
    # Pixel-level seasonal NH3 emission factor
    # Replace missing seasonal factors with the neutral value 1.0 so they do not modify emissions. Missingness remains recorded in the corresponding flag.
    # ------------------------------------------------------------
    if "seasonal_factor" in df.columns:
        df["seasonal_factor"] = pd.to_numeric(df["seasonal_factor"], errors="coerce")
        df["seasonal_factor"] = df["seasonal_factor"].fillna(1.0)
    else:
        df["seasonal_factor"] = 1.0

    # Combined retrieval quality weight
    df["w_quality_pixel"] = df["w_sza_pixel"] * df["w_tc_pixel"] * df["w_cloud_pixel"]

    # Signal-weighted pixel contribution.
    # hourfactor modifies emission strength, and SNR accumulation scales with signal^2.
    # df["w_signal_pixel"] = df["w_quality_pixel"] * df["hourfactor"]**2
    # Combined emission factor:
    # hourfactor controls within-day variability
    # seasonal_factor controls annual/seasonal variability
    df["combined_emission_factor"] = (
        df["hourfactor"] * df["seasonal_factor"]
    )

    # Signal-weighted pixel contribution.
    # Detectability accumulates with signal^2.
    df["w_signal_pixel"] = (
        df["w_quality_pixel"]
        * df["combined_emission_factor"]**2
    )

    # ------------------------------------------------------------
    # Collapse multiple pixel hits within same grid cell + pass
    # ------------------------------------------------------------
    df_pass = (
        df.groupby(["iy", "ix", "pass_id"], as_index=False)
        .agg(
            time_utc=("time_utc", "min"),

            # SZA
            sza_deg=("sza_deg", "min"),
            w_sza=("w_sza_pixel", "max"),

            # Thermal contrast
            thermal_contrast_K=("thermal_contrast_K", "median"),
            w_tc=("w_tc_pixel", "max"),

            # Cloud cover
            tcc=("tcc", "median"),
            w_cloud=("w_cloud_pixel", "max"),

            # Diurnal emission factor
            hourfactor=("hourfactor", "mean"),

            # Seasonal emission factor
            seasonal_factor=("seasonal_factor", "mean"),

            # Combined emission factor
            combined_emission_factor=("combined_emission_factor", "mean"),

            # Raw and effective samples inside pass
            N_raw_this_pass=("time_utc", "size"),
            N_eff_this_pass=("w_quality_pixel", "sum"),
            N_signal_eff_this_pass=("w_signal_pixel", "sum"),
        )
    )
    # One pass contributes one quality-weighted observation opportunity
    df_pass["w_quality"] = df_pass["w_sza"] * df_pass["w_tc"]* df_pass["w_cloud"]


    # One pass contributes one signal-weighted observation opportunity.
    # Retrieval quality comes from SZA and TC.
    # Emission strength comes from hourfactor.
    df_pass["combined_emission_factor"] = (
        df_pass["hourfactor"] * df_pass["seasonal_factor"]
    )

    df_pass["w_signal"] = (
        df_pass["w_quality"]
        * df_pass["combined_emission_factor"]**2
    )

    # ------------------------------------------------------------
    # Count observations per grid cell
    # ------------------------------------------------------------
    counts = (
        df_pass.groupby(["iy", "ix"], as_index=False)
        .agg(
            # Raw number of passes
            N_obs=("pass_id", "size"),

            # Effective pass-level observation count including SZA and TC
            N_obs_eff=("w_quality", "sum"),

            # Signal-weighted effective observation count including SZA, TC, and diurnal emissions
            N_obs_signal_eff=("w_signal", "sum"),

            # Pixel-count-per-pass diagnostics
            N_raw_per_pass_mean=("N_raw_this_pass", "mean"),
            N_raw_per_pass_median=("N_raw_this_pass", "median"),

            # Effective pixel-count-per-pass diagnostics
            N_eff_per_pass_mean=("N_eff_this_pass", "mean"),
            N_eff_per_pass_median=("N_eff_this_pass", "median"),

            # Signal-weighted pixel-count-per-pass diagnostics
            N_signal_eff_per_pass_mean=("N_signal_eff_this_pass", "mean"),
            N_signal_eff_per_pass_median=("N_signal_eff_this_pass", "median"),

            # SZA diagnostics
            sza_min_deg=("sza_deg", "min"),
            sza_median_deg=("sza_deg", "median"),
            w_sza_mean=("w_sza", "mean"),

            # TC diagnostics
            tc_median_K=("thermal_contrast_K", "median"),
            w_tc_mean=("w_tc", "mean"),

            # Cloud diagnostics
            tcc_median=("tcc", "median"),
            tcc_mean=("tcc", "mean"),
            w_cloud_mean=("w_cloud", "mean"),

            # Diurnal diagnostics
            hourfactor_mean=("hourfactor", "mean"),
            hourfactor_median=("hourfactor", "median"),
            hourfactor_min=("hourfactor", "min"),
            hourfactor_max=("hourfactor", "max"),

            # Seasonal diagnostics
            seasonal_factor_mean=("seasonal_factor", "mean"),
            seasonal_factor_median=("seasonal_factor", "median"),
            seasonal_factor_min=("seasonal_factor", "min"),
            seasonal_factor_max=("seasonal_factor", "max"),

            # Combined emission-factor diagnostics
            combined_emission_factor_mean=("combined_emission_factor", "mean"),
            combined_emission_factor_median=("combined_emission_factor", "median"),
            combined_emission_factor_min=("combined_emission_factor", "min"),
            combined_emission_factor_max=("combined_emission_factor", "max"),

            # Combined quality diagnostics
            w_quality_mean=("w_quality", "mean"),

            # Signal-weighted diagnostics
            w_signal_mean=("w_signal", "mean"),
        )
    )

    # ------------------------------------------------------------
    # Revisit statistics
    # ------------------------------------------------------------
    def revisit_stats(group):
        times = group.sort_values("time_utc")["time_utc"]

        if len(times) < 2:
            return pd.Series({
                "dt_median_s": np.nan,
                "dt_min_s": np.nan,
                "dt_max_s": np.nan
            })

        dt = times.diff().dt.total_seconds()

        return pd.Series({
            "dt_median_s": dt.median(),
            "dt_min_s": dt.min(),
            "dt_max_s": dt.max()
        })

    revisit = (
        df_pass.groupby(["iy", "ix"], group_keys=False)
        .apply(revisit_stats)
        .reset_index(drop=False)
    )

    sampling_summary = pd.merge(counts, revisit, on=["iy", "ix"], how="left")


    return sampling_summary


def make_local_time_diagnostics(polys_nl: pd.DataFrame, plot_dir: Path, orbit_name: str):
    """
    Create local solar time diagnostics for an orbit case.

    Produces:
    1. Pixel-weighted local solar time histogram
    2. Scanline-weighted local solar time histogram
    3. CSV file containing histogram values
    4. Summary metrics returned as a dictionary

    Pixel-weighted histogram:
    Every NL footprint pixel contributes one sample.

    Scanline-weighted histogram:
    Each scanline crossing the NL grid contributes once.
    This avoids overweighting scanlines with many pixels.
    """

    if polys_nl.empty:
        print("[LOCAL TIME] polys_nl is empty. Skipping local time diagnostics.")
        return polys_nl, {}

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = polys_nl.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)

    # Add decimal UTC hour for debugging
    df["utc_hour_decimal"] = (
        df["time_utc"].dt.hour
        + df["time_utc"].dt.minute / 60.0
        + df["time_utc"].dt.second / 3600.0
    )

    # Add true local solar hour at pixel center
    df["local_solar_hour"] = local_solar_hour_decimal(
        df["time_utc"],
        df["lon_c"].to_numpy(dtype=float)
    )

    # ------------------------------------------------------------
    # Pixel-weighted histogram
    # ------------------------------------------------------------
    bins = np.arange(0.0, 24.0 + 0.5, 0.5)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    pixel_counts, _ = np.histogram(df["local_solar_hour"].values, bins=bins)

    if pixel_counts.sum() > 0:
        pixel_fraction = pixel_counts / pixel_counts.sum()
    else:
        pixel_fraction = pixel_counts.astype(float)

    # ------------------------------------------------------------
    # Scanline-weighted histogram
    # ------------------------------------------------------------
    scanline_df = (
        df.groupby(["pass_id", "i_scan"], as_index=False)
        .agg(
            time_utc=("time_utc", "first"),
            local_solar_hour=("local_solar_hour", "median"),
            lon_c=("lon_c", "median"),
            lat_c=("lat_c", "median"),
            n_pixels=("time_utc", "size"),
        )
    )

    scan_counts, _ = np.histogram(scanline_df["local_solar_hour"].values, bins=bins)

    if scan_counts.sum() > 0:
        scan_fraction = scan_counts / scan_counts.sum()
    else:
        scan_fraction = scan_counts.astype(float)

    # ------------------------------------------------------------
    # Save histogram values
    # ------------------------------------------------------------
    hist_df = pd.DataFrame({
        "local_solar_hour_bin_center": bin_centers,
        "pixel_count": pixel_counts,
        "pixel_fraction": pixel_fraction,
        "scanline_count": scan_counts,
        "scanline_fraction": scan_fraction,
    })

    hist_csv = plot_dir / f"{orbit_name}_local_solar_time_histogram.csv"
    hist_df.to_csv(hist_csv, index=False)

    # ------------------------------------------------------------
    # Plot histogram
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        bin_centers,
        pixel_fraction,
        width=0.45,
        alpha=0.70,
        label="Pixel-weighted"
    )

    ax.plot(
        bin_centers,
        scan_fraction,
        marker="o",
        linewidth=2,
        label="Scanline-weighted"
    )

    ax.set_xlabel("Local solar time [hour]")
    ax.set_ylabel("Fraction of observations")
    ax.set_title(f"Local solar time distribution: {orbit_name}")
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_png = plot_dir / f"{orbit_name}_local_solar_time_histogram.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------
    # Helper functions for circular statistics
    # ------------------------------------------------------------
    def circular_mean_hour(hours):
        hours = np.asarray(hours, dtype=float)
        hours = hours[np.isfinite(hours)]

        if len(hours) == 0:
            return np.nan

        ang = 2.0 * np.pi * hours / 24.0
        mean_sin = np.mean(np.sin(ang))
        mean_cos = np.mean(np.cos(ang))

        mean_ang = np.arctan2(mean_sin, mean_cos)

        if mean_ang < 0:
            mean_ang += 2.0 * np.pi

        return 24.0 * mean_ang / (2.0 * np.pi)

    def circular_concentration(hours):
        """
        R value in [0, 1].
        R close to 1 means local times are tightly clustered.
        R close to 0 means local times are spread across the day.
        """

        hours = np.asarray(hours, dtype=float)
        hours = hours[np.isfinite(hours)]

        if len(hours) == 0:
            return np.nan

        ang = 2.0 * np.pi * hours / 24.0
        mean_sin = np.mean(np.sin(ang))
        mean_cos = np.mean(np.cos(ang))

        return float(np.sqrt(mean_sin**2 + mean_cos**2))

    def fraction_between(hours, h0, h1):
        """
        Fraction of hours in interval [h0, h1).
        Handles normal non-wrapping intervals.
        """

        hours = np.asarray(hours, dtype=float)
        hours = hours[np.isfinite(hours)]

        if len(hours) == 0:
            return np.nan

        return float(((hours >= h0) & (hours < h1)).mean())

    # ------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------
    pixel_hours = df["local_solar_hour"].values
    scan_hours = scanline_df["local_solar_hour"].values

    pixel_peak_hour = float(bin_centers[np.argmax(pixel_counts)]) if pixel_counts.sum() > 0 else np.nan
    scan_peak_hour = float(bin_centers[np.argmax(scan_counts)]) if scan_counts.sum() > 0 else np.nan

    metrics = {
        "local_time_pixel_peak_hour": pixel_peak_hour,
        "local_time_scanline_peak_hour": scan_peak_hour,

        "local_time_pixel_circular_mean_hour": circular_mean_hour(pixel_hours),
        "local_time_scanline_circular_mean_hour": circular_mean_hour(scan_hours),

        "local_time_pixel_concentration": circular_concentration(pixel_hours),
        "local_time_scanline_concentration": circular_concentration(scan_hours),

        "frac_pixel_local_06_10": fraction_between(pixel_hours, 6.0, 10.0),
        "frac_pixel_local_10_14": fraction_between(pixel_hours, 10.0, 14.0),
        "frac_pixel_local_14_18": fraction_between(pixel_hours, 14.0, 18.0),

        "frac_scanline_local_06_10": fraction_between(scan_hours, 6.0, 10.0),
        "frac_scanline_local_10_14": fraction_between(scan_hours, 10.0, 14.0),
        "frac_scanline_local_14_18": fraction_between(scan_hours, 14.0, 18.0),
    }

    print("\n========== LOCAL SOLAR TIME DIAGNOSTICS ==========")
    print("Orbit:", orbit_name)
    print("Saved histogram plot:", out_png)
    print("Saved histogram CSV:", hist_csv)
    print("Pixel-weighted peak local solar hour:", metrics["local_time_pixel_peak_hour"])
    print("Scanline-weighted peak local solar hour:", metrics["local_time_scanline_peak_hour"])
    print("Pixel circular mean local solar hour:", metrics["local_time_pixel_circular_mean_hour"])
    print("Scanline circular mean local solar hour:", metrics["local_time_scanline_circular_mean_hour"])
    print("Pixel local-time concentration:", metrics["local_time_pixel_concentration"])
    print("Scanline local-time concentration:", metrics["local_time_scanline_concentration"])
    print("Fraction pixel observations 06-10:", metrics["frac_pixel_local_06_10"])
    print("Fraction pixel observations 10-14:", metrics["frac_pixel_local_10_14"])
    print("Fraction pixel observations 14-18:", metrics["frac_pixel_local_14_18"])

    return df, metrics


def make_hourfactor_by_local_time_diagnostic(
    polys_nl: pd.DataFrame,
    plot_dir: Path,
    orbit_name: str
):
    """
    Diagnose whether an orbit observes during high or low NH3 emission hours.

    Requires:
    - local_solar_hour
    - hourfactor
    """

    if polys_nl.empty:
        print("[HOURFACTOR TIME] polys_nl is empty. Skipping.")
        return {}

    if "local_solar_hour" not in polys_nl.columns:
        print("[HOURFACTOR TIME] Missing local_solar_hour. Skipping.")
        return {}

    if "hourfactor" not in polys_nl.columns:
        print("[HOURFACTOR TIME] Missing hourfactor. Skipping.")
        return {}

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = polys_nl.copy()

    df["hour_bin"] = np.floor(df["local_solar_hour"]).astype(int)

    hourly = (
        df.groupby("hour_bin", as_index=False)
        .agg(
            mean_hourfactor=("hourfactor", "mean"),
            median_hourfactor=("hourfactor", "median"),
            n_pixels=("hourfactor", "size")
        )
    )

    out_csv = plot_dir / f"{orbit_name}_hourfactor_by_local_solar_hour.csv"
    hourly.to_csv(out_csv, index=False)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(
        hourly["hour_bin"],
        hourly["n_pixels"],
        color="lightgray",
        edgecolor="black",
        alpha=0.7,
        label="Observation count"
    )

    ax1.set_xlabel("Local solar time [hour]")
    ax1.set_ylabel("Observation count")
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_xlim(-0.5, 23.5)

    ax2 = ax1.twinx()

    ax2.plot(
        hourly["hour_bin"],
        hourly["mean_hourfactor"],
        marker="o",
        color="forestgreen",
        linewidth=2,
        label="Mean hourfactor"
    )

    ax2.set_ylabel("Mean hourfactor")

    ax1.set_title(f"Observed hourfactor by local solar time: {orbit_name}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()

    out_png = plot_dir / f"{orbit_name}_hourfactor_by_local_solar_hour.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    metrics = {
        "mean_observed_hourfactor": float(df["hourfactor"].mean()),
        "median_observed_hourfactor": float(df["hourfactor"].median()),
        "max_hourly_mean_hourfactor": float(hourly["mean_hourfactor"].max()),
        "hour_of_max_mean_hourfactor": int(
            hourly.loc[hourly["mean_hourfactor"].idxmax(), "hour_bin"]
        ),
    }

    print("\n========== HOURFACTOR BY LOCAL TIME DIAGNOSTICS ==========")
    print("Orbit:", orbit_name)
    print("Saved plot:", out_png)
    print("Saved CSV:", out_csv)
    print("Mean observed hourfactor:", metrics["mean_observed_hourfactor"])
    print("Median observed hourfactor:", metrics["median_observed_hourfactor"])
    print("Hour of max mean hourfactor:", metrics["hour_of_max_mean_hourfactor"])

    return metrics

# Conservatively remap CAMS source-cell emissions to the LOTOS-EUROS grid
# using spherical overlap-area fractions.

def load_emissions_to_le_grid(emis_path, countries_to_keep=None, unit_scale_to_kgyr=1e6):
    """
    Load CAMS source-level NH3 emissions and conservatively remap them to the LE grid
    using exact overlap-area weighting between CAMS cells and LE cells.

    Returns
    -------
    emis_cell : pd.DataFrame
        Columns:
        iy, ix, nh3_raw, cell_area_km2, nh3_kg_yr, nh3_kg_km2_yr, nh3_kg_m2_yr
    """
    nc = netCDF4.Dataset(emis_path)

    # Load source-level CAMS data
    lon_s = nc["longitude_source"][:]
    lat_s = nc["latitude_source"][:]
    nh3 = nc["nh3"][:]   # Gg/year

    lon_idx = nc["longitude_index"][:]
    lat_idx = nc["latitude_index"][:]

    lon_bounds = nc["longitude_bounds"][:]   # shape (nlon, 2)
    lat_bounds = nc["latitude_bounds"][:]    # shape (nlat, 2)

    print("\n========== RAW CAMS DATA ==========")
    print("NH3 units:", getattr(nc["nh3"], "units", "NO_UNITS_ATTRIBUTE"))
    nh3_series = pd.Series(nh3)
    print("Total NH3 in file [Gg/yr]:", nh3_series.sum())
    print(nh3_series.describe(percentiles=[0.5, 0.9, 0.99]))

    df = pd.DataFrame({
        "lon": lon_s,
        "lat": lat_s,
        "nh3": nh3,
        "lon_idx": lon_idx,
        "lat_idx": lat_idx,
    })

    # Optional country filtering
    if (
        countries_to_keep is not None
        and "country_index" in nc.variables
        and "country_name" in nc.variables
    ):
        cid = nc["country_index"][:]
        country_names = [
            str(uu, encoding="UTF-8").rstrip("\x00").strip()
            for uu in nc.variables["country_name"][:]
        ]
        keep_ids = {country_names.index(name) + 1 for name in countries_to_keep}

        df["country_index"] = cid
        df = df[df["country_index"].isin(keep_ids)].copy()

        print("\n========== COUNTRY FILTER CHECK ==========")
        print("Remaining rows after filtering:", len(df))
        print("Total NH3 after country filtering [Gg/yr]:", df["nh3"].sum())

    # Aggregate source records -> native CAMS cells
    cams_agg = (
        df.groupby(["lat_idx", "lon_idx"], as_index=False)["nh3"]
        .sum()
    )

    print("\n========== CAMS GRID AGGREGATION ==========")
    print("Number of native CAMS cells after aggregation:", len(cams_agg))
    print("Total NH3 in CAMS cells [Gg/yr]:", cams_agg["nh3"].sum())

    # Auto-detect whether CAMS indices are 0-based or 1-based
    lon_offset = 1 if np.min(lon_idx) == 1 else 0
    lat_offset = 1 if np.min(lat_idx) == 1 else 0

    # Precompute LE cell edges
    le_lon_edges = GRID_WEST + np.arange(GRID_NX + 1) * GRID_DLON
    le_lat_edges = GRID_SOUTH + np.arange(GRID_NY + 1) * GRID_DLAT

    le_records = []
    contrib_records = []     # To

    for _, row in cams_agg.iterrows():
        ci = int(row["lon_idx"]) - lon_offset
        cj = int(row["lat_idx"]) - lat_offset

        if ci < 0 or ci >= lon_bounds.shape[0] or cj < 0 or cj >= lat_bounds.shape[0]:
            continue

        # CAMS cell bounds
        lon_lo, lon_hi = lon_bounds[ci, 0], lon_bounds[ci, 1]
        lat_lo, lat_hi = lat_bounds[cj, 0], lat_bounds[cj, 1]

        cams_area = spherical_rect_area_km2(lon_lo, lon_hi, lat_lo, lat_hi)
        if cams_area <= 0:
            continue

        # Candidate LE index range
        ix_lo = int(np.floor((lon_lo - GRID_WEST) / GRID_DLON))
        ix_hi = int(np.floor((lon_hi - GRID_WEST) / GRID_DLON - 1e-12))
        iy_lo = int(np.floor((lat_lo - GRID_SOUTH) / GRID_DLAT))
        iy_hi = int(np.floor((lat_hi - GRID_SOUTH) / GRID_DLAT - 1e-12))

        ix_lo = max(ix_lo, 0)
        ix_hi = min(ix_hi, GRID_NX - 1)
        iy_lo = max(iy_lo, 0)
        iy_hi = min(iy_hi, GRID_NY - 1)

        if ix_lo > ix_hi or iy_lo > iy_hi:
            continue



        # Conservative overlap redistribution within the LE analysis domain.
        # An LE cell receives the fraction of each CAMS-cell total corresponding to
        # their spherical overlap area. Source-cell emissions outside the LE domain
        # are intentionally excluded.

        for iy in range(iy_lo, iy_hi + 1):
            le_lat_lo = le_lat_edges[iy]
            le_lat_hi = le_lat_edges[iy + 1]

            ov_lat_lo = max(lat_lo, le_lat_lo)
            ov_lat_hi = min(lat_hi, le_lat_hi)
            if ov_lat_lo >= ov_lat_hi:
                continue

            for ix in range(ix_lo, ix_hi + 1):
                le_lon_lo = le_lon_edges[ix]
                le_lon_hi = le_lon_edges[ix + 1]

                ov_lon_lo = max(lon_lo, le_lon_lo)
                ov_lon_hi = min(lon_hi, le_lon_hi)
                if ov_lon_lo >= ov_lon_hi:
                    continue

                overlap_area = spherical_rect_area_km2(
                    ov_lon_lo, ov_lon_hi, ov_lat_lo, ov_lat_hi
                )

                if overlap_area <= 0:
                    continue

                weight = overlap_area / cams_area
                nh3_to_le = row["nh3"] * weight
                le_records.append((iy, ix, nh3_to_le))
                contrib_records.append((iy, ix, 1))

    emis_cell = pd.DataFrame(le_records, columns=["iy", "ix", "nh3_raw"])

    if emis_cell.empty:
        nc.close()
        raise ValueError("No CAMS emissions overlap with LE grid after conservative redistribution.")

    emis_cell = (
        emis_cell.groupby(["iy", "ix"], as_index=False)["nh3_raw"]
        .sum()
    )

    contrib_cell = pd.DataFrame(contrib_records, columns=["iy", "ix", "n_contrib"])
    contrib_cell = (
        contrib_cell.groupby(["iy", "ix"], as_index=False)["n_contrib"]
        .sum()
    )

    emis_cell = pd.merge(emis_cell, contrib_cell, on=["iy", "ix"], how="left")

    print("\n========== CONSERVATIVE CAMS -> LE AGGREGATION ==========")
    print("Number of LE cells after conservative CAMS->LE mapping:", len(emis_cell))
    print("Total NH3 after conservative CAMS->LE aggregation [Gg/yr]:", emis_cell["nh3_raw"].sum())

    emis_cell["cell_area_km2"] = le_cell_area_km2(emis_cell["iy"].values)
    emis_cell["nh3_kg_yr"] = emis_cell["nh3_raw"] * unit_scale_to_kgyr
    emis_cell["nh3_kg_km2_yr"] = emis_cell["nh3_kg_yr"] / emis_cell["cell_area_km2"]
    emis_cell["nh3_kg_m2_yr"] = emis_cell["nh3_kg_yr"] / (emis_cell["cell_area_km2"] * 1e6)

    print("\n========== AGGREGATION CHECK ==========")
    print("Total raw NH3 over selected LE cells [Gg/yr]:", emis_cell["nh3_raw"].sum())
    print("Total NH3 over selected LE cells [kg/yr]:", emis_cell["nh3_kg_yr"].sum())
    print("Converted back to Gg/yr from kg:", emis_cell["nh3_kg_yr"].sum() / 1e6)

    print("\n========== CAMS CONTRIBUTIONS PER LE CELL ==========")
    print(emis_cell["n_contrib"].describe(percentiles=[0.5, 0.9, 0.99]))
    print("\nExample of high-contribution LE cells:")
    print(emis_cell.sort_values("n_contrib", ascending=False).head(10))


    nc.close()
    return emis_cell


# -----------------------------
# 5) Orbit-aware observability computation
# -----------------------------
def compute_observability(
    sampling_summary: pd.DataFrame,
    emis_cell: pd.DataFrame,
    sigma_I_molec_cm2=5e15,
    sigma_X_molec_cm2=0.75e15,
    F_clear=0.33,    #Using F_clear = 1 in the production and ablation calls pass as cloud loss is already taken into consideration by w_cloud.
    pressure_hpa=1000.0,
    wind_kmh=18.0,     # 5 m/s
    W_km=10.0,
    window_days=30.0
):
    """
    Combine orbit-derived sampling with emissions-derived signal estimates.

    Inputs
    ------
    sampling_summary:
        DataFrame with columns:
            iy, ix, N_obs, dt_median_s, dt_min_s, dt_max_s

    emis_cell:
        DataFrame with columns:
            iy, ix, nh3_kg_yr

    Outputs
    -------
    DataFrame with:
        iy, ix,
        nh3_kg_yr,
        cell_area_km2,
        E_flux_kg_km2_h,
        enhancement_ppb,
        sigma_ppb,
        N_obs,
        dt_median_days,
        time_to_detect_days,
        time_to_detect_years
    """
    s = sampling_summary.copy()
    e = emis_cell.copy()

    # Join by grid cell
    df = pd.merge(e, s, on=["iy", "ix"], how="inner")
    if df.empty:
        raise ValueError("No overlap between sampled cells and emission cells.")

    # Cell area [km²]
    # df["cell_area_km2"] = [
    #     le_cell_area_km2(int(iy), int(ix))
    #     for iy, ix in zip(df["iy"].values, df["ix"].values)
    # ]
    if "cell_area_km2" not in df.columns:
        df["cell_area_km2"] = le_cell_area_km2(df["iy"].values)


    df["nh3_kg_km2_yr"] = df["nh3_kg_yr"] / df["cell_area_km2"]

   # emis_cell["nh3_kg_km2_yr"] = emis_cell["nh3_kg_yr"] / emis_cell["cell_area_km2"]


    # Convert annual emissions to flux [kg km^-2 h^-1]
    df["E_flux_kg_km2_h"] = df["nh3_kg_yr"] / (df["cell_area_km2"] * 365.0 * 24.0)

    # Compute signal (column enhancement)
    alpha = calc_alpha(pressure_hpa=pressure_hpa, wind_kmh=wind_kmh)
    df["enhancement_ppb"] = calc_enhancement_ppb(alpha, df["E_flux_kg_km2_h"].values, W_km)

    # Noise
    sigma_ppb = calc_sigma_ppb(
        sigma_I_molec_cm2=sigma_I_molec_cm2,
        sigma_X_molec_cm2=sigma_X_molec_cm2,
        pressure_hpa=pressure_hpa
    )
    df["sigma_ppb"] = sigma_ppb

    # Orbit-aware detectability
    # Required sample-equivalent count for a target SNR of 2.
    # This assumes independent observations with uncorrelated uncertainty, so
    # accumulated SNR increases with sqrt(M). Temporal or spatial error
    # correlations are not modelled explicitly.

  
    sig = np.maximum(df["enhancement_ppb"].values, 1e-30)
    M_req = np.maximum(1.0, (2.0 * sigma_ppb / sig)**2)


    # Successful obs/day from orbit
    # Prefer signal-weighted effective observations when diurnal emissions are available.
    if "N_obs_signal_eff" in df.columns:
        obs_count_for_detect = df["N_obs_signal_eff"].values
        obs_count_source = "N_obs_signal_eff"
    elif "N_obs_eff" in df.columns:
        obs_count_for_detect = df["N_obs_eff"].values
        obs_count_source = "N_obs_eff"
    else:
        obs_count_for_detect = df["N_obs"].values
        obs_count_source = "N_obs"

    obs_rate_per_day = (F_clear * obs_count_for_detect) / max(window_days, 1e-6)


    # time_days = np.where(obs_rate_per_day > 0, M_req / obs_rate_per_day, np.inf)

    # df["time_to_detect_days"] = time_days
    # df["time_to_detect_years"] = time_days / 365.0
    # df["dt_median_days"] = df["dt_median_s"] / 86400.0

    # return df
    time_days = np.where(obs_rate_per_day > 0, M_req / obs_rate_per_day, np.inf)

    # Current method diagnostics
    df["M_req_obs"] = M_req
    df["obs_rate_per_day_current"] = obs_rate_per_day
    df["obs_count_for_detect"] = obs_count_for_detect
    df["obs_count_source"] = obs_count_source
    df["time_to_detect_days"] = time_days
    df["time_to_detect_years"] = time_days / 365.0
    df["dt_median_days"] = df["dt_median_s"] / 86400.0


    # ------------------------------------------------------------
    # Paper-style detectability using pipeline-derived observations
    # per individual pass
    # ------------------------------------------------------------
    # Paper structure:
    # t = t_R * max[1, M_req / (F * N)]
    #
    # Here:
    # t_R = median revisit time of that LE cell
    # N   = number of observations obtained during one pass,
    #       computed from the pipeline, NOT from (W/D)^2.
    #
    # We calculate two versions:
    # 1) raw-paper method:
    #    N = N_raw_per_pass_mean
    #
    # 2) effective-paper method:
    #    N = N_eff_per_pass_mean
    #    This includes SZA weighting if SZA is turned on.
    # ------------------------------------------------------------

    '''
    Now, time_to_detect_days = How long to detect based on total effective observations accumulated per day?
    time_to_detect_days_paper = Using paper style formula without SZA(raw) and with SZA (effective)'''

    '''
    
     The signal-effective paper-style formulation is used for final orbit ranking.
     The observation-rate, raw-N, and effective-N formulations are retained as
     diagnostic and ablation outputs.
     '''
# Final paper-style metric:
# Detection time is estimated as the cell's median revisit interval multiplied
# by the number of revisit opportunities required to accumulate sufficient
# signal-effective observations. A minimum of one revisit interval is imposed.


    t_R_days = df["dt_median_days"].values

    N_raw_pass = (
        df["N_raw_per_pass_mean"].values
        if "N_raw_per_pass_mean" in df.columns
        else np.ones(len(df))
    )

    N_eff_pass = (
        df["N_eff_per_pass_mean"].values
        if "N_eff_per_pass_mean" in df.columns
        else N_raw_pass
    )

    N_signal_eff_pass = (
        df["N_signal_eff_per_pass_mean"].values
        if "N_signal_eff_per_pass_mean" in df.columns
        else N_eff_pass
    )


    denom_paper_raw = F_clear * N_raw_pass
    denom_paper_eff = F_clear * N_eff_pass
    denom_paper_signal_eff = F_clear * N_signal_eff_pass


    time_days_paper_raw = np.where(
        (denom_paper_raw > 0) & np.isfinite(t_R_days),
        t_R_days * np.maximum(1.0, M_req / denom_paper_raw),
        np.inf
    )

    time_days_paper_eff = np.where(
        (denom_paper_eff > 0) & np.isfinite(t_R_days),
        t_R_days * np.maximum(1.0, M_req / denom_paper_eff),
        np.inf
    )

    time_days_paper_signal_eff = np.where(
        (denom_paper_signal_eff > 0) & np.isfinite(t_R_days),
        t_R_days * np.maximum(1.0, M_req / denom_paper_signal_eff),
        np.inf
    )

    df["time_to_detect_days_paper_rawN"] = time_days_paper_raw
    df["time_to_detect_years_paper_rawN"] = time_days_paper_raw / 365.0

    df["time_to_detect_days_paper_effN"] = time_days_paper_eff
    df["time_to_detect_years_paper_effN"] = time_days_paper_eff / 365.0

    df["time_to_detect_days_paper_signalEffN"] = time_days_paper_signal_eff
    df["time_to_detect_years_paper_signalEffN"] = time_days_paper_signal_eff / 365.0

    return df


# -----------------------------
# 6) Convenience wrapper: run everything directly from polys_nl
# -----------------------------
def run_observability_from_polys(
    polys_nl: pd.DataFrame,
    emis_path: str,
    countries_to_keep=None,
    unit_scale_to_kgyr=1e6,
    sigma_I_molec_cm2=5e15,
    sigma_X_molec_cm2=0.75e15,
    F_clear=0.33,
    pressure_hpa=1000.0,
    wind_kmh=18.0,
    W_km=10.0, #Assumed characteristic downwind plume/source length scale used in the column-enhancement approximation.
    window_days = None
):
    """
    One-stop function for use in the orbit pipeline.

    Inputs:
        polys_nl  -> directly from your orbit pipeline
        emis_path -> CAMS NH3 file

    Returns:
        observability_df, sampling_summary, emis_cell
    """
    if polys_nl.empty:
        raise ValueError("polys_nl is empty; cannot compute observability.")

    # Build sampling summary from orbit-generated polygons
    sampling_summary = build_sampling_summary_by_pass(polys_nl)

    df_pass_debug = (
    polys_nl
    .groupby(["iy", "ix", "pass_id"], as_index=False)
    .agg(time_utc=("time_utc", "min"))
    )


    if window_days is None:
        t0 = pd.to_datetime(polys_nl["time_utc"], utc=True).min()
        t1 = pd.to_datetime(polys_nl["time_utc"], utc=True).max()
        window_days = max((t1 - t0).total_seconds() / 86400.0, 1e-6)


    # Load and grid emissions
    emis_cell = load_emissions_to_le_grid(
        emis_path=emis_path,
        countries_to_keep=countries_to_keep,
        unit_scale_to_kgyr=unit_scale_to_kgyr
    )

    # Compute observability
    observability_df = compute_observability(
        sampling_summary=sampling_summary,
        emis_cell=emis_cell,
        sigma_I_molec_cm2=sigma_I_molec_cm2,
        sigma_X_molec_cm2=sigma_X_molec_cm2,
        F_clear=F_clear,
        pressure_hpa=pressure_hpa,
        wind_kmh=wind_kmh,
        W_km=W_km,
        window_days=window_days
    )

    return observability_df, sampling_summary, emis_cell




# ============================================================
# RESULTS PLOTTING SETTINGS
# ============================================================

GENERATE_RESULTS_PLOTS = True
GENERATE_SINGLE_ORBIT_FIGURES = False
GENERATE_MULTI_ORBIT_RESULTS = False
SAVE_RESULTS_PNG = True
SAVE_RESULTS_PDF = False

RESULTS_DPI = 300

# Maximum number of pixel rows plotted in distribution plots.
# The calculations still use the full dataset.
MAX_PLOT_PIXELS = 250_000

# Representative pass used for footprint visualisation.
# None means automatically select the pass containing most NL pixels.
REPRESENTATIVE_PASS_ID = None

# SZA threshold used in the current model.
SZA_THRESHOLD_DEG = 100.0

# Detectability thresholds reported in comparison figures.
DETECTION_THRESHOLDS_DAYS = [30.0, 90.0, 365.0]

# Display cap only. Values above this are shown using the uppermost colour.
DETECTABILITY_DISPLAY_CAP_DAYS = 3650.0

# Small positive value for logarithmic plots.
PLOT_EPS = 1e-12

# Compact information retained after every orbit finishes.
# This avoids keeping every full pixel file in memory.
ORBIT_RESULTS_CACHE = {}

# Consistent column names and labels.
DETECTABILITY_COLUMNS = {
    "Raw geometry": "time_to_detect_days_paper_rawN",
    "Retrieval quality": "time_to_detect_days_paper_effN",
    "Full model": "time_to_detect_days_paper_signalEffN",
}


# ============================================================
# RESULTS-PLOTTING CONFIGURATION
# ============================================================
# Controls the newer standardized single-orbit, multi-orbit,
# diagnostic, and ablation figures defined below.
#
# These settings do not control every legacy plot created inside
# run_orbit_case(); some older plots are generated separately.




def save_results_figure(fig, output_base):
    """
    Save a figure as PNG and/or PDF and close it.

    output_base should not contain a file extension.
    """
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if SAVE_RESULTS_PNG:
        fig.savefig(
            output_base.with_suffix(".png"),
            dpi=RESULTS_DPI,
            bbox_inches="tight"
        )

    if SAVE_RESULTS_PDF:
        fig.savefig(
            output_base.with_suffix(".pdf"),
            bbox_inches="tight"
        )

    plt.close(fig)


def safe_numeric_array(values, positive_only=False):
    """
    Convert input to a one-dimensional finite float array.
    """
    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(dtype=float)

    arr = arr[np.isfinite(arr)]

    if positive_only:
        arr = arr[arr > 0]

    return arr


def sample_dataframe_for_plotting(df, max_rows=MAX_PLOT_PIXELS):
    """
    Reduce plotting load without changing any calculations.
    """
    if len(df) <= max_rows:
        return df.copy()

    return df.sample(
        n=max_rows,
        random_state=42
    ).copy()


def dataframe_to_le_grid(
    df,
    value_col,
    aggregation="median",
    fill_value=np.nan
):
    """
    Convert a cell-level DataFrame with iy and ix to a full LE grid.

    aggregation:
        median, mean, sum, max, min, first
    """
    if df.empty:
        return np.full((GRID_NY, GRID_NX), fill_value, dtype=float)

    required = {"iy", "ix", value_col}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns for grid conversion: {sorted(missing)}"
        )

    work = df[["iy", "ix", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["iy", "ix", value_col])

    if work.empty:
        return np.full((GRID_NY, GRID_NX), fill_value, dtype=float)

    grouped = (
        work.groupby(["iy", "ix"], as_index=False)[value_col]
        .agg(aggregation)
    )

    grid = np.full((GRID_NY, GRID_NX), fill_value, dtype=float)

    iy = grouped["iy"].to_numpy(dtype=int)
    ix = grouped["ix"].to_numpy(dtype=int)
    values = grouped[value_col].to_numpy(dtype=float)

    valid = (
        (iy >= 0) & (iy < GRID_NY) &
        (ix >= 0) & (ix < GRID_NX)
    )

    grid[iy[valid], ix[valid]] = values[valid]

    return grid


def le_grid_edges():
    """
    Return longitude and latitude edges of the LOTOS-EUROS grid.
    """
    lon_edges = GRID_WEST + np.arange(GRID_NX + 1) * GRID_DLON
    lat_edges = GRID_SOUTH + np.arange(GRID_NY + 1) * GRID_DLAT

    return lon_edges, lat_edges





def plot_le_grid_map(
    df,
    value_col,
    title,
    colourbar_label,
    output_base,
    aggregation="median",
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None
):
    """
    Plot one cell-level variable on the LOTOS-EUROS grid.
    """
    grid = dataframe_to_le_grid(
        df=df,
        value_col=value_col,
        aggregation=aggregation
    )

    lon_edges, lat_edges = le_grid_edges()

    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        np.ma.masked_invalid(grid),
        shading="auto",
        cmap=cmap,
        norm=norm,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(colourbar_label)

    ax.set_xlim(GRID_WEST, GRID_EAST)
    ax.set_ylim(GRID_SOUTH, GRID_NORTH)

    ax.set_xlabel("Longitude [degrees east]")
    ax.set_ylabel("Latitude [degrees north]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    save_results_figure(fig, output_base)


def add_local_solar_time(df):
    """
    Add approximate local solar time using:

        local solar time = UTC hour + longitude / 15

    This is suitable for orbit-sampling diagnostics.
    """
    out = df.copy()
    out["time_utc"] = pd.to_datetime(
        out["time_utc"],
        utc=True,
        errors="coerce"
    )

    utc_hours = (
        out["time_utc"].dt.hour
        + out["time_utc"].dt.minute / 60.0
        + out["time_utc"].dt.second / 3600.0
    )

    out["local_solar_time_h"] = (
        utc_hours + out["lon_c"] / 15.0
    ) % 24.0

    return out


def build_pass_diagnostics(polys_nl):
    """
    Create one row per regional pass for local-time and pass-level plots.
    """
    df = add_local_solar_time(polys_nl)

    pass_diag = (
        df.groupby("pass_id", as_index=False)
        .agg(
            pass_time_utc=("time_utc", "min"),
            local_solar_time_h=("local_solar_time_h", "median"),
            median_latitude=("lat_c", "median"),
            median_longitude=("lon_c", "median"),
            n_pixels=("time_utc", "size"),
            sza_median_deg=("sza_deg", "median"),
            tc_median_K=("thermal_contrast_K", "median"),
            tcc_median=("tcc", "median"),
            hourfactor_median=("hourfactor", "median"),
            seasonal_factor_median=("seasonal_factor", "median"),
        )
    )

    pass_diag["pass_date"] = (
        pd.to_datetime(pass_diag["pass_time_utc"], utc=True)
        .dt.floor("D")
    )

    return pass_diag


def plot_ground_track(
    track_coarse,
    orbit_name,
    output_base
):
    """
    Plot the propagated coarse ground track and highlight the NL study domain.
    """
    track_plot = track_coarse.copy().sort_values("time_utc")

    fig, ax = plt.subplots(figsize=(11.0, 5.8))

    # Avoid connecting the track across longitude discontinuities.
    lon = track_plot["lon"].to_numpy(dtype=float)
    lat = track_plot["lat"].to_numpy(dtype=float)

    split_indices = np.where(np.abs(np.diff(lon)) > 180.0)[0] + 1
    lon_segments = np.split(lon, split_indices)
    lat_segments = np.split(lat, split_indices)

    for lon_segment, lat_segment in zip(lon_segments, lat_segments):
        ax.plot(
            lon_segment,
            lat_segment,
            linewidth=0.35,
            alpha=0.65
        )

    domain = Rectangle(
        (GRID_WEST, GRID_SOUTH),
        GRID_EAST - GRID_WEST,
        GRID_NORTH - GRID_SOUTH,
        fill=False,
        edgecolor="red",
        linewidth=1.8,
        label="LOTOS-EUROS analysis domain"
    )

    ax.add_patch(domain)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))

    ax.set_xlabel("Longitude [degrees east]")
    ax.set_ylabel("Latitude [degrees north]")
    ax.set_title(f"Propagated ground track: {orbit_name}")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

    save_results_figure(fig, output_base)


def plot_netherlands_groundtrack(
    track_coarse,
    orbit_name,
    output_base
):
    """
    Plot the subsatellite ground track around the Netherlands domain.
    """

    track_plot = (
        track_coarse.copy()
        .sort_values("time_utc")
        .reset_index(drop=True)
    )

    # Show the Netherlands box plus a surrounding geographic margin.
    margin_lon = 5.0
    margin_lat = 4.0

    regional_track = track_plot[
        (track_plot["lon"] >= GRID_WEST - margin_lon)
        & (track_plot["lon"] <= GRID_EAST + margin_lon)
        & (track_plot["lat"] >= GRID_SOUTH - margin_lat)
        & (track_plot["lat"] <= GRID_NORTH + margin_lat)
    ].copy()

    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    if not regional_track.empty:
        # Split separate passes so unrelated points are not connected.
        time_gap_seconds = (
            regional_track["time_utc"]
            .diff()
            .dt.total_seconds()
            .fillna(0.0)
        )

        regional_track["track_section"] = (
            time_gap_seconds > 120.0
        ).cumsum()

        for _, section in regional_track.groupby("track_section"):
            ax.plot(
                section["lon"],
                section["lat"],
                color="tab:blue",
                linewidth=1.2,
                alpha=0.8
            )

            ax.scatter(
                section["lon"],
                section["lat"],
                s=7,
                color="tab:blue",
                alpha=0.7
            )

    domain = Rectangle(
        (GRID_WEST, GRID_SOUTH),
        GRID_EAST - GRID_WEST,
        GRID_NORTH - GRID_SOUTH,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
        label="Netherlands analysis domain"
    )

    ax.add_patch(domain)

    ax.set_xlim(
        GRID_WEST - margin_lon,
        GRID_EAST + margin_lon
    )
    ax.set_ylim(
        GRID_SOUTH - margin_lat,
        GRID_NORTH + margin_lat
    )

    ax.set_xlabel("Longitude [degrees east]")
    ax.set_ylabel("Latitude [degrees north]")
    ax.set_title(
        f"Ground track near the Netherlands: {orbit_name}"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    save_results_figure(fig, output_base)

def plot_local_solar_time_sampling(
    pass_diag,
    orbit_name,
    output_base
):
    """
    Plot regional pass date against approximate local solar time.
    """
    if pass_diag.empty:
        return

    fig, ax = plt.subplots(figsize=(10.2, 5.4))

    ax.scatter(
        pass_diag["pass_time_utc"],
        pass_diag["local_solar_time_h"],
        s=26,
        alpha=0.75
    )

    ax.set_ylim(0, 24)
    ax.set_yticks(np.arange(0, 25, 3))
    ax.set_ylabel("Approximate local solar time [h]")
    ax.set_xlabel("Date")
    ax.set_title(f"Regional observation-time sampling: {orbit_name}")
    ax.grid(alpha=0.25)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    save_results_figure(fig, output_base)

def select_representative_pass(polys_nl):
    """
    Select a representative pass.

    If REPRESENTATIVE_PASS_ID is None, the pass containing the
    largest number of pixels in the analysis domain is selected.
    """
    if REPRESENTATIVE_PASS_ID is not None:
        selected = polys_nl[
            polys_nl["pass_id"] == REPRESENTATIVE_PASS_ID
        ].copy()

        if not selected.empty:
            return REPRESENTATIVE_PASS_ID, selected

        warnings.warn(
            f"Requested representative pass "
            f"{REPRESENTATIVE_PASS_ID} was unavailable. "
            f"Selecting the largest pass instead."
        )

    pass_id = polys_nl.groupby("pass_id").size().idxmax()

    selected = polys_nl[
        polys_nl["pass_id"] == pass_id
    ].copy()

    return int(pass_id), selected


def plot_representative_footprints(
    polys_nl,
    orbit_name,
    output_base
):
    """
    Plot representative synthetic pixel polygons over the model grid.

    To avoid an unreadable figure, one scanline near the centre of the
    selected regional pass is displayed.
    """
    pass_id, pass_df = select_representative_pass(polys_nl)

    scanline_ids = np.sort(pass_df["i_scan"].unique())

    if len(scanline_ids) == 0:
        return

    selected_scanline = scanline_ids[len(scanline_ids) // 2]

    scan_df = pass_df[
        pass_df["i_scan"] == selected_scanline
    ].copy()

    if scan_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.4, 7.0))

    # Light model-grid lines.
    lon_edges, lat_edges = le_grid_edges()

    for lon in lon_edges[::4]:
        ax.axvline(lon, color="0.88", linewidth=0.35, zorder=0)

    for lat in lat_edges[::4]:
        ax.axhline(lat, color="0.88", linewidth=0.35, zorder=0)

    for _, row in scan_df.iterrows():
        corners = np.array([
            [row["lon_ul"], row["lat_ul"]],
            [row["lon_ur"], row["lat_ur"]],
            [row["lon_lr"], row["lat_lr"]],
            [row["lon_ll"], row["lat_ll"]],
        ])

        polygon = Polygon(
            corners,
            closed=True,
            facecolor="tab:blue",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.35
        )

        ax.add_patch(polygon)

    ax.scatter(
        scan_df["lon_c"],
        scan_df["lat_c"],
        s=8,
        color="black",
        label="Pixel centres",
        zorder=3
    )

    xpad = max(0.08, 0.1 * (scan_df["lon_c"].max() - scan_df["lon_c"].min()))
    ypad = max(0.08, 0.1 * (scan_df["lat_c"].max() - scan_df["lat_c"].min()))

    ax.set_xlim(
        max(GRID_WEST, scan_df["lon_c"].min() - xpad),
        min(GRID_EAST, scan_df["lon_c"].max() + xpad)
    )

    ax.set_ylim(
        max(GRID_SOUTH, scan_df["lat_c"].min() - ypad),
        min(GRID_NORTH, scan_df["lat_c"].max() + ypad)
    )

    ax.set_xlabel("Longitude [degrees east]")
    ax.set_ylabel("Latitude [degrees north]")
    ax.set_title(
        f"Representative synthetic scanline\n"
        f"{orbit_name}, pass {pass_id}"
    )

    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(alpha=0.15)

    save_results_figure(fig, output_base)


def plot_pixel_geometry(
    polys_nl,
    orbit_name,
    output_base
):
    """
    Plot footprint dimensions as a function of off-nadir angle.
    """
    geom = (
        polys_nl[
            ["j_across", "theta_deg", "w_x_m", "w_y_m"]
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby("j_across", as_index=False)
        .median(numeric_only=True)
        .sort_values("theta_deg")
    )

    if geom.empty:
        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.2, 4.8)
    )

    axes[0].plot(
        geom["theta_deg"],
        geom["w_x_m"] / 1000.0,
        linewidth=1.8
    )

    axes[0].set_xlabel("Off-nadir angle [degrees]")
    axes[0].set_ylabel("Across-track pixel width [km]")
    axes[0].set_title("Across-track footprint growth")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        geom["theta_deg"],
        geom["w_y_m"] / 1000.0,
        linewidth=1.8
    )

    axes[1].set_xlabel("Off-nadir angle [degrees]")
    axes[1].set_ylabel("Along-track pixel length [km]")
    axes[1].set_title("Along-track footprint growth")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Synthetic pixel geometry: {orbit_name}")
    fig.tight_layout()

    save_results_figure(fig, output_base)


def plot_single_orbit_spatial_maps(
    orbit_name,
    sampling_summary,
    emis_cell,
    observability_df,
    plot_dir
):
    """
    Generate main cell-level maps for one orbit.
    """
    spatial_dir = Path(plot_dir) / "spatial_maps"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    plot_le_grid_map(
        df=sampling_summary,
        value_col="N_obs",
        title=f"Number of regional passes: {orbit_name}",
        colourbar_label="Number of passes",
        output_base=spatial_dir / "01_raw_pass_count",
        aggregation="sum",
        cmap="viridis"
    )

    revisit_df = sampling_summary.copy()
    revisit_df["dt_median_days"] = (
        revisit_df["dt_median_s"] / 86400.0
    )

    finite_revisit = safe_numeric_array(
        revisit_df["dt_median_days"],
        positive_only=True
    )

    if len(finite_revisit) > 0:
        revisit_norm = mcolors.LogNorm(
            vmin=max(np.nanpercentile(finite_revisit, 2), PLOT_EPS),
            vmax=max(np.nanpercentile(finite_revisit, 98), PLOT_EPS * 10)
        )

        plot_le_grid_map(
            df=revisit_df,
            value_col="dt_median_days",
            title=f"Median grid-cell revisit time: {orbit_name}",
            colourbar_label="Median revisit time [days]",
            output_base=spatial_dir / "02_median_revisit_days",
            aggregation="median",
            cmap="viridis",
            norm=revisit_norm
        )

    emission_values = safe_numeric_array(
        emis_cell["nh3_kg_km2_yr"],
        positive_only=True
    )

    if len(emission_values) > 0:
        emission_norm = mcolors.LogNorm(
            vmin=max(np.nanpercentile(emission_values, 2), PLOT_EPS),
            vmax=max(np.nanpercentile(emission_values, 98), PLOT_EPS * 10)
        )

        plot_le_grid_map(
            df=emis_cell,
            value_col="nh3_kg_km2_yr",
            title="CAMS-REG NH$_3$ emission density on the LOTOS-EUROS grid",
            colourbar_label=r"NH$_3$ emission density [kg km$^{-2}$ yr$^{-1}$]",
            output_base=spatial_dir / "03_nh3_emission_density",
            aggregation="sum",
            cmap="magma",
            norm=emission_norm
        )

    mreq_values = safe_numeric_array(
        observability_df["M_req_obs"],
        positive_only=True
    )

    if len(mreq_values) > 0:
        mreq_plot = observability_df.copy()
        mreq_plot["log10_M_req"] = np.log10(
            np.maximum(mreq_plot["M_req_obs"], 1.0)
        )

        plot_le_grid_map(
            df=mreq_plot,
            value_col="log10_M_req",
            title="Required number of independent observations",
            colourbar_label=r"$\log_{10}(M_{\mathrm{req}})$",
            output_base=spatial_dir / "04_required_observations",
            aggregation="median",
            cmap="plasma"
        )

    for label, colname, filename in [
        (
            "Raw detectability time",
            "time_to_detect_days_paper_rawN",
            "05_detectability_raw"
        ),
        (
            "Retrieval-quality detectability time",
            "time_to_detect_days_paper_effN",
            "06_detectability_quality"
        ),
        (
            "Full signal-effective detectability time",
            "time_to_detect_days_paper_signalEffN",
            "07_detectability_full"
        ),
    ]:
        values = safe_numeric_array(
            observability_df[colname],
            positive_only=True
        )

        if len(values) == 0:
            continue

        finite_upper = min(
            np.nanpercentile(values, 98),
            DETECTABILITY_DISPLAY_CAP_DAYS
        )

        finite_lower = max(
            np.nanpercentile(values, 2),
            PLOT_EPS
        )

        if finite_upper <= finite_lower:
            finite_upper = finite_lower * 10.0

        detect_norm = mcolors.LogNorm(
            vmin=finite_lower,
            vmax=finite_upper
        )

        plot_le_grid_map(
            df=observability_df,
            value_col=colname,
            title=f"{label}: {orbit_name}",
            colourbar_label="Time to detection [days]",
            output_base=spatial_dir / filename,
            aggregation="median",
            cmap="viridis",
            norm=detect_norm
        )

    ratio_df = observability_df.copy()

    raw = ratio_df["time_to_detect_days_paper_rawN"].to_numpy(dtype=float)
    full = ratio_df[
        "time_to_detect_days_paper_signalEffN"
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(raw) &
        np.isfinite(full) &
        (raw > 0) &
        (full > 0)
    )

    ratio_df["log10_full_to_raw"] = np.nan
    ratio_df.loc[valid, "log10_full_to_raw"] = np.log10(
        full[valid] / raw[valid]
    )

    ratio_values = safe_numeric_array(ratio_df["log10_full_to_raw"])

    if len(ratio_values) > 0:
        limit = max(
            abs(np.nanpercentile(ratio_values, 2)),
            abs(np.nanpercentile(ratio_values, 98)),
            0.1
        )

        plot_le_grid_map(
            df=ratio_df,
            value_col="log10_full_to_raw",
            title=f"Full-model detectability penalty relative to raw geometry: {orbit_name}",
            colourbar_label=r"$\log_{10}(T_{\mathrm{full}}/T_{\mathrm{raw}})$",
            output_base=spatial_dir / "08_full_to_raw_detectability_ratio",
            aggregation="median",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit
        )


def plot_condition_diagnostics(
    polys_nl,
    orbit_name,
    plot_dir
):
    """
    Plot the conditions encountered by one orbit.
    """
    condition_dir = Path(plot_dir) / "retrieval_conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)

    plot_df = sample_dataframe_for_plotting(polys_nl)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(14.0, 8.2)
    )

    variables = [
        (
            "sza_deg",
            "Solar zenith angle [degrees]",
            "Sampled solar zenith angle"
        ),
        (
            "thermal_contrast_K",
            "Thermal contrast [K]",
            "Sampled thermal contrast"
        ),
        (
            "tcc",
            "Total cloud-cover fraction",
            "Sampled cloud cover"
        ),
        (
            "hourfactor",
            "Hourly emission factor",
            "Sampled diurnal factor"
        ),
        (
            "seasonal_factor",
            "Seasonal emission factor",
            "Sampled seasonal factor"
        ),
    ]

    for ax, (column, xlabel, title) in zip(axes.flat, variables):
        values = safe_numeric_array(plot_df[column])

        if len(values) > 0:
            ax.hist(values, bins=60)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Pixel count")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    combined = (
        pd.to_numeric(plot_df["hourfactor"], errors="coerce")
        *
        pd.to_numeric(plot_df["seasonal_factor"], errors="coerce")
    )

    combined_values = safe_numeric_array(combined)

    if len(combined_values) > 0:
        axes.flat[5].hist(combined_values, bins=60)

    axes.flat[5].set_xlabel("Combined emission factor")
    axes.flat[5].set_ylabel("Pixel count")
    axes.flat[5].set_title("Combined temporal emission factor")
    axes.flat[5].grid(alpha=0.2)

    fig.suptitle(f"Orbit-sampled observation conditions: {orbit_name}")
    fig.tight_layout()

    save_results_figure(
        fig,
        condition_dir / "01_sampled_condition_distributions"
    )

    # Quality-retention decomposition
    sza_valid = (
        pd.to_numeric(polys_nl["sza_deg"], errors="coerce")
        <= SZA_THRESHOLD_DEG
    ).mean()

    tc_weight = thermal_contrast_quality_weight(
        pd.to_numeric(
            polys_nl["thermal_contrast_K"],
            errors="coerce"
        ).to_numpy()
    )

    cloud_valid = (
        pd.to_numeric(polys_nl["tcc"], errors="coerce")
        <= TCC_THRESHOLD
    ).mean()

    tc_mean_weight = np.nanmean(tc_weight)

    combined_quality = (
        sza_quality_weight(polys_nl["sza_deg"].to_numpy())
        * thermal_contrast_quality_weight(
            polys_nl["thermal_contrast_K"].to_numpy()
        )
        * cloud_quality_weight(polys_nl["tcc"].to_numpy())
    )

    quality_retention = np.nanmean(combined_quality)

    labels = [
        "SZA-valid fraction",
        "Mean TC weight",
        "Clear-sky fraction",
        "Combined quality retention"
    ]

    values = [
        sza_valid,
        tc_mean_weight,
        cloud_valid,
        quality_retention
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    bars = ax.bar(labels, values)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction or mean weight")
    ax.set_title(f"Retrieval-quality decomposition: {orbit_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    save_results_figure(
        fig,
        condition_dir / "02_quality_retention_decomposition"
    )

    # Hourly sampled condition profiles
    hourly = add_local_solar_time(plot_df)

    hourly["local_hour_bin"] = np.floor(
        hourly["local_solar_time_h"]
    ).astype("Int64")

    hourly_summary = (
        hourly.groupby("local_hour_bin", as_index=False)
        .agg(
            thermal_contrast_K=("thermal_contrast_K", "median"),
            hourfactor=("hourfactor", "median"),
            tcc=("tcc", "median"),
            n_pixels=("time_utc", "size")
        )
    )

    if not hourly_summary.empty:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(9.5, 7.2),
            sharex=True
        )

        axes[0].plot(
            hourly_summary["local_hour_bin"],
            hourly_summary["thermal_contrast_K"],
            marker="o"
        )

        axes[0].axhline(
            TC_BAD_K,
            color="red",
            linestyle=":",
            linewidth=1.0,
            label=f"TC bad threshold: {TC_BAD_K:g} K"
        )

        axes[0].axhline(
            TC_GOOD_K,
            color="green",
            linestyle=":",
            linewidth=1.0,
            label=f"TC good threshold: {TC_GOOD_K:g} K"
        )

        axes[0].set_ylabel("Median thermal contrast [K]")
        axes[0].set_title("Conditions sampled by local solar hour")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        axes[1].plot(
            hourly_summary["local_hour_bin"],
            hourly_summary["hourfactor"],
            marker="o"
        )

        axes[1].axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.0
        )

        axes[1].set_xlabel("Approximate local solar hour")
        axes[1].set_ylabel("Median hourly emission factor")
        axes[1].set_xlim(0, 23)
        axes[1].set_xticks(np.arange(0, 24, 2))
        axes[1].grid(alpha=0.25)

        fig.suptitle(orbit_name)
        fig.tight_layout()

        save_results_figure(
            fig,
            condition_dir / "03_local_time_tc_and_emission_factor"
        )

ABLATION_CONFIGURATIONS = {
    "Raw geometry": {
        "use_sza": False,
        "use_tc": False,
        "use_cloud": False,
        "use_diurnal": False,
        "use_seasonal": False,
    },
    "SZA only": {
        "use_sza": True,
        "use_tc": False,
        "use_cloud": False,
        "use_diurnal": False,
        "use_seasonal": False,
    },
    "Thermal contrast only": {
        "use_sza": False,
        "use_tc": True,
        "use_cloud": False,
        "use_diurnal": False,
        "use_seasonal": False,
    },
    "Cloud only": {
        "use_sza": False,
        "use_tc": False,
        "use_cloud": True,
        "use_diurnal": False,
        "use_seasonal": False,
    },
    "Diurnal emissions only": {
        "use_sza": False,
        "use_tc": False,
        "use_cloud": False,
        "use_diurnal": True,
        "use_seasonal": False,
    },
    "Seasonal emissions only": {
        "use_sza": False,
        "use_tc": False,
        "use_cloud": False,
        "use_diurnal": False,
        "use_seasonal": True,
    },
    "All retrieval quality": {
        "use_sza": True,
        "use_tc": True,
        "use_cloud": True,
        "use_diurnal": False,
        "use_seasonal": False,
    },
    "Full model": {
        "use_sza": True,
        "use_tc": True,
        "use_cloud": True,
        "use_diurnal": True,
        "use_seasonal": True,
    },
}


def build_sampling_summary_for_configuration(
    polys_nl,
    use_sza=True,
    use_tc=True,
    use_cloud=True,
    use_diurnal=True,
    use_seasonal=True
):
    """
    Recalculate pass and grid-cell summaries from an existing complete
    pixel dataset without rerunning the orbit propagation.
    """
    if polys_nl.empty:
        raise ValueError("polys_nl is empty.")

    df = polys_nl.copy()

    df["time_utc"] = pd.to_datetime(
        df["time_utc"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["time_utc", "iy", "ix", "pass_id"])

    # --------------------------------------------------------
    # Retrieval-quality terms
    # --------------------------------------------------------
    if use_sza:
        df["w_sza_config"] = sza_quality_weight(
            df["sza_deg"].to_numpy()
        )
    else:
        df["w_sza_config"] = 1.0

    if use_tc:
        df["w_tc_config"] = thermal_contrast_quality_weight(
            df["thermal_contrast_K"].to_numpy()
        )
    else:
        df["w_tc_config"] = 1.0

    if use_cloud:
        df["w_cloud_config"] = cloud_quality_weight(
            df["tcc"].to_numpy()
        )
    else:
        df["w_cloud_config"] = 1.0

    df["w_quality_config"] = (
        df["w_sza_config"]
        * df["w_tc_config"]
        * df["w_cloud_config"]
    )

    # --------------------------------------------------------
    # Temporal emission terms
    # --------------------------------------------------------
    if use_diurnal:
        hourfactor = pd.to_numeric(
            df["hourfactor"],
            errors="coerce"
        ).fillna(1.0)
    else:
        hourfactor = pd.Series(
            np.ones(len(df)),
            index=df.index
        )

    if use_seasonal:
        seasonal_factor = pd.to_numeric(
            df["seasonal_factor"],
            errors="coerce"
        ).fillna(1.0)
    else:
        seasonal_factor = pd.Series(
            np.ones(len(df)),
            index=df.index
        )

    df["hourfactor_config"] = hourfactor
    df["seasonal_factor_config"] = seasonal_factor

    df["combined_emission_factor_config"] = (
        df["hourfactor_config"]
        * df["seasonal_factor_config"]
    )

    df["w_signal_config"] = (
        df["w_quality_config"]
        * df["combined_emission_factor_config"] ** 2
    )

    # --------------------------------------------------------
    # Collapse pixels within each cell and pass
    # --------------------------------------------------------
    df_pass = (
        df.groupby(["iy", "ix", "pass_id"], as_index=False)
        .agg(
            time_utc=("time_utc", "min"),
            N_raw_this_pass=("time_utc", "size"),
            N_eff_this_pass=("w_quality_config", "sum"),
            N_signal_eff_this_pass=("w_signal_config", "sum"),
            combined_emission_factor=(
                "combined_emission_factor_config",
                "mean"
            ),
        )
    )

    counts = (
        df_pass.groupby(["iy", "ix"], as_index=False)
        .agg(
            N_obs=("pass_id", "size"),
            N_raw_per_pass_mean=("N_raw_this_pass", "mean"),
            N_raw_per_pass_median=("N_raw_this_pass", "median"),
            N_eff_per_pass_mean=("N_eff_this_pass", "mean"),
            N_eff_per_pass_median=("N_eff_this_pass", "median"),
            N_signal_eff_per_pass_mean=(
                "N_signal_eff_this_pass",
                "mean"
            ),
            N_signal_eff_per_pass_median=(
                "N_signal_eff_this_pass",
                "median"
            ),
            combined_emission_factor_mean=(
                "combined_emission_factor",
                "mean"
            ),
        )
    )

    # --------------------------------------------------------
    # Revisit statistics
    # --------------------------------------------------------
    sorted_passes = df_pass.sort_values(
        ["iy", "ix", "time_utc"]
    ).copy()

    sorted_passes["dt_s"] = (
        sorted_passes
        .groupby(["iy", "ix"])["time_utc"]
        .diff()
        .dt.total_seconds()
    )

    revisit = (
        sorted_passes.groupby(["iy", "ix"], as_index=False)
        .agg(
            dt_median_s=("dt_s", "median"),
            dt_min_s=("dt_s", "min"),
            dt_max_s=("dt_s", "max")
        )
    )

    sampling_summary = counts.merge(
        revisit,
        on=["iy", "ix"],
        how="left"
    )

    return sampling_summary


def summarize_detectability_configuration(
    orbit_name,
    configuration_name,
    observability_df,
    total_emitting_cells
):
    """
    Summarize detectability over both finite cells and the full emitting-cell
    reference population.
    Distribution statistics such as the median, P10 and P90 are calculated only
    over sampled emitting cells with finite detectability. Coverage relative to
    all emitting cells is reported separately and must be interpreted alongside
    these finite-only statistics.
    """
    df = observability_df.copy()

    detect_col = "time_to_detect_days_paper_signalEffN"

    emitting = df[
        pd.to_numeric(df["nh3_kg_yr"], errors="coerce") > 0
    ].copy()

    values_all_sampled = pd.to_numeric(
        emitting[detect_col],
        errors="coerce"
    ).to_numpy(dtype=float)

    finite_mask = np.isfinite(values_all_sampled)
    finite_values = values_all_sampled[finite_mask]

    sampled_emitting_cells = len(emitting)
    finite_cells = int(finite_mask.sum())

    if len(finite_values) > 0:
        median_days = float(np.median(finite_values))
        p10_days = float(np.percentile(finite_values, 10))
        p90_days = float(np.percentile(finite_values, 90))

        finite_emitting = emitting.loc[finite_mask].copy()

        weights = finite_emitting[
            "nh3_kg_yr"
        ].to_numpy(dtype=float)

        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)

        weighted_mean_days = float(
            np.sum(finite_values * weights)
        )
    else:
        median_days = np.nan
        p10_days = np.nan
        p90_days = np.nan
        weighted_mean_days = np.nan

    result = {
        "orbit_name": orbit_name,
        "configuration": configuration_name,
        "total_emitting_cells": int(total_emitting_cells),
        "sampled_emitting_cells": int(sampled_emitting_cells),
        "finite_detectable_cells": int(finite_cells),
        "fraction_sampled_emitting_cells": (
            sampled_emitting_cells / total_emitting_cells
            if total_emitting_cells > 0
            else np.nan
        ),
        "fraction_finite_all_emitting_cells": (
            finite_cells / total_emitting_cells
            if total_emitting_cells > 0
            else np.nan
        ),
        "median_detect_days_finite": median_days,
        "p10_detect_days_finite": p10_days,
        "p90_detect_days_finite": p90_days,
        "weighted_mean_detect_days_finite": weighted_mean_days,
    }

    for threshold in DETECTION_THRESHOLDS_DAYS:
        count_within = int(
            np.sum(finite_values <= threshold)
        )

        result[
            f"fraction_detectable_{int(threshold)}d_all_emitting"
        ] = (
            count_within / total_emitting_cells
            if total_emitting_cells > 0
            else np.nan
        )

    return result


def run_filter_ablation_analysis(
    orbit_name,
    polys_nl,
    emis_cell,
    output_dir,
    sigma_I_molec_cm2=5e15,
    sigma_X_molec_cm2=0.75e15,
    pressure_hpa=1000.0,
    wind_kmh=18.0,
    W_km=10.0
):
    """
    Run all filter/ablation configurations using one existing pixel-level dataset.
    Only quality and emission weights are enabled or disabled.
    No orbit propagation or environmental resampling is repeated.
    """
    ablation_dir = Path(output_dir) / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    total_emitting_cells = int(
        (
            pd.to_numeric(
                emis_cell["nh3_kg_yr"],
                errors="coerce"
            ) > 0
        ).sum()
    )

    ablation_rows = []
    ablation_observability = {}

    for config_name, config in ABLATION_CONFIGURATIONS.items():

        print(
            f"[ABLATION] {orbit_name}: {config_name}"
        )

        config_sampling = build_sampling_summary_for_configuration(
            polys_nl=polys_nl,
            **config
        )

        config_observability = compute_observability(
            sampling_summary=config_sampling,
            emis_cell=emis_cell,
            sigma_I_molec_cm2=sigma_I_molec_cm2,
            sigma_X_molec_cm2=sigma_X_molec_cm2,
            F_clear=1.0,
            pressure_hpa=pressure_hpa,
            wind_kmh=wind_kmh,
            W_km=W_km
        )

        config_observability["orbit_name"] = orbit_name
        config_observability["configuration"] = config_name

        summary = summarize_detectability_configuration(
            orbit_name=orbit_name,
            configuration_name=config_name,
            observability_df=config_observability,
            total_emitting_cells=total_emitting_cells
        )

        ablation_rows.append(summary)
        ablation_observability[config_name] = config_observability

        config_sampling.to_parquet(
            ablation_dir /
            f"sampling_{config_name.lower().replace(' ', '_')}.parquet",
            index=False
        )

        config_observability.to_parquet(
            ablation_dir /
            f"observability_{config_name.lower().replace(' ', '_')}.parquet",
            index=False
        )

    ablation_summary = pd.DataFrame(ablation_rows)

    ablation_summary.to_csv(
        ablation_dir / "ablation_summary.csv",
        index=False
    )

    ablation_summary.to_parquet(
        ablation_dir / "ablation_summary.parquet",
        index=False
    )

    return ablation_summary, ablation_observability


def plot_single_orbit_ablation(
    ablation_summary,
    orbit_name,
    plot_dir
):
    """
    Plot detectability time and all-emitting-cell completeness for one orbit.
    """
    if ablation_summary.empty:
        return

    config_order = list(ABLATION_CONFIGURATIONS.keys())

    plot_df = (
        ablation_summary
        .set_index("configuration")
        .reindex(config_order)
        .reset_index()
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11.0, 8.0),
        sharex=True
    )

    x = np.arange(len(plot_df))

    axes[0].plot(
        x,
        plot_df["median_detect_days_finite"],
        marker="o",
        linewidth=1.8
    )

    axes[0].fill_between(
        x,
        plot_df["p10_detect_days_finite"],
        plot_df["p90_detect_days_finite"],
        alpha=0.2,
        label="P10 to P90"
    )

    positive_values = safe_numeric_array(
        plot_df["median_detect_days_finite"],
        positive_only=True
    )

    if len(positive_values) > 0:
        axes[0].set_yscale("log")

    axes[0].set_ylabel("Detectability time [days]")
    axes[0].set_title(
        f"Influence of model components: {orbit_name}"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        x,
        plot_df["fraction_finite_all_emitting_cells"],
        marker="o",
        label="Finite detectability"
    )

    for threshold in DETECTION_THRESHOLDS_DAYS:
        axes[1].plot(
            x,
            plot_df[
                f"fraction_detectable_{int(threshold)}d_all_emitting"
            ],
            marker="o",
            label=f"Detectable within {int(threshold)} days"
        )

    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of all emitting cells")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        plot_df["configuration"],
        rotation=30,
        ha="right"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2)

    fig.tight_layout()

    save_results_figure(
        fig,
        Path(plot_dir) / "ablation" / "single_orbit_ablation"
    )
def generate_single_orbit_results(
    orbit_name,
    track_coarse,
    polys_nl,
    sampling_summary,
    emis_cell,
    observability_df,
    out_dir
):
    """
    Generate all principal single-orbit Results figures and ablation outputs.
    """
    out_dir = Path(out_dir)
    results_plot_dir = out_dir / "plots" / "results"

    propagation_dir = results_plot_dir / "propagation"
    footprint_dir = results_plot_dir / "footprints"

    propagation_dir.mkdir(parents=True, exist_ok=True)
    footprint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating Results figures for {orbit_name}...")

    pass_diag = build_pass_diagnostics(polys_nl)

    if GENERATE_SINGLE_ORBIT_FIGURES:
        plot_ground_track(
            track_coarse=track_coarse,
            orbit_name=orbit_name,
            output_base=propagation_dir / "01_ground_track"
        )

        plot_local_solar_time_sampling(
            pass_diag=pass_diag,
            orbit_name=orbit_name,
            output_base=propagation_dir / "02_local_solar_time"
        )

        plot_representative_footprints(
            polys_nl=polys_nl,
            orbit_name=orbit_name,
            output_base=footprint_dir / "01_representative_scanline"
        )

        plot_pixel_geometry(
            polys_nl=polys_nl,
            orbit_name=orbit_name,
            output_base=footprint_dir / "02_pixel_geometry"
        )

        plot_single_orbit_spatial_maps(
            orbit_name=orbit_name,
            sampling_summary=sampling_summary,
            emis_cell=emis_cell,
            observability_df=observability_df,
            plot_dir=results_plot_dir
        )

        plot_condition_diagnostics(
            polys_nl=polys_nl,
            orbit_name=orbit_name,
            plot_dir=results_plot_dir
        )

    ablation_summary, ablation_observability = (
        run_filter_ablation_analysis(
            orbit_name=orbit_name,
            polys_nl=polys_nl,
            emis_cell=emis_cell,
            output_dir=out_dir,
            sigma_I_molec_cm2=5e15,
            sigma_X_molec_cm2=0.75e15,
            pressure_hpa=1000.0,
            wind_kmh=18.0,
            W_km=10.0
        )
    )

    if GENERATE_SINGLE_ORBIT_FIGURES:
        plot_single_orbit_ablation(
            ablation_summary=ablation_summary,
            orbit_name=orbit_name,
            plot_dir=results_plot_dir
        )



    # ------------------------------------------------------------
    # Cache compact single-orbit results for final comparisons
    # ------------------------------------------------------------
    # A raw observation is one row of polys_nl, corresponding to one
    # synthetic pixel observation inside the analysis domain before
    # retrieval-quality or emission weighting is applied.
    raw_observation_count = int(len(polys_nl))

    # A raw regional pass is a unique pass_id that produced at least
    # one raw pixel observation inside the analysis domain.
    raw_pass_count = int(polys_nl["pass_id"].nunique())

    # ------------------------------------------------------------
    # Local solar time histogram for multi-orbit comparison
    # ------------------------------------------------------------
    # make_local_time_diagnostics() has already added the
    # local_solar_hour column to polys_nl earlier in run_orbit_case().
    local_time_bins = np.arange(0.0, 24.0 + 0.5, 0.5)
    local_time_bin_centers = (
        0.5 * (local_time_bins[:-1] + local_time_bins[1:])
    )

    if "local_solar_hour" in polys_nl.columns:
        local_time_values = pd.to_numeric(
            polys_nl["local_solar_hour"],
            errors="coerce"
        ).to_numpy(dtype=float)

        local_time_values = local_time_values[
            np.isfinite(local_time_values)
        ]

        local_time_pixel_counts, _ = np.histogram(
            local_time_values,
            bins=local_time_bins
        )

        if local_time_pixel_counts.sum() > 0:
            local_time_pixel_fraction = (
                local_time_pixel_counts
                / local_time_pixel_counts.sum()
            )
        else:
            local_time_pixel_fraction = np.zeros(
                len(local_time_bin_centers),
                dtype=float
            )

    else:
        print(
            f"[WARNING] {orbit_name}: local_solar_hour is missing. "
            "Its multi-orbit local-time curve will be empty."
        )

        local_time_pixel_counts = np.zeros(
            len(local_time_bin_centers),
            dtype=int
        )

        local_time_pixel_fraction = np.zeros(
            len(local_time_bin_centers),
            dtype=float
        )


    # ------------------------------------------------------------
    # Annual diurnal observation and emission-condition profile
    # ------------------------------------------------------------
    # local_solar_hour was added by make_local_time_diagnostics().
    # hourfactor was sampled from the gridded diurnal emission file.
    required_diurnal_columns = {
        "local_solar_hour",
        "hourfactor",
    }

    if required_diurnal_columns.issubset(polys_nl.columns):
        diurnal_profile_df = polys_nl[
            ["local_solar_hour", "hourfactor"]
        ].copy()

        diurnal_profile_df["local_solar_hour"] = pd.to_numeric(
            diurnal_profile_df["local_solar_hour"],
            errors="coerce"
        )

        diurnal_profile_df["hourfactor"] = pd.to_numeric(
            diurnal_profile_df["hourfactor"],
            errors="coerce"
        )

        diurnal_profile_df = diurnal_profile_df.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna(
            subset=["local_solar_hour", "hourfactor"]
        )

        # One-hour local-solar-time bins: 0, 1, ..., 23.
        diurnal_profile_df["local_hour_bin"] = np.floor(
            diurnal_profile_df["local_solar_hour"]
        ).astype(int)

        diurnal_profile_df["local_hour_bin"] = (
            diurnal_profile_df["local_hour_bin"] % 24
        )

        annual_diurnal_profile = (
            diurnal_profile_df
            .groupby("local_hour_bin", as_index=False)
            .agg(
                observation_count=("hourfactor", "size"),
                mean_observed_hourfactor=("hourfactor", "mean"),
                median_observed_hourfactor=("hourfactor", "median"),
            )
        )

        # Guarantee that all 24 hours are represented.
        complete_hours = pd.DataFrame({
            "local_hour_bin": np.arange(24, dtype=int)
        })

        annual_diurnal_profile = complete_hours.merge(
            annual_diurnal_profile,
            on="local_hour_bin",
            how="left"
        )

        annual_diurnal_profile["observation_count"] = (
            annual_diurnal_profile["observation_count"]
            .fillna(0)
            .astype(int)
        )

        total_hourly_observations = (
            annual_diurnal_profile["observation_count"].sum()
        )

        if total_hourly_observations > 0:
            annual_diurnal_profile["observation_fraction"] = (
                annual_diurnal_profile["observation_count"]
                / total_hourly_observations
            )
        else:
            annual_diurnal_profile["observation_fraction"] = 0.0

    else:
        missing_diurnal_columns = sorted(
            required_diurnal_columns.difference(polys_nl.columns)
        )

        print(
            f"[WARNING] {orbit_name}: missing columns required "
            f"for annual diurnal comparison: "
            f"{missing_diurnal_columns}"
        )

        annual_diurnal_profile = pd.DataFrame({
            "local_hour_bin": np.arange(24, dtype=int),
            "observation_count": np.zeros(24, dtype=int),
            "mean_observed_hourfactor": np.full(24, np.nan),
            "median_observed_hourfactor": np.full(24, np.nan),
            "observation_fraction": np.zeros(24, dtype=float),
        })

    # ------------------------------------------------------------
    # Orbit-wide raw, quality-effective and signal-effective totals
    # ------------------------------------------------------------
    # N_raw:
    # Every raw pixel observation within the analysis domain
    # contributes exactly one.
    N_raw_total = float(len(polys_nl))

    # Recalculate the three retrieval-quality terms using exactly
    # the same functions as build_sampling_summary_by_pass().
    if "sza_deg" in polys_nl.columns:
        w_sza_total = sza_quality_weight(
            pd.to_numeric(
                polys_nl["sza_deg"],
                errors="coerce"
            ).to_numpy(dtype=float)
        )
    else:
        w_sza_total = np.ones(len(polys_nl), dtype=float)

    if "thermal_contrast_K" in polys_nl.columns:
        w_tc_total = thermal_contrast_quality_weight(
            pd.to_numeric(
                polys_nl["thermal_contrast_K"],
                errors="coerce"
            ).to_numpy(dtype=float)
        )
    else:
        w_tc_total = np.ones(len(polys_nl), dtype=float)

    if "tcc" in polys_nl.columns:
        w_cloud_total = cloud_quality_weight(
            pd.to_numeric(
                polys_nl["tcc"],
                errors="coerce"
            ).to_numpy(dtype=float)
        )
    else:
        w_cloud_total = np.ones(len(polys_nl), dtype=float)

    # Combined retrieval-quality contribution:
    # SZA x thermal contrast x cloud cover.
    w_quality_total = (
        w_sza_total
        * w_tc_total
        * w_cloud_total
    )

    # N_eff:
    # Total effective observations after SZA, TC and cloud weighting.
    N_eff_total = float(np.sum(w_quality_total))

    # Diurnal emission factor.
    if "hourfactor" in polys_nl.columns:
        hourfactor_total = (
            pd.to_numeric(
                polys_nl["hourfactor"],
                errors="coerce"
            )
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
    else:
        hourfactor_total = np.ones(
            len(polys_nl),
            dtype=float
        )

    # Seasonal emission factor.
    if "seasonal_factor" in polys_nl.columns:
        seasonal_factor_total = (
            pd.to_numeric(
                polys_nl["seasonal_factor"],
                errors="coerce"
            )
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
    else:
        seasonal_factor_total = np.ones(
            len(polys_nl),
            dtype=float
        )

    combined_emission_factor_total = (
        hourfactor_total
        * seasonal_factor_total
    )

    # Signal-effective contribution:
    # retrieval quality x combined emission factor squared.
    w_signal_total = (
        w_quality_total
        * combined_emission_factor_total**2
    )

    # N_sig_eff:
    # Total signal-effective observations after retrieval-quality
    # weighting and diurnal plus seasonal emission scaling.
    N_sig_eff_total = float(np.sum(w_signal_total))

    # Retention/effective ratios.
    if N_raw_total > 0:
        quality_retention_total = (
            N_eff_total / N_raw_total
        )

        signal_effective_ratio_total = (
            N_sig_eff_total / N_raw_total
        )
    else:
        quality_retention_total = np.nan
        signal_effective_ratio_total = np.nan

    # ------------------------------------------------------------
    # Environmental-data completeness diagnostics
    # ------------------------------------------------------------
    data_completeness = calculate_orbit_data_completeness(
        orbit_name=orbit_name,
        polys_nl=polys_nl,
        emis_cell=emis_cell
    )

    print("\n" + "=" * 80)
    print(f"DATA COMPLETENESS: {orbit_name}")
    print("=" * 80)

    print(
        f"Fraction of non-emitting cells: "
        f"{data_completeness['fraction_non_emitting_cells']:.6f} "
        f"({100.0 * data_completeness['fraction_non_emitting_cells']:.3f}%)"
    )

    print(
        f"Fraction missing TC: "
        f"{data_completeness['fraction_missing_tc']:.6f} "
        f"({100.0 * data_completeness['fraction_missing_tc']:.3f}%)"
    )

    print(
        f"Fraction missing TCC: "
        f"{data_completeness['fraction_missing_tcc']:.6f} "
        f"({100.0 * data_completeness['fraction_missing_tcc']:.3f}%)"
    )

    print(
        f"Fraction missing diurnal factor: "
        f"{data_completeness['fraction_missing_diurnal_factor']:.6f} "
        f"({100.0 * data_completeness['fraction_missing_diurnal_factor']:.3f}%)"
    )

    print(
        f"Fraction missing seasonal factor: "
        f"{data_completeness['fraction_missing_seasonal_factor']:.6f} "
        f"({100.0 * data_completeness['fraction_missing_seasonal_factor']:.3f}%)"
    )

    ORBIT_RESULTS_CACHE[orbit_name] = {
        "track_coarse": track_coarse[
            ["time_utc", "lat", "lon", "alt_km"]
        ].copy(),

        "pass_diagnostics": pass_diag.copy(),
        "sampling_summary": sampling_summary.copy(),

        "observability_full": (
            ablation_observability["Full model"].copy()
        ),

        "ablation_summary": ablation_summary.copy(),

        # New raw geometric sampling totals
        "raw_pass_count": raw_pass_count,
        "raw_observation_count": raw_observation_count,
        # Data-completeness diagnostics
        "data_completeness": data_completeness,
        #Local solar time histogram for multiple orbit comparison
        "local_time_bin_centers": local_time_bin_centers.copy(),
        "local_time_pixel_counts": local_time_pixel_counts.copy(),
        "local_time_pixel_fraction": local_time_pixel_fraction.copy(),

        # Annual diurnal profile used for the final multi-orbit plot
        "annual_diurnal_profile": annual_diurnal_profile.copy(),

        # Orbit-wide effective observation totals
        "N_raw_total": N_raw_total,
        "N_eff_total": N_eff_total,
        "N_sig_eff_total": N_sig_eff_total,

        # Orbit-wide retention/effective ratios
        "quality_retention_total": quality_retention_total,
        "signal_effective_ratio_total": (
            signal_effective_ratio_total
        ),
    }

    cache_file = out_dir / "comparison_cache.pkl"
    with open(cache_file, "wb") as cache_stream:
        pickle.dump(
            {"orbit_name": orbit_name, "cache": ORBIT_RESULTS_CACHE[orbit_name]},
            cache_stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[CACHE] Saved standalone comparison input: {cache_file}")

    print(
        f"[CACHE] {orbit_name}: "
        f"{raw_pass_count:,} raw regional passes, "
        f"N_raw={N_raw_total:,.0f}, "
        f"N_eff={N_eff_total:,.2f}, "
        f"N_sig_eff={N_sig_eff_total:,.2f}, "
        f"N_eff/N_raw={quality_retention_total:.3f}, "
        f"N_sig_eff/N_raw={signal_effective_ratio_total:.3f}"
    )


    print(
        f"Finished generating Results figures for {orbit_name}."
    )


def plot_multi_orbit_ablation_heatmap(
    combined_ablation,
    output_dir
):
    """
    Plot median detectability time and orbit rank for every configuration.
    """
    if combined_ablation.empty:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_order = list(ABLATION_CONFIGURATIONS.keys())

    time_table = combined_ablation.pivot(
        index="orbit_name",
        columns="configuration",
        values="median_detect_days_finite"
    )

    time_table = time_table.reindex(columns=config_order)

    # Rank 1 is the shortest median detectability time.
    rank_table = time_table.rank(
        axis=0,
        method="min",
        ascending=True
    )

    # --------------------------------------------------------
    # Detectability heatmap
    # --------------------------------------------------------
    plot_values = np.log10(
        np.maximum(time_table.to_numpy(dtype=float), PLOT_EPS)
    )

    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 1.35 * len(config_order)),
            max(4.5, 0.65 * len(time_table.index) + 2.0)
        )
    )

    image = ax.imshow(
        plot_values,
        aspect="auto",
        cmap="viridis"
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        r"$\log_{10}$ median detectability time [days]"
    )

    ax.set_xticks(np.arange(len(config_order)))
    ax.set_xticklabels(
        config_order,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(np.arange(len(time_table.index)))
    ax.set_yticklabels(time_table.index)

    for i in range(time_table.shape[0]):
        for j in range(time_table.shape[1]):
            value = time_table.iloc[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                )

    ax.set_title("Median detectability time by orbit and model configuration")
    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir / "01_ablation_detectability_heatmap"
    )

    # --------------------------------------------------------
    # Rank heatmap
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 1.35 * len(config_order)),
            max(4.5, 0.65 * len(rank_table.index) + 2.0)
        )
    )

    image = ax.imshow(
        rank_table.to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis_r",
        vmin=1,
        vmax=max(1, len(rank_table.index))
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Orbit rank, where 1 is best")

    ax.set_xticks(np.arange(len(config_order)))
    ax.set_xticklabels(
        config_order,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(np.arange(len(rank_table.index)))
    ax.set_yticklabels(rank_table.index)

    for i in range(rank_table.shape[0]):
        for j in range(rank_table.shape[1]):
            value = rank_table.iloc[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=9
                )

    ax.set_title("Orbit-ranking sensitivity to model configuration")
    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir / "02_ablation_rank_heatmap"
    )


def empirical_cdf(values):
    values = safe_numeric_array(
        values,
        positive_only=True
    )

    if len(values) == 0:
        return np.array([]), np.array([])

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


def plot_final_detectability_comparison(
    output_dir
):
    """
    Plot final-model ECDFs and median/P10/P90 orbit comparison.
    """
    if not ORBIT_RESULTS_CACHE:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # --------------------------------------------------------
    # Full-model ECDF
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    for orbit_name, cache in ORBIT_RESULTS_CACHE.items():
        df = cache["observability_full"]

        emitting = df[
            pd.to_numeric(
                df["nh3_kg_yr"],
                errors="coerce"
            ) > 0
        ]

        x, y = empirical_cdf(
            emitting[
                "time_to_detect_days_paper_signalEffN"
            ]
        )

        if len(x) == 0:
            continue

        ax.step(
            x,
            y,
            where="post",
            label=orbit_name
        )

        summary_rows.append({
            "orbit_name": orbit_name,
            "median": float(np.percentile(x, 50)),
            "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)),
        })

    ax.set_xscale("log")
    ax.set_xlabel("Full-model detectability time [days]")
    ax.set_ylabel("Fraction of finite emitting cells")
    ax.set_title("Distribution of full-model detectability time")
    ax.grid(alpha=0.25)
    ax.legend()

    for threshold in DETECTION_THRESHOLDS_DAYS:
        ax.axvline(
            threshold,
            color="0.5",
            linestyle=":",
            linewidth=0.9
        )

    save_results_figure(
        fig,
        output_dir / "03_full_model_detectability_ecdf"
    )

    # --------------------------------------------------------
    # Median with P10-P90 interval
    # --------------------------------------------------------
    interval_df = pd.DataFrame(summary_rows)

    if interval_df.empty:
        return

    interval_df = interval_df.sort_values("median")

    y_positions = np.arange(len(interval_df))

    lower = interval_df["median"] - interval_df["p10"]
    upper = interval_df["p90"] - interval_df["median"]

    fig, ax = plt.subplots(
        figsize=(9.0, max(4.5, 0.65 * len(interval_df) + 2.0))
    )

    ax.errorbar(
        interval_df["median"],
        y_positions,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4
    )

    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(interval_df["orbit_name"])
    ax.invert_yaxis()

    ax.set_xlabel("Full-model detectability time [days]")
    ax.set_title("Median detectability time with P10-P90 interval")
    ax.grid(alpha=0.25)

    save_results_figure(
        fig,
        output_dir / "04_full_model_detection_intervals"
    )


def plot_orbit_condition_comparison(output_dir):
    """
    Compare pass-level observation conditions across all orbit cases.
    """
    if not ORBIT_RESULTS_CACHE:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orbit_names = list(ORBIT_RESULTS_CACHE.keys())
    x = np.arange(len(orbit_names))

    metrics = {
        "Median local solar time [h]": [],
        "Median SZA [degrees]": [],
        "Median absolute TC [K]": [],
        "Clear-sky pass fraction": [],
        "Median hourly factor": [],
        "Median seasonal factor": [],
    }

    for orbit_name in orbit_names:
        pass_diag = ORBIT_RESULTS_CACHE[
            orbit_name
        ]["pass_diagnostics"]

        metrics["Median local solar time [h]"].append(
            np.nanmedian(pass_diag["local_solar_time_h"])
        )

        metrics["Median SZA [degrees]"].append(
            np.nanmedian(pass_diag["sza_median_deg"])
        )

        metrics["Median absolute TC [K]"].append(
            np.nanmedian(
                np.abs(pass_diag["tc_median_K"])
            )
        )

        metrics["Clear-sky pass fraction"].append(
            np.nanmean(
                pass_diag["tcc_median"] <= TCC_THRESHOLD
            )
        )

        metrics["Median hourly factor"].append(
            np.nanmedian(pass_diag["hourfactor_median"])
        )

        metrics["Median seasonal factor"].append(
            np.nanmedian(pass_diag["seasonal_factor_median"])
        )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(14.5, 8.3)
    )

    for ax, (metric_name, values) in zip(
        axes.flat,
        metrics.items()
    ):
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(
            orbit_names,
            rotation=30,
            ha="right"
        )
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Orbit-sampled observation conditions")
    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir / "05_orbit_condition_comparison"
    )


def plot_raw_sampling_and_revisit_comparison(output_dir):
    """
    Generate targeted multi-orbit geometry and revisit comparisons.

    Outputs
    -------
    06_raw_pass_count
        Total number of unique regional passes for each orbit.

    07_median_revisit_common_emitting_cells
        Median cell-level revisit time for each orbit, evaluated over
        the same common emitting-cell population.

    08_raw_observation_count
        Total number of raw synthetic pixel observations inside the
        analysis domain before quality or emission weighting.

    09_revisit_time_ecdf_common_emitting_cells
        ECDF of median revisit time across common emitting cells,
        with one curve per orbit.
    """
    if not ORBIT_RESULTS_CACHE:
        print(
            "[WARNING] No cached orbit results were available for "
            "raw sampling and revisit comparison."
        )
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orbit_names = list(ORBIT_RESULTS_CACHE.keys())

    # A distinct colour is assigned to every orbit and reused in all
    # four plots.
    colour_map = plt.get_cmap("tab10")
    orbit_colours = {
        orbit_name: colour_map(i % 10)
        for i, orbit_name in enumerate(orbit_names)
    }

    def annotate_bars(ax, bars, integer_values=False):
        """
        Add value labels above bars.
        """
        finite_heights = [
            bar.get_height()
            for bar in bars
            if np.isfinite(bar.get_height())
        ]

        if finite_heights:
            maximum_height = max(finite_heights)
            vertical_offset = (
                0.015 * maximum_height
                if maximum_height > 0
                else 0.02
            )
        else:
            vertical_offset = 0.02

        for bar in bars:
            value = bar.get_height()

            if not np.isfinite(value):
                continue

            label = (
                f"{int(round(value)):,}"
                if integer_values
                else f"{value:.2f}"
            )

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + vertical_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=9
            )

    # ========================================================
    # Identify the common emitting-cell population
    # ========================================================
    # For each orbit, observability_full contains cells formed by
    # merging the emission grid with the orbit sampling summary.
    #
    # We retain cells that:
    #   1. have positive NH3 emissions;
    #   2. have a finite positive median revisit time; and
    #   3. are present for every selected orbit.
    #
    # The intersection ensures that differences between ECDF curves
    # are caused by revisit performance rather than different spatial
    # cell populations.
    revisit_by_orbit = {}
    common_cell_keys = None

    for orbit_name in orbit_names:
        observability_df = (
            ORBIT_RESULTS_CACHE[orbit_name]["observability_full"]
            .copy()
        )

        required_columns = {
            "iy",
            "ix",
            "nh3_kg_yr",
            "dt_median_s",
        }
        missing_columns = (
            required_columns.difference(observability_df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{orbit_name}: columns required for the revisit "
                f"comparison are missing: {sorted(missing_columns)}"
            )

        revisit_df = observability_df[
            ["iy", "ix", "nh3_kg_yr", "dt_median_s"]
        ].copy()

        revisit_df["nh3_kg_yr"] = pd.to_numeric(
            revisit_df["nh3_kg_yr"],
            errors="coerce"
        )

        revisit_df["median_revisit_days"] = (
            pd.to_numeric(
                revisit_df["dt_median_s"],
                errors="coerce"
            )
            / 86400.0
        )

        revisit_df = revisit_df[
            (revisit_df["nh3_kg_yr"] > 0)
            & np.isfinite(revisit_df["median_revisit_days"])
            & (revisit_df["median_revisit_days"] > 0)
        ].copy()

        # There should normally be one row per cell already.
        # Grouping protects the comparison against accidental
        # duplicate cell rows.
        revisit_df = (
            revisit_df
            .groupby(["iy", "ix"], as_index=False)
            .agg(
                median_revisit_days=(
                    "median_revisit_days",
                    "median"
                )
            )
        )

        revisit_by_orbit[orbit_name] = revisit_df

        orbit_cell_keys = set(
            zip(
                revisit_df["iy"].astype(int),
                revisit_df["ix"].astype(int)
            )
        )

        if common_cell_keys is None:
            common_cell_keys = orbit_cell_keys
        else:
            common_cell_keys = (
                common_cell_keys.intersection(orbit_cell_keys)
            )

    if common_cell_keys is None:
        common_cell_keys = set()

    if len(common_cell_keys) == 0:
        print(
            "[WARNING] No common emitting cells with finite revisit "
            "times were found across all selected orbits. "
            "The raw-count plots will still be generated, but the "
            "revisit plots will be skipped."
        )

    # Convert the common keys to a DataFrame for an explicit inner join.
    common_cells_df = pd.DataFrame(
        sorted(common_cell_keys),
        columns=["iy", "ix"]
    )

    # ========================================================
    # Assemble comparison metrics
    # ========================================================
    comparison_rows = []
    common_revisit_values = {}

    for orbit_name in orbit_names:
        cache = ORBIT_RESULTS_CACHE[orbit_name]

        raw_pass_count = int(
            cache.get(
                "raw_pass_count",
                cache["pass_diagnostics"]["pass_id"].nunique()
            )
        )

        raw_observation_count = cache.get(
            "raw_observation_count",
            np.nan
        )

        if len(common_cells_df) > 0:
            common_orbit_df = common_cells_df.merge(
                revisit_by_orbit[orbit_name],
                on=["iy", "ix"],
                how="inner",
                validate="one_to_one"
            )

            revisit_values = safe_numeric_array(
                common_orbit_df["median_revisit_days"],
                positive_only=True
            )
        else:
            revisit_values = np.array([], dtype=float)

        common_revisit_values[orbit_name] = revisit_values

        comparison_rows.append({
            "orbit_name": orbit_name,
            "raw_pass_count": raw_pass_count,
            "raw_observation_count": raw_observation_count,
            "median_revisit_days_common_emitting_cells": (
                float(np.median(revisit_values))
                if len(revisit_values) > 0
                else np.nan
            ),
            "common_emitting_cells_with_revisit": (
                int(len(revisit_values))
            ),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # Save the exact data used by the bar charts.
    comparison_df.to_csv(
        output_dir / "raw_sampling_and_revisit_comparison.csv",
        index=False
    )

    comparison_df.to_parquet(
        output_dir / "raw_sampling_and_revisit_comparison.parquet",
        index=False
    )

    print("\n" + "=" * 80)
    print("RAW SAMPLING AND REVISIT COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(
        f"Common emitting cells with finite revisit for all orbits: "
        f"{len(common_cell_keys)}"
    )

    x_positions = np.arange(len(orbit_names))
    colours = [
        orbit_colours[orbit_name]
        for orbit_name in orbit_names
    ]

    # ========================================================
    # Plot 1: raw pass count
    # ========================================================
    raw_pass_values = comparison_df[
        "raw_pass_count"
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.25 * len(orbit_names)), 5.6)
    )

    bars = ax.bar(
        x_positions,
        raw_pass_values,
        color=colours
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        orbit_names,
        rotation=30,
        ha="right"
    )
    ax.set_ylabel("Number of unique regional passes")
    ax.set_title("Raw regional pass count by orbit")
    ax.grid(axis="y", alpha=0.25)

    annotate_bars(
        ax,
        bars,
        integer_values=True
    )

    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir / "06_raw_pass_count"
    )

    # ========================================================
    # Plot 2: median revisit over common emitting cells
    # ========================================================
    if len(common_cell_keys) > 0:
        median_revisit_values = comparison_df[
            "median_revisit_days_common_emitting_cells"
        ].to_numpy(dtype=float)

        fig, ax = plt.subplots(
            figsize=(max(8.0, 1.25 * len(orbit_names)), 5.6)
        )

        bars = ax.bar(
            x_positions,
            median_revisit_values,
            color=colours
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            orbit_names,
            rotation=30,
            ha="right"
        )
        ax.set_ylabel("Median revisit time [days]")
        ax.set_title(
            "Median revisit time over common emitting cells\n"
            f"Common cells: {len(common_cell_keys):,}"
        )
        ax.grid(axis="y", alpha=0.25)

        annotate_bars(
            ax,
            bars,
            integer_values=False
        )

        fig.tight_layout()

        save_results_figure(
            fig,
            output_dir
            / "07_median_revisit_common_emitting_cells"
        )

    # ========================================================
    # Plot 3: total raw observation count
    # ========================================================
    raw_observation_values = comparison_df[
        "raw_observation_count"
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.25 * len(orbit_names)), 5.6)
    )

    bars = ax.bar(
        x_positions,
        raw_observation_values,
        color=colours
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        orbit_names,
        rotation=30,
        ha="right"
    )
    ax.set_ylabel("Number of raw pixel observations")
    ax.set_title(
        "Raw observations before quality and emission weighting"
    )
    ax.grid(axis="y", alpha=0.25)

    annotate_bars(
        ax,
        bars,
        integer_values=True
    )

    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir / "08_raw_observation_count"
    )

    # ========================================================
    # Plot 4: revisit-time ECDF
    # ========================================================
    if len(common_cell_keys) > 0:
        fig, ax = plt.subplots(figsize=(9.2, 6.0))

        for orbit_name in orbit_names:
            revisit_values = common_revisit_values[orbit_name]

            ecdf_x, ecdf_y = empirical_cdf(revisit_values)

            if len(ecdf_x) == 0:
                continue

            ax.step(
                ecdf_x,
                ecdf_y,
                where="post",
                linewidth=2.0,
                color=orbit_colours[orbit_name],
                label=orbit_name
            )

        ax.set_xlabel(
            "Median revisit time per common emitting cell [days]"
        )
        ax.set_ylabel("Fraction of common emitting cells")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            "ECDF of median revisit time\n"
            f"Common emitting cells: {len(common_cell_keys):,}"
        )
        ax.grid(alpha=0.25)
        ax.legend(title="Orbit")

        # A logarithmic x-axis is useful if revisit times span a
        # wide range. Remove this line if you want a linear x-axis.
        all_revisit_values = np.concatenate([
            values
            for values in common_revisit_values.values()
            if len(values) > 0
        ])

        if len(all_revisit_values) > 0:
            revisit_ratio = (
                np.max(all_revisit_values)
                / np.min(all_revisit_values)
            )

            # Use log scale only when the range exceeds one order
            # of magnitude.
            if revisit_ratio >= 10.0:
                ax.set_xscale("log")

        fig.tight_layout()

        save_results_figure(
            fig,
            output_dir
            / "09_revisit_time_ecdf_common_emitting_cells"
        )

def plot_multi_orbit_local_solar_time_histogram(output_dir):
    """
    Compare the normalized pixel-weighted local solar time
    distributions of all selected orbit cases.

    One line is plotted per orbit using common 0.5-hour bins.
    """
    if not ORBIT_RESULTS_CACHE:
        print(
            "[WARNING] No cached orbit results were available for "
            "the multi-orbit local solar time comparison."
        )
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orbit_names = list(ORBIT_RESULTS_CACHE.keys())

    colour_map = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10.2, 5.8))

    histogram_rows = []

    for i, orbit_name in enumerate(orbit_names):
        cache = ORBIT_RESULTS_CACHE[orbit_name]

        if (
            "local_time_bin_centers" not in cache
            or "local_time_pixel_fraction" not in cache
        ):
            print(
                f"[WARNING] {orbit_name}: cached local solar time "
                "histogram was unavailable. Skipping this orbit."
            )
            continue

        bin_centers = np.asarray(
            cache["local_time_bin_centers"],
            dtype=float
        )

        pixel_fraction = np.asarray(
            cache["local_time_pixel_fraction"],
            dtype=float
        )

        if len(bin_centers) != len(pixel_fraction):
            print(
                f"[WARNING] {orbit_name}: local-time bin and "
                "fraction arrays have different lengths. "
                "Skipping this orbit."
            )
            continue

        colour = colour_map(i % 10)

        ax.step(
            bin_centers,
            pixel_fraction,
            where="mid",
            linewidth=2.2,
            color=colour,
            label=orbit_name
        )

        ax.plot(
            bin_centers,
            pixel_fraction,
            linestyle="none",
            marker="o",
            markersize=3.5,
            color=colour
        )

        pixel_counts = np.asarray(
            cache.get(
                "local_time_pixel_counts",
                np.full(len(bin_centers), np.nan)
            ),
            dtype=float
        )

        for bin_center, count, fraction in zip(
            bin_centers,
            pixel_counts,
            pixel_fraction
        ):
            histogram_rows.append({
                "orbit_name": orbit_name,
                "local_solar_hour_bin_center": bin_center,
                "pixel_count": count,
                "pixel_fraction": fraction,
            })

    if not histogram_rows:
        plt.close(fig)

        print(
            "[WARNING] No valid local solar time histograms were "
            "available for the multi-orbit comparison."
        )
        return

    ax.set_xlabel("Local solar time [hour]")
    ax.set_ylabel("Fraction of raw pixel observations")
    ax.set_title(
        "Local solar time distribution by orbit"
    )

    ax.set_xlim(0.0, 24.0)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_ylim(bottom=0.0)

    ax.grid(alpha=0.25)
    ax.legend(
        title="Orbit",
        loc="best"
    )

    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir / "10_multi_orbit_local_solar_time_histogram"
    )

    histogram_comparison_df = pd.DataFrame(histogram_rows)

    histogram_comparison_df.to_csv(
        output_dir
        / "multi_orbit_local_solar_time_histogram.csv",
        index=False
    )

    histogram_comparison_df.to_parquet(
        output_dir
        / "multi_orbit_local_solar_time_histogram.parquet",
        index=False
    )

    print(
        "[MULTI-ORBIT LOCAL TIME] Saved comparison using "
        f"{len(orbit_names)} orbit case(s)."
    )


def plot_multi_orbit_diurnal_emission_alignment(output_dir):
    """
    Compare the local-solar-time observation distributions of all
    selected orbit cases with the annual mean diurnal NH3 emission
    profile.

    Left y-axis:
        Annual mean hourly NH3 emission factor.

    Right y-axis:
        Fraction of each orbit's raw observations occurring in each
        local-solar-time hour.

    Interpretation:
        Orbit curves overlapping high emission-factor hours indicate
        more favourable diurnal emission sampling.
    """
    if not ORBIT_RESULTS_CACHE:
        print(
            "[WARNING] No cached orbit results were available for "
            "the diurnal emission-alignment comparison."
        )
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orbit_names = list(ORBIT_RESULTS_CACHE.keys())
    colour_map = plt.get_cmap("tab10")

    valid_profiles = {}
    exported_rows = []

    for orbit_name in orbit_names:
        cache = ORBIT_RESULTS_CACHE[orbit_name]

        if "annual_diurnal_profile" not in cache:
            print(
                f"[WARNING] {orbit_name}: annual diurnal profile "
                "was not present in the cache. Skipping."
            )
            continue

        profile = cache["annual_diurnal_profile"].copy()

        required_columns = {
            "local_hour_bin",
            "observation_count",
            "observation_fraction",
            "mean_observed_hourfactor",
        }

        missing_columns = required_columns.difference(profile.columns)

        if missing_columns:
            print(
                f"[WARNING] {orbit_name}: diurnal profile is missing "
                f"columns {sorted(missing_columns)}. Skipping."
            )
            continue

        profile["local_hour_bin"] = pd.to_numeric(
            profile["local_hour_bin"],
            errors="coerce"
        )

        profile["observation_count"] = pd.to_numeric(
            profile["observation_count"],
            errors="coerce"
        ).fillna(0)

        profile["observation_fraction"] = pd.to_numeric(
            profile["observation_fraction"],
            errors="coerce"
        ).fillna(0.0)

        profile["mean_observed_hourfactor"] = pd.to_numeric(
            profile["mean_observed_hourfactor"],
            errors="coerce"
        )

        profile = profile.sort_values(
            "local_hour_bin"
        ).reset_index(drop=True)

        valid_profiles[orbit_name] = profile

        export_profile = profile.copy()
        export_profile.insert(0, "orbit_name", orbit_name)
        exported_rows.append(export_profile)

    if not valid_profiles:
        print(
            "[WARNING] No valid annual diurnal profiles were "
            "available. The plot was not generated."
        )
        return

    # --------------------------------------------------------
    # Construct a completely orbit-independent annual diurnal
    # NH3 emission reference directly from the source NetCDF
    # --------------------------------------------------------
    common_emission_profile = (
        build_orbit_independent_diurnal_reference()
    )

    common_emission_profile.to_csv(
        output_dir
        / "orbit_independent_annual_diurnal_emission_profile.csv",
        index=False
    )

    common_emission_profile.to_parquet(
        output_dir
        / "orbit_independent_annual_diurnal_emission_profile.parquet",
        index=False
    )

    emission_hours = common_emission_profile[
        "local_hour_center"
    ].to_numpy(dtype=float)

    emission_values = common_emission_profile[
        "annual_mean_hourfactor"
    ].to_numpy(dtype=float)

    # Close the circular daily curve by appending the first point
    # one full day later.
    emission_hours_closed = np.append(
        emission_hours,
        emission_hours[0] + 24.0
    )

    emission_values_closed = np.append(
        emission_values,
        emission_values[0]
    )

    # --------------------------------------------------------
    # Draw the combined figure
    # --------------------------------------------------------
    fig, ax_emission = plt.subplots(figsize=(11.0, 6.2))

    # Annual mean emission-factor reference curve.

    ax_emission.plot(
        emission_hours_closed,
        emission_values_closed,
        color="black",
        linewidth=2.8,
        label=(
            "Orbit-independent annual mean "
            "NH$_3$ hourly emission factor"
        ),
        zorder=5
    )

    ax_emission.fill_between(
        emission_hours_closed,
        0.0,
        emission_values_closed,
        color="0.75",
        alpha=0.25,
        zorder=1
    )

    # A factor of 1 represents the daily mean emission condition.
    ax_emission.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Daily mean emission factor = 1"
    )

    ax_emission.set_xlabel("Local solar time [hour]")
    ax_emission.set_ylabel(
        "Annual mean NH$_3$ hourly emission factor"
    )

    ax_emission.set_xlim(0.0, 24.0)
    ax_emission.set_xticks(np.arange(0, 25, 2))
    ax_emission.set_ylim(bottom=0.0)
    ax_emission.grid(alpha=0.25)

    # --------------------------------------------------------
    # Add one observation-distribution curve per orbit
    # --------------------------------------------------------
    ax_observations = ax_emission.twinx()

    for i, (orbit_name, profile) in enumerate(
        valid_profiles.items()
    ):
        colour = colour_map(i % 10)

        hours = (
            profile["local_hour_bin"].to_numpy(dtype=float)
            + 0.5
        )


        observation_fraction = profile[
            "observation_fraction"
        ].to_numpy(dtype=float)

        # Repeat hour 0 at hour 24 to show the circular daily profile.
        plot_hours = np.append(
            hours,
            hours[0] + 24.0
        )
        #plot_hours = np.append(hours, 24.0)
        plot_fraction = np.append(
            observation_fraction,
            observation_fraction[0]
        )

        ax_observations.step(
            plot_hours,
            plot_fraction,
            where="mid",
            linewidth=2.2,
            color=colour,
            label=orbit_name,
            zorder=4
        )

        ax_observations.plot(
            hours,
            observation_fraction,
            linestyle="none",
            marker="o",
            markersize=4.0,
            color=colour,
            zorder=6
        )

    ax_observations.set_ylabel(
        "Fraction of raw observations per local hour"
    )
    ax_observations.set_ylim(bottom=0.0)

    ax_emission.set_title(
        "Alignment of orbit observation times with the annual "
        "diurnal NH$_3$ emission cycle"
    )

    # Combine both axes into one legend.
    emission_handles, emission_labels = (
        ax_emission.get_legend_handles_labels()
    )

    observation_handles, observation_labels = (
        ax_observations.get_legend_handles_labels()
    )

    ax_emission.legend(
        emission_handles + observation_handles,
        emission_labels + observation_labels,
        loc="upper right",
        fontsize=9
    )

    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir
        / "11_multi_orbit_diurnal_emission_alignment"
    )

    # --------------------------------------------------------
    # Save all numerical values used in the figure
    # --------------------------------------------------------
    all_profiles_df = pd.concat(
        exported_rows,
        ignore_index=True
    )

    all_profiles_df.to_csv(
        output_dir
        / "multi_orbit_diurnal_emission_alignment.csv",
        index=False
    )

    all_profiles_df.to_parquet(
        output_dir
        / "multi_orbit_diurnal_emission_alignment.parquet",
        index=False
    )

    common_emission_profile.to_csv(
        output_dir
        / "annual_mean_diurnal_emission_profile.csv",
        index=False
    )

    print(
        "[MULTI-ORBIT DIURNAL ALIGNMENT] Saved comparison for "
        f"{len(valid_profiles)} orbit case(s)."
    )

def plot_signal_effective_samples_per_cell_comparison(output_dir):
    """
    Compare cell-level signal-effective observations per pass across
    all orbit cases using the same common emitting-cell population.

    Two ECDF panels are generated:
        1. Mean signal-effective observations per pass per cell.
        2. Median signal-effective observations per pass per cell.

    Only cells that:
        - have positive NH3 emissions;
        - have finite, positive signal-effective values; and
        - are available for every orbit
    are retained.

    This prevents differences in the plotted distributions from being
    caused by different cell populations.
    """

    if not ORBIT_RESULTS_CACHE:
        print(
            "[WARNING] No cached orbit results were available for the "
            "cell-level signal-effective comparison."
        )
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orbit_names = list(ORBIT_RESULTS_CACHE.keys())

    metric_columns = {
        "Mean per pass": "N_signal_eff_per_pass_mean",
        "Median per pass": "N_signal_eff_per_pass_median",
    }

    # ============================================================
    # Extract valid emitting cells for every orbit
    # ============================================================

    signal_by_orbit = {}
    common_cell_keys = None

    for orbit_name in orbit_names:

        cache = ORBIT_RESULTS_CACHE[orbit_name]

        observability_df = (
            cache["observability_full"]
            .copy()
        )

        required_columns = {
            "iy",
            "ix",
            "nh3_kg_yr",
            "N_signal_eff_per_pass_mean",
            "N_signal_eff_per_pass_median",
        }

        missing_columns = required_columns.difference(
            observability_df.columns
        )

        if missing_columns:
            raise ValueError(
                f"{orbit_name}: columns required for the cell-level "
                f"signal-effective comparison are missing: "
                f"{sorted(missing_columns)}"
            )

        signal_df = observability_df[
            [
                "iy",
                "ix",
                "nh3_kg_yr",
                "N_signal_eff_per_pass_mean",
                "N_signal_eff_per_pass_median",
            ]
        ].copy()

        # Convert relevant columns safely to numeric values.
        for column_name in [
            "nh3_kg_yr",
            "N_signal_eff_per_pass_mean",
            "N_signal_eff_per_pass_median",
        ]:
            signal_df[column_name] = pd.to_numeric(
                signal_df[column_name],
                errors="coerce"
            )

        # Retain emitting cells with valid positive signal-effective
        # quantities for both comparison metrics.
        valid_mask = (
            (signal_df["nh3_kg_yr"] > 0.0)
            & np.isfinite(
                signal_df["N_signal_eff_per_pass_mean"]
            )
            & (
                signal_df["N_signal_eff_per_pass_mean"]
                >= 0.0
            )
            & np.isfinite(
                signal_df["N_signal_eff_per_pass_median"]
            )
            & (
                signal_df["N_signal_eff_per_pass_median"]
                >= 0.0
            )
        )

        signal_df = (
            signal_df.loc[valid_mask]
            .drop_duplicates(
                subset=["iy", "ix"],
                keep="first"
            )
            .copy()
        )

        signal_by_orbit[orbit_name] = signal_df

        current_keys = set(
            zip(
                signal_df["iy"].astype(int),
                signal_df["ix"].astype(int),
            )
        )

        if common_cell_keys is None:
            common_cell_keys = current_keys
        else:
            common_cell_keys = (
                common_cell_keys.intersection(current_keys)
            )

    if not common_cell_keys:
        print(
            "[WARNING] No common emitting cells with finite positive "
            "signal-effective samples were found."
        )
        return

    common_cells_df = pd.DataFrame(
        sorted(common_cell_keys),
        columns=["iy", "ix"]
    )

    common_cells_df["iy"] = (
        common_cells_df["iy"].astype(int)
    )
    common_cells_df["ix"] = (
        common_cells_df["ix"].astype(int)
    )

    number_common_cells = len(common_cells_df)

    print("\n" + "=" * 80)
    print("CELL-LEVEL SIGNAL-EFFECTIVE COMPARISON")
    print("=" * 80)
    print(
        f"Common emitting cells used: "
        f"{number_common_cells:,}"
    )

    # ============================================================
    # Create ECDF figure
    # ============================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.0),
        sharey=True
    )

    colour_map = plt.get_cmap("tab10")

    exported_rows = []
    summary_rows = []

    for orbit_index, orbit_name in enumerate(orbit_names):

        orbit_signal_df = common_cells_df.merge(
            signal_by_orbit[orbit_name],
            on=["iy", "ix"],
            how="inner",
            validate="one_to_one"
        )

        orbit_colour = colour_map(orbit_index % 10)

        for axis, (metric_label, metric_column) in zip(
            axes,
            metric_columns.items()
        ):

            values = pd.to_numeric(
                orbit_signal_df[metric_column],
                errors="coerce"
            ).to_numpy(dtype=float)

            values = values[
                np.isfinite(values)
                & (values > 0.0)
            ]

            if values.size == 0:
                continue

            values_sorted = np.sort(values)

            ecdf_fraction = (
                np.arange(1, values_sorted.size + 1)
                / values_sorted.size
            )

            axis.step(
                values_sorted,
                ecdf_fraction,
                where="post",
                linewidth=2.2,
                color=orbit_colour,
                label=orbit_name
            )

            # Save all plotted cell-level values.
            exported_rows.extend(
                {
                    "orbit_name": orbit_name,
                    "iy": int(row["iy"]),
                    "ix": int(row["ix"]),
                    "metric": metric_column,
                    "value": float(row[metric_column]),
                }
                for _, row in orbit_signal_df.iterrows()
                if (
                    np.isfinite(row[metric_column])
                    and row[metric_column] > 0.0
                )
            )

            summary_rows.append(
                {
                    "orbit_name": orbit_name,
                    "metric": metric_column,
                    "common_cell_count": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                    "p90": float(np.percentile(values, 90)),
                }
            )

        print(f"\n{orbit_name}")

        for metric_label, metric_column in metric_columns.items():

            values = pd.to_numeric(
                orbit_signal_df[metric_column],
                errors="coerce"
            ).to_numpy(dtype=float)

            values = values[
                np.isfinite(values)
                & (values > 0.0)
            ]

            if values.size == 0:
                continue

            print(
                f"  {metric_label}: "
                f"median={np.median(values):.6f}, "
                f"mean={np.mean(values):.6f}, "
                f"p10={np.percentile(values, 10):.6f}, "
                f"p90={np.percentile(values, 90):.6f}"
            )

    # ============================================================
    # Format ECDF panels
    # ============================================================

    panel_titles = {
        "Mean per pass": (
            "Mean signal-effective observations\n"
            "per pass and emitting cell"
        ),
        "Median per pass": (
            "Median signal-effective observations\n"
            "per pass and emitting cell"
        ),
    }

    for axis, metric_label in zip(
        axes,
        metric_columns.keys()
    ):
        axis.set_title(panel_titles[metric_label])

        axis.set_xlabel(
            r"Signal-effective observations per pass, "
            r"$N_{\mathrm{sig,eff/pass}}$"
        )

        axis.set_ylabel(
            "Fraction of common emitting cells"
        )

        axis.set_ylim(0.0, 1.0)

        # A logarithmic x-axis is important because these values may
        # span several orders of magnitude.
        axis.set_xscale("log")

        axis.grid(
            alpha=0.25,
            which="both"
        )

    axes[0].legend(
        title="Orbit",
        loc="best"
    )

    fig.suptitle(
        "ECDF of cell-level signal-effective observations per pass\n"
        f"Common emitting cells: {number_common_cells:,}",
        fontsize=15
    )

    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir
        / "14_signal_effective_samples_per_pass_ecdf_common_cells"
    )

    # ============================================================
    # Export plotted data and summary
    # ============================================================

    exported_df = pd.DataFrame(exported_rows)

    exported_df.to_csv(
        output_dir
        / "signal_effective_samples_per_pass_common_cells.csv",
        index=False
    )

    exported_df.to_parquet(
        output_dir
        / "signal_effective_samples_per_pass_common_cells.parquet",
        index=False
    )

    summary_df = (
        pd.DataFrame(summary_rows)
        .drop_duplicates(
            subset=["orbit_name", "metric"],
            keep="first"
        )
    )

    summary_df.to_csv(
        output_dir
        / "signal_effective_samples_per_pass_summary.csv",
        index=False
    )

    summary_df.to_parquet(
        output_dir
        / "signal_effective_samples_per_pass_summary.parquet",
        index=False
    )

    print("\nSignal-effective summary:")
    print(
        summary_df.to_string(index=False)
    )

    print(
        "\n[CELL-LEVEL SIGNAL-EFFECTIVE COMPARISON] "
        f"Saved ECDF comparison for {len(orbit_names)} orbit(s)."
    )


def plot_multi_orbit_effective_observation_comparison(output_dir):
    """
    Generate two grouped multi-orbit bar charts.

    Plot 1
    ------
    N_raw:
        Total number of raw pixel observations.

    N_eff:
        Total quality-effective observation count after applying:
            SZA weight x TC weight x cloud weight.

    N_sig_eff:
        Total signal-effective observation count after applying:
            quality weight x
            (hourfactor x seasonal_factor)^2.

    Plot 2
    ------
    Quality retention:
        N_eff / N_raw.

    Signal-effective ratio:
        N_sig_eff / N_raw.
    """
    if not ORBIT_RESULTS_CACHE:
        print(
            "[WARNING] No cached orbit results were available for "
            "the effective-observation comparison."
        )
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orbit_names = list(ORBIT_RESULTS_CACHE.keys())

    comparison_rows = []

    for orbit_name in orbit_names:
        cache = ORBIT_RESULTS_CACHE[orbit_name]

        required_cache_keys = {
            "N_raw_total",
            "N_eff_total",
            "N_sig_eff_total",
            "quality_retention_total",
            "signal_effective_ratio_total",
        }

        missing_keys = required_cache_keys.difference(
            cache.keys()
        )

        if missing_keys:
            print(
                f"[WARNING] {orbit_name}: missing cached effective-"
                f"observation metrics: {sorted(missing_keys)}. "
                "Skipping this orbit."
            )
            continue

        comparison_rows.append({
            "orbit_name": orbit_name,
            "N_raw": float(cache["N_raw_total"]),
            "N_eff": float(cache["N_eff_total"]),
            "N_sig_eff": float(cache["N_sig_eff_total"]),
            "N_eff_over_N_raw": float(
                cache["quality_retention_total"]
            ),
            "N_sig_eff_over_N_raw": float(
                cache["signal_effective_ratio_total"]
            ),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    if comparison_df.empty:
        print(
            "[WARNING] No valid orbit metrics were available for "
            "the effective-observation grouped bar charts."
        )
        return

    # Save the exact numerical values plotted.
    comparison_df.to_csv(
        output_dir
        / "multi_orbit_effective_observation_comparison.csv",
        index=False
    )

    comparison_df.to_parquet(
        output_dir
        / "multi_orbit_effective_observation_comparison.parquet",
        index=False
    )

    print("\n" + "=" * 80)
    print("MULTI-ORBIT EFFECTIVE OBSERVATION COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    orbit_names = comparison_df[
        "orbit_name"
    ].tolist()

    x = np.arange(len(orbit_names), dtype=float)

    # ========================================================
    # Plot 1: N_raw, N_eff and N_sig_eff
    # ========================================================
    bar_width = 0.24

    fig, ax = plt.subplots(
        figsize=(
            max(9.5, 1.65 * len(orbit_names)),
            6.2
        )
    )

    raw_bars = ax.bar(
        x - bar_width,
        comparison_df["N_raw"],
        width=bar_width,
        color="tab:blue",
        label=r"$N_{\mathrm{raw}}$"
    )

    effective_bars = ax.bar(
        x,
        comparison_df["N_eff"],
        width=bar_width,
        color="tab:orange",
        label=r"$N_{\mathrm{eff}}$"
    )

    signal_effective_bars = ax.bar(
        x + bar_width,
        comparison_df["N_sig_eff"],
        width=bar_width,
        color="tab:green",
        label=r"$N_{\mathrm{sig,eff}}$"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        orbit_names,
        rotation=30,
        ha="right"
    )

    ax.set_ylabel("Effective number of pixel observations")

    ax.set_title(
        "Raw, retrieval-quality and signal-effective "
        "observations by orbit"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    # Apply scientific notation if totals are large.
    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )

    # Add values above every bar.
    for bars in [
        raw_bars,
        effective_bars,
        signal_effective_bars,
    ]:
        for bar in bars:
            value = bar.get_height()

            if not np.isfinite(value):
                continue

            ax.annotate(
                f"{value:,.0f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2.0,
                    value
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )

    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir
        / "12_multi_orbit_raw_effective_signal_counts"
    )

    # ========================================================
    # Plot 2: N_eff/N_raw and N_sig_eff/N_raw
    # ========================================================
    ratio_bar_width = 0.34

    fig, ax = plt.subplots(
        figsize=(
            max(9.5, 1.55 * len(orbit_names)),
            6.0
        )
    )

    quality_bars = ax.bar(
        x - ratio_bar_width / 2.0,
        comparison_df["N_eff_over_N_raw"],
        width=ratio_bar_width,
        color="tab:orange",
        label=r"$N_{\mathrm{eff}}/N_{\mathrm{raw}}$"
    )

    signal_ratio_bars = ax.bar(
        x + ratio_bar_width / 2.0,
        comparison_df["N_sig_eff_over_N_raw"],
        width=ratio_bar_width,
        color="tab:green",
        label=r"$N_{\mathrm{sig,eff}}/N_{\mathrm{raw}}$"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        orbit_names,
        rotation=30,
        ha="right"
    )

    ax.set_ylabel("Ratio relative to raw observations")

    ax.set_title(
        "Quality retention and signal-effective ratio by orbit"
    )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        label="Raw-observation reference"
    )

    ax.set_ylim(
        bottom=0.0
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    for bars in [
        quality_bars,
        signal_ratio_bars,
    ]:
        for bar in bars:
            value = bar.get_height()

            if not np.isfinite(value):
                continue

            ax.annotate(
                f"{value:.3f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2.0,
                    value
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )

    fig.tight_layout()

    save_results_figure(
        fig,
        output_dir
        / "13_multi_orbit_quality_and_signal_retention"
    )

    print(
        "[MULTI-ORBIT EFFECTIVE OBSERVATIONS] "
        f"Saved grouped comparisons for "
        f"{len(comparison_df)} orbit case(s)."
    )

def print_final_data_completeness_report(output_dir):
    """
    Print and save environmental-data completeness diagnostics
    for every completed orbit case.
    """

    if not ORBIT_RESULTS_CACHE:
        print(
            "[WARNING] No cached orbit results were available for "
            "the data-completeness report."
        )
        return

    completeness_rows = []

    for orbit_name, cache in ORBIT_RESULTS_CACHE.items():
        diagnostics = cache.get("data_completeness")

        if diagnostics is None:
            print(
                f"[WARNING] {orbit_name}: data-completeness "
                "diagnostics were not found in the cache."
            )
            continue

        completeness_rows.append(diagnostics)

    completeness_df = pd.DataFrame(completeness_rows)

    if completeness_df.empty:
        print(
            "[WARNING] No data-completeness diagnostics were "
            "available."
        )
        return

    # Sort in the same order that the orbit cases were processed.
    orbit_order = list(ORBIT_RESULTS_CACHE.keys())

    completeness_df["orbit_name"] = pd.Categorical(
        completeness_df["orbit_name"],
        categories=orbit_order,
        ordered=True
    )

    completeness_df = (
        completeness_df
        .sort_values("orbit_name")
        .reset_index(drop=True)
    )

    completeness_df["orbit_name"] = (
        completeness_df["orbit_name"].astype(str)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completeness_df.to_csv(
        output_dir / "orbit_data_completeness_summary.csv",
        index=False
    )

    completeness_df.to_parquet(
        output_dir / "orbit_data_completeness_summary.parquet",
        index=False
    )

    print("\n" + "=" * 90)
    print("FINAL ORBIT DATA-COMPLETENESS REPORT")
    print("=" * 90)

    for _, row in completeness_df.iterrows():
        print("\n" + "-" * 90)
        print(f"Orbit: {row['orbit_name']}")
        print("-" * 90)

        print(
            "Total LE analysis-grid cells: "
            f"{int(row['total_grid_cells']):,}"
        )

        print(
            "Positive-emission cells: "
            f"{int(row['positive_emission_cells']):,}"
        )

        print(
            "Non-emitting cells: "
            f"{int(row['non_emitting_cells']):,}"
        )

        print(
            "Fraction of non-emitting cells: "
            f"{row['fraction_non_emitting_cells']:.6f} "
            f"({100.0 * row['fraction_non_emitting_cells']:.3f}%)"
        )

        print(
            "Total observed pixel rows: "
            f"{int(row['total_observed_pixel_rows']):,}"
        )

        print(
            "Fraction missing TC: "
            f"{row['fraction_missing_tc']:.6f} "
            f"({100.0 * row['fraction_missing_tc']:.3f}%)"
        )

        print(
            "Fraction missing TCC: "
            f"{row['fraction_missing_tcc']:.6f} "
            f"({100.0 * row['fraction_missing_tcc']:.3f}%)"
        )

        print(
            "Fraction missing diurnal factor: "
            f"{row['fraction_missing_diurnal_factor']:.6f} "
            f"({100.0 * row['fraction_missing_diurnal_factor']:.3f}%)"
        )

        print(
            "Fraction missing seasonal factor: "
            f"{row['fraction_missing_seasonal_factor']:.6f} "
            f"({100.0 * row['fraction_missing_seasonal_factor']:.3f}%)"
        )

    print("\n" + "=" * 90)
    print(
        "Saved completeness summary to:",
        output_dir / "orbit_data_completeness_summary.csv"
    )
    print("=" * 90)


def generate_multi_orbit_results(output_root="./outputs"):
    """
    Generate final comparison figures after every orbit has completed.
    """
    if not ORBIT_RESULTS_CACHE:
        print(
            "[WARNING] No cached orbit results were available for "
            "multi-orbit plotting."
        )
        return

    comparison_dir = (
        Path(output_root)
        / "comparison"
        / "results_figures"
    )

    comparison_dir.mkdir(parents=True, exist_ok=True)

    combined_ablation = pd.concat(
        [
            cache["ablation_summary"]
            for cache in ORBIT_RESULTS_CACHE.values()
        ],
        ignore_index=True
    )

    combined_ablation.to_csv(
        comparison_dir / "combined_ablation_summary.csv",
        index=False
    )

    combined_ablation.to_parquet(
        comparison_dir / "combined_ablation_summary.parquet",
        index=False
    )

    plot_multi_orbit_ablation_heatmap(
        combined_ablation=combined_ablation,
        output_dir=comparison_dir
    )

    plot_final_detectability_comparison(
        output_dir=comparison_dir
    )

    plot_orbit_condition_comparison(
        output_dir=comparison_dir
    )


    # New targeted raw-sampling and revisit comparison figures
    plot_raw_sampling_and_revisit_comparison(
        output_dir=comparison_dir
    )

    # Multi-orbit local solar time distribution
    plot_multi_orbit_local_solar_time_histogram(
        output_dir=comparison_dir
    )


    # Compare orbit observation times with the annual diurnal
    # NH3 emission cycle.
    plot_multi_orbit_diurnal_emission_alignment(
        output_dir=comparison_dir
    )

    # Grouped comparison of raw, quality-effective and
    # signal-effective observation totals.
    plot_multi_orbit_effective_observation_comparison(
        output_dir=comparison_dir
    )

        # Cell-level distribution of mean and median signal-effective
    # observations per pass over the same common emitting cells.
    plot_signal_effective_samples_per_cell_comparison(
        output_dir=comparison_dir
    )
    # Final data-completeness report for all completed orbit cases.
    print_final_data_completeness_report(
        output_dir=comparison_dir
    )
    print(
        f"Final multi-orbit Results figures saved to: "
        f"{comparison_dir}"
    )


SAT = "SAT-A"
YEAR = 2019
SWATH_WIDTH_KM = 512.0 # 320.0 # In kilometers
SWATH_HALF_WIDTH_M = (SWATH_WIDTH_KM * 1000.0) / 2.0

# NITROSAT = 160 for 500m res at 824km altitude, Earthinspect = 256 for 2km res at 824km
N_ACROSS = 256 #256   # number of cross-track pixels, choose based on desired sampling density,

# Active instrument geometry:
# 256 across-track pixels with a nominal 2 km nadir spacing produce an
# approximate 512 km nadir-width scan. The actual surface swath and edge-pixel
# dimensions are calculated from altitude and spherical-Earth viewing geometry.


# Along-track IFOV: set equal to across-track for "square" pixels at nadir
SQUARE_AT_NADIR = True


USE_DYNAMIC_ALTITUDE = True
# If True, use the propagated Tudat altitude separately for each scanline.
# If False, use the fixed fallback altitude defined in the footprint function.



# INSTRUMENT GEOMETRY MODE
# -------------------------
# Mode A: "fixed_swath" -> SWATH_WIDTH_KM fixed, nadir pixel spacing implied by N_ACROSS
# Mode B: "fixed_nadir" -> NADIR_PIXEL_M fixed, swath width becomes N_ACROSS * NADIR_PIXEL_M (variable)
GEOM_MODE = "fixed_nadir"   # "fixed_swath" or "fixed_nadir"

#NADIR_PIXEL_M = 500.0 #2000.0 # only used if GEOM_MODE="fixed_nadir"
NADIR_PIXEL_M = 2000.0

# Use spherical-Earth pixel geometry instead of the flat-Earth approximation.
USE_EARTH_CURVATURE = True  # spherical approximation


# LOTOS-EUROS NL-2x2 GRID
# -------------------------
# For filtering down to only footprints/scanlines that fall within the model grid boundaries
GRID_WEST  = 3.15
GRID_SOUTH = 50.65
GRID_DLON  =  0.025
GRID_DLAT  =  0.0125
GRID_NX    = 174
GRID_NY    = 244


GRID_EAST  = GRID_WEST  + GRID_NX * GRID_DLON   # 7.50
GRID_NORTH = GRID_SOUTH + GRID_NY * GRID_DLAT   # 53.70


# -------------------------
# COARSE-to-FINE settings (Tudat windowing)
# -------------------------
'''
The fine propagation output cadence is also treated as the instrument
scanline cadence. One synthetic cross-track scanline is generated at every
fine-track epoch. Consequently, changing this value changes both temporal
sampling density and the total number of synthetic pixel observations.
'''


COARSE_STEP_SECONDS = 10.0      # matches your coarse Tudat RK4 step
FINE_STEP_SECONDS   = 0.075     # match supervisor scanline cadence
WINDOW_PAD_SECONDS  = 600       # +/- 10 min padding
WINDOW_MERGE_GAP_SECONDS = 1800 # merge coarse hits within 30 min





orbit_cases = [


    {
        "name": "orbit_nominal_SSO_LTAN_1530",
        "t0_utc": datetime(2019, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        "t1_utc": datetime(2019, 12, 30, 0, 0, 0, tzinfo=timezone.utc),
        "semi_major_axis": 7000000.000000000,
        "eccentricity": 0.001000000000,
        "inclination": np.deg2rad(97.873928816589),
        "argument_of_periapsis": np.deg2rad(0.000000000000),
        "longitude_of_ascending_node": np.deg2rad(333.794034217372),
        "true_anomaly": np.deg2rad(0.000000000000),
    },

    # {
    #     "name": "orbit_near_SSO_96_inc_0_RAAN_fullyear",
    #     "t0_utc": datetime(2019, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
    #     "t1_utc": datetime(2019, 12, 30, 0, 0, 0, tzinfo=timezone.utc),
    #     "semi_major_axis": 7.000e+06, #meters
    #     "eccentricity": 0.001, #unitless
    #     "inclination": np.deg2rad(96.0), #radians
    #     "argument_of_periapsis": 0.0, #radians
    #     "longitude_of_ascending_node": np.deg2rad(0.0), #radians
    #     "true_anomaly": 0.0, #radians
    # },

    # {
    #     "name": "orbit_nominal_fixed_inclination_rgt_233rev_16day",
    #     "t0_utc": datetime(2019, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
    #     "t1_utc": datetime(2019, 12, 30, 0, 0, 0, tzinfo=timezone.utc),
    #     "semi_major_axis": 7024520.387705519,
    #     "eccentricity": 0.001000000000,
    #     "inclination": np.deg2rad(60.000000000000),
    #     "argument_of_periapsis": np.deg2rad(0.000000000000),
    #     "longitude_of_ascending_node": np.deg2rad(160),
    #     "true_anomaly": np.deg2rad(0.000000000000),
    # },


    # {
    #     "name": "orbit_rgt_233rev_16day_RAAN_111",
    #     "t0_utc": datetime(
    #         2019, 1, 1, 9, 0, 0,
    #         tzinfo=timezone.utc
    #     ),
    #     "t1_utc": datetime(
    #         2019, 12, 30, 0, 0, 0,
    #         tzinfo=timezone.utc
    #     ),
    #     "semi_major_axis": 7024520.387705519,
    #     "eccentricity": 0.001,
    #     "inclination": np.deg2rad(60.0),
    #     "argument_of_periapsis": np.deg2rad(0.0),
    #     "longitude_of_ascending_node": np.deg2rad(111.0),
    #     "true_anomaly": np.deg2rad(0.0),
    # },


    # {
    #     "name": "orbit_nominal_sso_rgt_233rev_16day",
    #     "t0_utc": datetime(2019, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
    #     "t1_utc": datetime(2019, 12, 30, 0, 0, 0, tzinfo=timezone.utc),
    #     "semi_major_axis": 7077722.190292655,
    #     "eccentricity": 0.001000000000,
    #     "inclination": np.deg2rad(98.186276078101),
    #     "argument_of_periapsis": np.deg2rad(0.000000000000),
    #     "longitude_of_ascending_node": np.deg2rad(258.794034217372),
    #     "true_anomaly": np.deg2rad(0.000000000000),
    # },


    # {
    #     "name": "orbit_sso_rgt_233/16_10.5LTDN_210TA",
    #     "t0_utc": datetime(2019, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
    #     "t1_utc": datetime(2019, 12, 30, 0, 0, 0, tzinfo=timezone.utc),
    #     "semi_major_axis": 7077722.190286835,
    #     "eccentricity": 0.001000000000,
    #     "inclination": np.deg2rad(98.186276078077),
    #     "argument_of_periapsis": np.deg2rad(0.000000000000),
    #     "longitude_of_ascending_node": np.deg2rad(78.794034217372),
    #     "true_anomaly": np.deg2rad(210.000000000000),
    # },


    # {
    #     "name": "orbit_Molniya_NL_195",
    #     "t0_utc": datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    #     "t1_utc": datetime(2019, 12, 30, 0, 0, 0, tzinfo=timezone.utc),
    #     "semi_major_axis": 26562e3,                 # meters. This value gives a time period of 12 hours
    #     "eccentricity": 0.72,
    #     "inclination": np.deg2rad(63.4),
    #     "argument_of_periapsis": np.deg2rad(270.0), # northern apogee
    #     "longitude_of_ascending_node": np.deg2rad(195.0),  # This parameter is what needs to be tuned for it to cross over the required area.
    #     "true_anomaly": np.deg2rad(180.0),          # start near apogee
    # },


    # {
    #     "name": "orbit_Highly_Elliptical_fullyear",
    #     "t0_utc": datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    #     "t1_utc": datetime(2019, 12, 30, 0, 0, 0, tzinfo=timezone.utc),
    #     "semi_major_axis": 1.3e7,                 # meters. This value gives a time period of 12 hours
    #     "eccentricity": 0.4,
    #     "inclination": np.deg2rad(63.4),
    #     "argument_of_periapsis": np.deg2rad(270.0), # northern apogee
    #     "longitude_of_ascending_node": np.deg2rad(0.0),  # Need to tune.
    #     "true_anomaly": np.deg2rad(180.0),          # start near apogee
    # },


    # {
    #     "name": "orbit_SSO_85_fullyear",
    #     "t0_utc": datetime(2019, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
    #     "t1_utc": datetime(2019, 12, 30, 0, 0, 0, tzinfo=timezone.utc),
    #     "semi_major_axis": 7.000e+06, #meters
    #     "eccentricity": 0.001, #unitless
    #     "inclination": np.deg2rad(97.5), #radians
    #     "argument_of_periapsis": 0.0, #radians
    #     "longitude_of_ascending_node": np.deg2rad(85.0), #radians
    #     "true_anomaly": 0.0, #radians
    # },


    # {
    #     "name": "orbit_NonSSO_60_inc_0_raan_fullyear",
    #     "t0_utc": datetime(2019, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
    #     "t1_utc": datetime(2019, 12, 30, 0, 0, 0, tzinfo=timezone.utc),
    #     "semi_major_axis": 7.0000000e+06,
    #     "eccentricity": 1.0e-03,
    #     "inclination": np.deg2rad(60.0),
    #     "argument_of_periapsis": 0.0,
    #     "longitude_of_ascending_node": np.deg2rad(0.0),
    #     "true_anomaly": 0.0,
    # },


    # Add more candidate orbits here
]


# To incorporate metrix for the comparison when SZA was added:

def summarize_orbit_case(orbit_name, sampling_summary, observability_df):
    """
    Build compact orbit-level summary metrics for comparison across cases.

    Includes:
    - raw sampling
    - SZA-weighted sampling
    - SZA statistics
    - finite/infinite detectability statistics
    - emission-weighted detectability
    """

    s = sampling_summary.copy()
    df = observability_df.copy()

    # Cells with nonzero emissions in merged observability dataframe
    emission_cells = df[df["nh3_kg_yr"] > 0].copy()

    # Finite detectability among emitting cells
    finite_emission_cells = emission_cells[
        np.isfinite(emission_cells["time_to_detect_days"])
    ].copy()

    n_emission = len(emission_cells)
    n_finite = len(finite_emission_cells)

    frac_finite = n_finite / n_emission if n_emission > 0 else np.nan
    frac_inf = 1.0 - frac_finite if n_emission > 0 else np.nan

    # SZA reduction ratio for all sampled cells
    sza_ratio_all = np.where(
        s["N_obs"].values > 0,
        s["N_obs_eff"].values / s["N_obs"].values,
        np.nan
    )

    # SZA reduction ratio for emitting merged cells
    if "N_obs_eff" in emission_cells.columns:
        sza_ratio_emis = np.where(
            emission_cells["N_obs"].values > 0,
            emission_cells["N_obs_eff"].values / emission_cells["N_obs"].values,
            np.nan
        )
    else:
        sza_ratio_emis = np.array([np.nan])

    def safe_median(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) > 0 else np.nan

    def safe_mean(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if len(x) > 0 else np.nan

    def safe_percentile(x, p):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.percentile(x, p)) if len(x) > 0 else np.nan

    # Detectability stats only for finite emitting cells
    if n_finite > 0:
        detect_days = finite_emission_cells["time_to_detect_days"].values

        weights = finite_emission_cells["nh3_kg_yr"].values.astype(float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

        weighted_mean_detect_days = float(np.sum(detect_days * weights))

        median_detect_days = safe_median(detect_days)
        mean_detect_days = safe_mean(detect_days)
        p10_detect_days = safe_percentile(detect_days, 10)
        p90_detect_days = safe_percentile(detect_days, 90)

        frac_detect_30d = float((detect_days <= 30).mean())
        frac_detect_90d = float((detect_days <= 90).mean())
        frac_detect_365d = float((detect_days <= 365).mean())
    else:
        weighted_mean_detect_days = np.nan
        median_detect_days = np.nan
        mean_detect_days = np.nan
        p10_detect_days = np.nan
        p90_detect_days = np.nan
        frac_detect_30d = np.nan
        frac_detect_90d = np.nan
        frac_detect_365d = np.nan
        # ------------------------------------------------------------
    # Paper-style detectability stats
    # rawN  = paper formula using raw pixel samples per pass
    # effN  = paper formula using SZA-weighted pixel samples per pass
    # ------------------------------------------------------------
    def detect_stats_for_column(emission_cells_in, colname):
        if colname not in emission_cells_in.columns:
            return {
                "median": np.nan,
                "p10": np.nan,
                "p90": np.nan,
                "weighted_mean": np.nan,
                "frac_30d": np.nan,
                "frac_90d": np.nan,
                "frac_365d": np.nan,
            }

        finite_cells = emission_cells_in[
            np.isfinite(emission_cells_in[colname])
        ].copy()

        if len(finite_cells) == 0:
            return {
                "median": np.nan,
                "p10": np.nan,
                "p90": np.nan,
                "weighted_mean": np.nan,
                "frac_30d": np.nan,
                "frac_90d": np.nan,
                "frac_365d": np.nan,
            }

        vals = finite_cells[colname].values

        weights_local = finite_cells["nh3_kg_yr"].values.astype(float)
        weights_local = (
            weights_local / weights_local.sum()
            if weights_local.sum() > 0
            else np.ones_like(weights_local) / len(weights_local)
        )

        return {
            "median": safe_median(vals),
            "p10": safe_percentile(vals, 10),
            "p90": safe_percentile(vals, 90),
            "weighted_mean": float(np.sum(vals * weights_local)),
            "frac_30d": float((vals <= 30).mean()),
            "frac_90d": float((vals <= 90).mean()),
            "frac_365d": float((vals <= 365).mean()),
        }


    paper_raw_stats = detect_stats_for_column(
        emission_cells,
        "time_to_detect_days_paper_rawN"
    )

    paper_eff_stats = detect_stats_for_column(
        emission_cells,
        "time_to_detect_days_paper_effN"
    )

    paper_signal_eff_stats = detect_stats_for_column(
        emission_cells,
        "time_to_detect_days_paper_signalEffN"
    )
    #------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------
    if "N_obs_signal_eff" in s.columns:
        signal_ratio_all = np.where(
            s["N_obs_eff"].values > 0,
            s["N_obs_signal_eff"].values / s["N_obs_eff"].values,
            np.nan
        )
    else:
        signal_ratio_all = np.array([np.nan])

    if "N_obs_signal_eff" in emission_cells.columns:
        signal_ratio_emis = np.where(
            emission_cells["N_obs_eff"].values > 0,
            emission_cells["N_obs_signal_eff"].values / emission_cells["N_obs_eff"].values,
            np.nan
        )
    else:
        signal_ratio_emis = np.array([np.nan])


    return {
        "orbit_name": orbit_name,

        # Sampling coverage
        "sampled_cells": int(len(s)),
        "emission_cells_merged": int(n_emission),
        "finite_detectable_emission_cells": int(n_finite),
        "frac_finite_emission_cells": float(frac_finite) if np.isfinite(frac_finite) else np.nan,
        "frac_infinite_emission_cells": float(frac_inf) if np.isfinite(frac_inf) else np.nan,

        # Raw observation count
        "median_N_obs_all": safe_median(s["N_obs"]),
        #"mean_N_obs_all": safe_mean(s["N_obs"]),

        # SZA-weighted observation count
        "median_N_obs_eff_all": safe_median(s["N_obs_eff"]),
        #"mean_N_obs_eff_all": safe_mean(s["N_obs_eff"]),

        # SZA reduction factor
        "median_SZA_reduction_all": safe_median(sza_ratio_all),
        #"mean_SZA_reduction_all": safe_mean(sza_ratio_all),

        # SZA stats
        "median_sza_min_deg_all": safe_median(s["sza_min_deg"]),
        #"mean_sza_min_deg_all": safe_mean(s["sza_min_deg"]),
        "median_sza_median_deg_all": safe_median(s["sza_median_deg"]),

        # Thermal contrast diagnostics
        "median_tc_K_all": safe_median(s["tc_median_K"]) if "tc_median_K" in s.columns else np.nan,
        "median_w_tc_all": safe_median(s["w_tc_mean"]) if "w_tc_mean" in s.columns else np.nan,
        "median_w_quality_all": safe_median(s["w_quality_mean"]) if "w_quality_mean" in s.columns else np.nan,

        # Cloud diagnostics
        "median_tcc_all": safe_median(s["tcc_median"]) if "tcc_median" in s.columns else np.nan,
        "median_w_cloud_all": safe_median(s["w_cloud_mean"]) if "w_cloud_mean" in s.columns else np.nan,

        "median_tcc_emission_cells": safe_median(emission_cells["tcc_median"]) if "tcc_median" in emission_cells.columns and n_emission > 0 else np.nan,
        "median_w_cloud_emission_cells": safe_median(emission_cells["w_cloud_mean"]) if "w_cloud_mean" in emission_cells.columns and n_emission > 0 else np.nan,
        # Revisit
        "median_revisit_days_all": safe_median(s["dt_median_s"] / 86400.0),
        #"mean_revisit_days_all": safe_mean(s["dt_median_s"] / 86400.0),

        # Emission-cell-specific sampling
        "median_N_obs_emission_cells": safe_median(emission_cells["N_obs"]) if n_emission > 0 else np.nan,
        "median_N_obs_eff_emission_cells": safe_median(emission_cells["N_obs_eff"]) if n_emission > 0 else np.nan,
        "median_SZA_reduction_emission_cells": safe_median(sza_ratio_emis),
        "median_revisit_days_emission_cells": safe_median(emission_cells["dt_median_days"]) if n_emission > 0 else np.nan,

        # Current observation-rate detectability
        "median_detect_days_finite": median_detect_days,
        #"mean_detect_days_finite": mean_detect_days,
        "p10_detect_days_finite": p10_detect_days,
        "p90_detect_days_finite": p90_detect_days,
        "weighted_mean_detect_days": weighted_mean_detect_days,

        # Detection thresholds for current method
        "frac_detect_30d": frac_detect_30d,
        "frac_detect_90d": frac_detect_90d,
        "frac_detect_365d": frac_detect_365d,

        # Paper-style raw pixel-count-per-pass method
        "median_detect_days_paper_rawN": paper_raw_stats["median"],
        "p10_detect_days_paper_rawN": paper_raw_stats["p10"],
        "p90_detect_days_paper_rawN": paper_raw_stats["p90"],
        "weighted_mean_detect_days_paper_rawN": paper_raw_stats["weighted_mean"],
        "frac_detect_30d_paper_rawN": paper_raw_stats["frac_30d"],
        "frac_detect_90d_paper_rawN": paper_raw_stats["frac_90d"],
        "frac_detect_365d_paper_rawN": paper_raw_stats["frac_365d"],

        # Paper-style effective pixel-count-per-pass method
        "median_detect_days_paper_effN": paper_eff_stats["median"],
        "p10_detect_days_paper_effN": paper_eff_stats["p10"],
        "p90_detect_days_paper_effN": paper_eff_stats["p90"],
        "weighted_mean_detect_days_paper_effN": paper_eff_stats["weighted_mean"],
        "frac_detect_30d_paper_effN": paper_eff_stats["frac_30d"],
        "frac_detect_90d_paper_effN": paper_eff_stats["frac_90d"],
        "frac_detect_365d_paper_effN": paper_eff_stats["frac_365d"],

        # Per-pass N used by the paper-style methods
        "median_N_raw_per_pass_mean": safe_median(s["N_raw_per_pass_mean"]) if "N_raw_per_pass_mean" in s.columns else np.nan,
        "median_N_eff_per_pass_mean": safe_median(s["N_eff_per_pass_mean"]) if "N_eff_per_pass_mean" in s.columns else np.nan,

        # Paper-style signal-effective method: SZA + TC + diurnal emissions
        "median_detect_days_paper_signalEffN": paper_signal_eff_stats["median"],
        "p10_detect_days_paper_signalEffN": paper_signal_eff_stats["p10"],
        "p90_detect_days_paper_signalEffN": paper_signal_eff_stats["p90"],
        "weighted_mean_detect_days_paper_signalEffN": paper_signal_eff_stats["weighted_mean"],
        #"frac_detect_30d_paper_signalEffN": paper_signal_eff_stats["frac_30d"],
        "frac_detect_30d_among_finite_paper_signalEffN": paper_signal_eff_stats["frac_30d"],
        #"frac_detect_90d_paper_signalEffN": paper_signal_eff_stats["frac_90d"],
        "frac_detect_90d_among_finite_paper_signalEffN": paper_signal_eff_stats["frac_90d"],
        #"frac_detect_365d_paper_signalEffN": paper_signal_eff_stats["frac_365d"],
        "frac_detect_365d_among_finite_paper_signalEffN": paper_signal_eff_stats["frac_365d"],
        # Diurnal emission diagnostics
        "median_hourfactor_all": safe_median(s["hourfactor_median"]) if "hourfactor_median" in s.columns else np.nan,
        "median_N_obs_signal_eff_all": safe_median(s["N_obs_signal_eff"]) if "N_obs_signal_eff" in s.columns else np.nan,
        "median_signal_to_quality_ratio_all": safe_median(signal_ratio_all),

        "median_hourfactor_emission_cells": safe_median(emission_cells["hourfactor_median"]) if "hourfactor_median" in emission_cells.columns and n_emission > 0 else np.nan,
        "median_N_obs_signal_eff_emission_cells": safe_median(emission_cells["N_obs_signal_eff"]) if "N_obs_signal_eff" in emission_cells.columns and n_emission > 0 else np.nan,
        "median_signal_to_quality_ratio_emission_cells": safe_median(signal_ratio_emis),

        # Seasonal emission diagnostics
        "median_seasonal_factor_all": safe_median(s["seasonal_factor_median"]) if "seasonal_factor_median" in s.columns else np.nan,
        "median_combined_emission_factor_all": safe_median(s["combined_emission_factor_median"]) if "combined_emission_factor_median" in s.columns else np.nan,

        "median_seasonal_factor_emission_cells": safe_median(emission_cells["seasonal_factor_median"]) if "seasonal_factor_median" in emission_cells.columns and n_emission > 0 else np.nan,
        "median_combined_emission_factor_emission_cells": safe_median(emission_cells["combined_emission_factor_median"]) if "combined_emission_factor_median" in emission_cells.columns and n_emission > 0 else np.nan,

    }


def run_orbit_case(case):
    orbit_name = case["name"]
    print("\n" + "=" * 80)
    print(f"RUNNING ORBIT CASE: {orbit_name}")
    print("=" * 80)
    time1 = time.time()
    #SImulation start and end time
    # simulation_start_epoch = DateTime(2008, 4, 28).to_epoch()
    # simulation_end_epoch   = DateTime(2008, 5, 28).to_epoch()
    t0_utc = case["t0_utc"]
    t1_utc = case["t1_utc"]


    # Fixed analysis window for fair orbit-to-orbit comparison
    case_window_days = (t1_utc - t0_utc).total_seconds() / 86400.0

    # t0_utc = datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    # t1_utc = datetime(2019, 1, 20, 0, 0, 0, tzinfo=timezone.utc)

    simulation_start_epoch = utc_datetime_to_et(t0_utc)
    simulation_end_epoch   = utc_datetime_to_et(t1_utc)

    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter

    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=earth_gravitational_parameter,
        semi_major_axis=case["semi_major_axis"],
        eccentricity=case["eccentricity"],
        inclination=case["inclination"],
        argument_of_periapsis=case["argument_of_periapsis"],
        longitude_of_ascending_node=case["longitude_of_ascending_node"],
        true_anomaly=case["true_anomaly"],
    )


    from tudatpy.dynamics.propagation_setup import dependent_variable

    # Define list of dependent variables to save
    dependent_variables_to_save = [
        dependent_variable.total_acceleration("Sat-A"),
        dependent_variable.keplerian_state("Sat-A", "Earth"),
        dependent_variable.latitude("Sat-A", "Earth"),
        dependent_variable.longitude("Sat-A", "Earth"),
        dependent_variable.altitude("Sat-A","Earth"),
    ]
    acceleration_dependent_variables_to_save = [
        dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "Sat-A", "Sun"
        ),
        dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "Sat-A", "Moon"
        ),
        dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "Sat-A", "Mars"
        ),
        dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "Sat-A", "Venus"
        ),
        dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.spherical_harmonic_gravity_type,
            "Sat-A",
            "Earth",
        ),
        dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.aerodynamic_type, "Sat-A", "Earth"
        ),
        dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.radiation_pressure_type, "Sat-A", "Sun"
        ),
    ]

    dependent_variables_to_save += acceleration_dependent_variables_to_save


    # Create termination settings
    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

    # Create numerical integrator settings
    fixed_step_size = 10.0
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        # state_tudat,
        simulation_start_epoch,
        integrator_settings,
        termination_condition,
        output_variables=dependent_variables_to_save
    )

    # Create simulation object and propagate the dynamics
    dynamics_simulator = simulator.create_dynamics_simulator(
        bodies, propagator_settings
    )


    from tudatpy.dynamics.propagation import create_dependent_variable_dictionary

    # Extract the resulting state and dependent variable history and convert it to an ndarray
    states_history = dynamics_simulator.propagation_results.state_history
    # ! Retrieving dependent variables from the dep_vars_array requires careful indexing
    dep_vars_history = dynamics_simulator.propagation_results.dependent_variable_history

    # Create dependent variable dictionary
    dep_var_dict = create_dependent_variable_dictionary(dynamics_simulator)


    relative_time_hours = (dep_var_dict.time_history - dep_var_dict.time_history[0])/3600

    # numpy array of shape (len(relative_time_hours), 3)
    delfi_total_acceleration = dep_var_dict.asarray(dependent_variable.total_acceleration("Sat-A"))

    # the dependent variable settings could also be reused from the dependent_variables_to_save list, be careful with the indexing though!
    # delfi_total_acceleration = dep_var_dict.asarray(dependent_variables_to_save[0])


    # Plot total acceleration as function of time
    total_acceleration_norm = np.linalg.norm(delfi_total_acceleration, axis=1)

    # -------------------------
    # Convert Tudat output to footprint-compatible track


    dependent_variables_array = np.column_stack([
        dep_var_dict.time_history,
        dep_var_dict.asarray(dependent_variable.latitude("Sat-A", "Earth")),
        dep_var_dict.asarray(dependent_variable.longitude("Sat-A", "Earth")),
        dep_var_dict.asarray(dependent_variable.altitude("Sat-A", "Earth")),
    ])

    track_coarse = tudat_to_track(dependent_variables_array)

    print("min/max time_utc:", track_coarse["time_utc"].min(), track_coarse["time_utc"].max())
    print("median dt:", track_coarse["time_utc"].diff().dt.total_seconds().median())

    track_coarse = tudat_to_track(dependent_variables_array)

    print(
        "min/max time_utc:",
        track_coarse["time_utc"].min(),
        track_coarse["time_utc"].max()
    )
    print(
        "median dt:",
        track_coarse["time_utc"].diff().dt.total_seconds().median()
    )

    # Check whether the ground-track centreline crosses the Netherlands.
    centreline_crosses_nl, nl_crossing_segments = (
        check_groundtrack_over_netherlands(track_coarse)
    )

    # Existing sanity check
    print(track_coarse.head())
    print("Track length:", len(track_coarse))

    # Check whether the subsatellite ground-track centreline crosses NL.
    centreline_crosses_nl, nl_crossing_segments = (
        check_groundtrack_over_netherlands(track_coarse)
    )

    # track = tudat_to_track(dep_vars_array)

    #Sanity check
    print(track_coarse.head())
    print("Track length:", len(track_coarse))


    #time1= time.time()


    # Conservative half-swath estimate for expanded bbox (no curvature needed just for windowing)
    # fixed_swath: use SWATH_WIDTH_KM; fixed_nadir: approximate swath = N_ACROSS * NADIR_PIXEL_M
    if GEOM_MODE == "fixed_swath":
        half_swath_m = 0.5 * SWATH_WIDTH_KM * 1000.0
    else:
        half_swath_m = 0.5 * N_ACROSS * float(NADIR_PIXEL_M)

    # Expand bbox by half-swath (rough degree conversion; sufficient for candidate window detection)
    buffer_lat_deg = (half_swath_m / 1000.0) / 111.0
    buffer_lon_deg = (half_swath_m / 1000.0) / (111.0 * np.cos(np.deg2rad(52.0)))

    LAT_MIN = GRID_SOUTH - buffer_lat_deg
    LAT_MAX = GRID_NORTH + buffer_lat_deg
    LON_MIN = GRID_WEST  - buffer_lon_deg
    LON_MAX = GRID_EAST  + buffer_lon_deg

    # Keep only coarse points that fall inside expanded bbox
    coarse_hits = track_coarse[
        (track_coarse["lat"] >= LAT_MIN) & (track_coarse["lat"] <= LAT_MAX) &
        (track_coarse["lon"] >= LON_MIN) & (track_coarse["lon"] <= LON_MAX)
    ].copy()

    candidate_windows = windows_from_coarse_track(
        coarse_hits,
        pad_seconds=WINDOW_PAD_SECONDS,
        max_gap_seconds=WINDOW_MERGE_GAP_SECONDS
    )
    candidate_windows = merge_overlapping_windows(candidate_windows)

    print("-------------------------")
    print("Tudat coarse-to-fine windowing")
    print("Coarse hits in expanded bbox:", len(coarse_hits))
    print("Candidate windows:", len(candidate_windows))
    if candidate_windows:
        print("First window:", candidate_windows[0][0], "→", candidate_windows[0][1])


    # -------------------------------------------------------------
    # (B) Run fine Tudat propagation inside each candidate window


    # Build a sorted list of coarse epochs so we can get a near initial state for each fine window
    coarse_epochs = np.array(sorted(states_history.keys()), dtype=float)


    # Fine-window initialization:
    # The initial Cartesian state is selected from the nearest saved 10 s coarse
    # epoch. No interpolation is applied. Consequently, the selected state may
    # correspond to an epoch up to approximately half a coarse step away from the
    # requested fine-window start time.


    def get_nearest_coarse_state(epoch_s):
        """Return a Cartesian state from the coarse Tudat run at the nearest saved epoch."""

        # Initialize each fine window from the nearest saved coarse state.
        # No state interpolation is performed between adjacent 10 s coarse outputs.


        i = np.searchsorted(coarse_epochs, epoch_s)
        if i <= 0:
            return states_history[coarse_epochs[0]]
        if i >= len(coarse_epochs):
            return states_history[coarse_epochs[-1]]
        e0 = coarse_epochs[i - 1]
        e1 = coarse_epochs[i]
        return states_history[e0] if abs(epoch_s - e0) <= abs(epoch_s - e1) else states_history[e1]

    def datetime_to_tudat_epoch(dt_utc):
        """
        Convert a timezone-aware datetime (UTC) to Tudat epoch seconds since J2000.
        """
        # Pandas Timestamp -> python datetime
        dt_py = pd.to_datetime(dt_utc, utc=True).to_pydatetime()
        #return DateTime.from_python_datetime(dt_py).to_epoch()
        return utc_datetime_to_et(dt_py)

    fine_tracks = []

    for iwin, (ws, we) in enumerate(candidate_windows, start=1):
        ws_epoch = datetime_to_tudat_epoch(ws)
        we_epoch = datetime_to_tudat_epoch(we)

        # Initial state for fine propagation from coarse history
        x0 = get_nearest_coarse_state(ws_epoch)

        # Fine integrator
        fine_integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step=FINE_STEP_SECONDS,
            coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
        )

        fine_termination = propagation_setup.propagator.time_termination(we_epoch)

        # Save ONLY what track needs (keeps fine runs lightweight)
        fine_dep_vars = [
            dependent_variable.latitude("Sat-A", "Earth"),
            dependent_variable.longitude("Sat-A", "Earth"),
            dependent_variable.altitude("Sat-A", "Earth"),
        ]

        fine_propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            x0,
            ws_epoch,
            fine_integrator_settings,
            fine_termination,
            output_variables=fine_dep_vars
        )

        sim_fine = simulator.create_dynamics_simulator(bodies, fine_propagator_settings)
        dep_fine = result2array(sim_fine.propagation_results.dependent_variable_history)

        # dep_fine columns: [time, lat(rad), lon(rad), alt(m)] because we saved only these three
        track_fine_i = tudat_to_track(dep_fine)
        track_fine_i["pass_id"] = iwin # Giving every fine-track sample a label telling us which overpass it belongs to

        fine_tracks.append(track_fine_i)

        print(f"[{iwin}/{len(candidate_windows)}] Fine Tudat window: {ws} → {we} | scanlines: {len(track_fine_i)}")

    # Merge all fine tracks into one and use it for footprint generation
    if fine_tracks:
        track = (
            pd.concat(fine_tracks, ignore_index=True)
            .drop_duplicates(subset=["time_utc"])
            .sort_values("time_utc")
            .reset_index(drop=True)
        )
        track["model_hour"] = track["time_utc"].dt.floor("h")
    else:
        track = pd.DataFrame(columns=["time_utc", "lat", "lon", "alt_km", "brng_rad", "model_hour", "pass_id"])

    print("Final fine track length:", len(track))


#Output options

    SAVE_POLYGON_PARQUET = False
    OUT_DIR = Path(
        f"./outputs/{orbit_name}__"
        f"{int((simulation_end_epoch-simulation_start_epoch)/3600)} hours_"
        f"{GEOM_MODE}_"
        f"{int(NADIR_PIXEL_M)}m_"
        f"{N_ACROSS}pix/"
    )

    PLOT_DIR = OUT_DIR / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Save Netherlands centreline-crossing segments.
    if not nl_crossing_segments.empty:
        nl_crossing_segments.to_csv(
            OUT_DIR / "netherlands_groundtrack_crossings.csv",
            index=False
        )


    # Plot the ground track around the Netherlands.
    plot_netherlands_groundtrack(
        track_coarse=track_coarse,
        orbit_name=orbit_name,
        output_base=(
            PLOT_DIR / "netherlands_groundtrack"
        )
    )

    time2 = time.time()
    polys_all = []

    SWATH_HALF_WIDTH_KM = SWATH_WIDTH_KM / 2.0

    for i_scan, row in track.iterrows():
        b = float(row["brng_rad"])
        lat0 = np.deg2rad(row["lat"])
        lon0 = np.deg2rad(row["lon"])
        b_left = b - np.pi / 2.0
        b_right = b + np.pi / 2.0

        # --------
        # Scanline prefilter: do swath edges intersect NL bbox?
        # Use IFOV swath half-width (specified earlier) here.
        # --------

        # Compute effective half swath on the surface for this scanline
        H = float(row["alt_km"]) * 1000.0 if USE_DYNAMIC_ALTITUDE else 824000.0
        if GEOM_MODE == "fixed_swath":
            half_swath_m = (SWATH_WIDTH_KM * 1000.0) / 2.0
        else:
            # fixed_nadir: half swath ~ max |x_center| from theta grid (spherical or flat)
            ifov_x = float(NADIR_PIXEL_M) / H
            theta_edge = ((N_ACROSS - 1) / 2.0) * ifov_x
            if USE_EARTH_CURVATURE:
                psi_edge, _ = ground_central_angle_from_look(theta_edge, H, R_m=EARTH_RADIUS_M)
                half_swath_m = EARTH_RADIUS_M * abs(psi_edge)
            else:
                half_swath_m = H * np.tan(theta_edge)

        latL, lonL = destination_point(lat0, lon0, b_left,  half_swath_m)
        latR, lonR = destination_point(lat0, lon0, b_right, half_swath_m)

        # Old
        # latL, lonL = destination_point(lat0, lon0, b_left,  SWATH_HALF_WIDTH_KM * 1000.0)
        # latR, lonR = destination_point(lat0, lon0, b_right, SWATH_HALF_WIDTH_KM * 1000.0)

        x0, y0 = np.rad2deg(lonL), np.rad2deg(latL)
        x1, y1 = np.rad2deg(lonR), np.rad2deg(latR)

        if not segment_intersects_bbox(x0, y0, x1, y1, GRID_WEST, GRID_EAST, GRID_SOUTH, GRID_NORTH):
            continue

        # --------
        # Build ALL pixels for this scanline using IFOV
        # --------
        df_scan = build_scanline_polys_ifov(i_scan, row, b)
        df_scan["pass_id"] = int(row["pass_id"])

        # Diagnostic print
        if i_scan % 2000 == 0:
            # Effective swath width at surface based on center positions
            print(
                f"i={i_scan} H={H/1000:.1f} km "
                f"w_nadir={df_scan.loc[df_scan['j_across']==N_ACROSS//2,'w_x_m'].iloc[0]:.0f} m "
                f"w_edge={df_scan.loc[df_scan['j_across']==0,'w_x_m'].iloc[0]:.0f} m"
            )


        # --------
        # Pixel-level filter: keep only pixels whose centers fall inside LE grid
        # Assign observations by pixel-centre location.
        # A polygon that only partially overlaps the domain is excluded when its centre lies outside the LOTOS-EUROS grid.
        # --------
        m = in_le_grid(df_scan["lon_c"].to_numpy(), df_scan["lat_c"].to_numpy())
        if not np.any(m):
            continue

        df_scan = df_scan.loc[m].copy()

        # Add LE grid indices for sampling
        ix, iy = lonlat_to_ij(df_scan["lon_c"].to_numpy(), df_scan["lat_c"].to_numpy())
        df_scan["ix"] = ix
        df_scan["iy"] = iy

        polys_all.append(df_scan)

    polys_nl = pd.concat(polys_all, ignore_index=True) if polys_all else pd.DataFrame()


    # ============================================================
    # FINAL NETHERLANDS COVERAGE CHECK
    # ============================================================

    print("\n" + "=" * 80)
    print("FINAL NETHERLANDS COVERAGE CHECK")
    print("=" * 80)

    print(
        "Ground-track centreline crosses NL:",
        centreline_crosses_nl
    )

    if polys_nl.empty:
        print("Instrument swath overlaps NL: False")
        print("Number of NL footprint pixels: 0")
    else:
        print("Instrument swath overlaps NL: True")
        print(f"Number of NL footprint pixels: {len(polys_nl):,}")

        if "pass_id" in polys_nl.columns:
            print(
                "Number of regional passes:",
                polys_nl["pass_id"].nunique()
            )

        print(
            "First NL footprint time:",
            polys_nl["time_utc"].min()
        )
        print(
            "Last NL footprint time:",
            polys_nl["time_utc"].max()
        )

    print("=" * 80)

    # ------------------------------------------------------------
    # Local solar time diagnostics for NL observations
    # ------------------------------------------------------------
    local_time_metrics = {}

    # ------------------------------------------------------------
    # Local solar time diagnostics for NL observations
    # ------------------------------------------------------------
    local_time_metrics = {}

    if not polys_nl.empty:
        polys_nl, local_time_metrics = make_local_time_diagnostics(
            polys_nl=polys_nl,
            plot_dir=PLOT_DIR,
            orbit_name=orbit_name
        )


    # ============================================================
    # ADD THERMAL CONTRAST FIELDS TO POLYS_NL
    # ============================================================

    if not polys_nl.empty:
        print("-------------------------")
        print("Adding hourly T500-based thermal contrast to polys_nl...")

        polys_nl = add_tc_fields_to_polys(polys_nl)

        print("TC columns added:")
        print([c for c in ["thermal_contrast_K", "w_tc"] if c in polys_nl.columns])


    # ============================================================
    # ADD CLOUD COVER FIELDS TO POLYS_NL
    # ============================================================

    if not polys_nl.empty:
        print("-------------------------")
        print("Adding ECMWF total cloud cover to polys_nl...")

        polys_nl = add_tcc_fields_to_polys(polys_nl)

        print("Cloud columns added:")
        print([c for c in ["tcc", "w_cloud"] if c in polys_nl.columns])

    # ============================================================
    # DETAILED TC/TCC MISSINGNESS DIAGNOSTICS
    # ============================================================
    environmental_missingness_metrics = {}

    if not polys_nl.empty:
        environmental_missingness_metrics = (
            generate_environmental_missingness_diagnostics(
                orbit_name=orbit_name,
                polys_nl=polys_nl,
                output_dir=(
                    PLOT_DIR
                    / "environmental_missingness_diagnostics"
                )
            )
        )
    # ============================================================
    # ADD DIURNAL NH3 EMISSION FACTOR TO POLYS_NL
    # ============================================================
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Hourfactor diagnostics initialized here so they always exist
    # ------------------------------------------------------------
    hourfactor_time_metrics = {}
    peak_hourfactor_metrics = {}

    if not polys_nl.empty:
        print("-------------------------")
        print("Adding SZA/solar-based hourly NH3 emission factors to polys_nl...")

        polys_nl = add_diurnal_factor_to_polys(polys_nl)

        print("Diurnal columns added:")
        print([c for c in ["hourfactor"] if c in polys_nl.columns])

        # ------------------------------------------------------------
        # Diagnose whether this orbit observes during high-hourfactor hours
        # ------------------------------------------------------------
        hourfactor_time_metrics = make_hourfactor_by_local_time_diagnostic(
            polys_nl=polys_nl,
            plot_dir=PLOT_DIR,
            orbit_name=orbit_name
        )

        # ------------------------------------------------------------
        # Diagnose whether observations occur near the daily peak hourfactor
        # ------------------------------------------------------------
        polys_nl, peak_hourfactor_metrics = add_relative_to_peak_hourfactor(
            polys_nl=polys_nl,
            plot_dir=PLOT_DIR,
            orbit_name=orbit_name
        )

    # ============================================================
    # ADD SEASONAL NH3 EMISSION FACTOR TO POLYS_NL
    # ============================================================

    if not polys_nl.empty:
        print("-------------------------")
        print("Adding agriculture-manure seasonal NH3 emission factors to polys_nl...")

        polys_nl = add_seasonal_factor_to_polys(polys_nl)

        print("Seasonal columns added:")
        print([c for c in ["seasonal_factor"] if c in polys_nl.columns])
    # ============================================================
    # Run the emission-weighted observability and detectability model.
    # ============================================================

    print("-------------------------")
    print("Running observability model...")

    emis_path = (
    DATA_ROOT
    / "emissions"
    / "Base-emissions-CAMS-REG_v8_0_2022.nc"
)

    # Spatial NH3 totals are taken from the CAMS-REG 2022 inventory, while the
    # propagated observation times and diurnal/seasonal temporal factors represent
    # 2019. The inventory is therefore treated as a static spatial baseline whose
    # annual total is redistributed in time by the 2019 temporal profiles.



    # countries_to_keep = [
    #     'Austria', 'Belgium', 'Switzerland',
    #     'Czech Republic', 'Germany', 'Denmark', 'Spain',
    #     'France', 'United Kingdom', 'Hungary', 'Ireland',
    #     'Italy', 'Luxembourg', 'Netherlands', 'Poland', 'Portugal'
   # ]
    countries_to_keep = [
         'Netherlands'
    ]

    # Retain emissions assigned to the Netherlands in the CAMS country metadata.
    # This is an administrative source filter, not merely a geographic crop of all
    # source cells intersecting the LOTOS-EUROS analysis domain.



    if polys_nl.empty:
        print(f"{orbit_name}: No NL pixels -> skipping orbit case.")
        return None
    else:
        observability_df, sampling_summary, emis_cell = run_observability_from_polys(
            polys_nl=polys_nl,
            emis_path=emis_path,
            countries_to_keep=countries_to_keep,
            unit_scale_to_kgyr=1e6, # CAMS values are in Gg/yr; 1 Gg = 1e6 kg.
            sigma_I_molec_cm2=5e15,
            sigma_X_molec_cm2=0.75e15,
            F_clear=1.0,  # Cloud loss is represented explicitly by the pixel-level cloud weight.
            pressure_hpa=1000.0,
            wind_kmh=18.0,
            W_km=10.0,
            window_days = case_window_days
        )
        print("Sampling summary columns:", sampling_summary.columns.tolist())
        print("Median raw N_obs:", sampling_summary["N_obs"].median())
        print("Median effective N_obs_eff:", sampling_summary["N_obs_eff"].median())
        print("SZA min/median:")
        print(sampling_summary[["sza_min_deg", "sza_median_deg"]].describe())

        if "tc_median_K" in sampling_summary.columns:
            print("Thermal contrast diagnostics:")
            print(
                sampling_summary[
                    ["tc_median_K", "w_tc_mean", "w_quality_mean"]
                ].describe()
            )

        if "tcc_median" in sampling_summary.columns:
            print("\n========== CLOUD COVER DIAGNOSTICS ==========")
            print(
                sampling_summary[
                    [
                        "tcc_median",
                        "tcc_mean",
                        "w_cloud_mean",
                        "N_obs",
                        "N_obs_eff",
                        "N_obs_signal_eff",
                        "w_quality_mean",
                        "w_signal_mean",
                    ]
                ].describe()
            )

            cloud_reduction_ratio = (
                sampling_summary["N_obs_eff"]
                / sampling_summary["N_obs"].replace(0, np.nan)
            )

            print("\nN_obs_eff / N_obs ratio after SZA + TC + cloud:")
            print(cloud_reduction_ratio.describe())


        if "hourfactor_mean" in sampling_summary.columns:
            print("\n========== DIURNAL EMISSION FACTOR DIAGNOSTICS ==========")
            print(
                sampling_summary[
                    [
                        "hourfactor_mean",
                        "hourfactor_median",
                        "hourfactor_min",
                        "hourfactor_max",
                        "N_obs",
                        "N_obs_eff",
                        "N_obs_signal_eff",
                        "w_quality_mean",
                        "w_signal_mean",
                    ]
                ].describe()
            )

            ratio = sampling_summary["N_obs_signal_eff"] / sampling_summary["N_obs_eff"].replace(0, np.nan)
            print("\nN_obs_signal_eff / N_obs_eff ratio:")
            print(ratio.describe())

        if "seasonal_factor_mean" in sampling_summary.columns:
            print("\n========== SEASONAL EMISSION FACTOR DIAGNOSTICS ==========")
            print(
                sampling_summary[
                    [
                        "seasonal_factor_mean",
                        "seasonal_factor_median",
                        "seasonal_factor_min",
                        "seasonal_factor_max",
                        "combined_emission_factor_mean",
                        "combined_emission_factor_median",
                        "combined_emission_factor_min",
                        "combined_emission_factor_max",
                        "N_obs",
                        "N_obs_eff",
                        "N_obs_signal_eff",
                        "w_quality_mean",
                        "w_signal_mean",
                    ]
                ].describe()
            )

            ratio = (
                sampling_summary["N_obs_signal_eff"]
                / sampling_summary["N_obs_eff"].replace(0, np.nan)
            )

            print("\nN_obs_signal_eff / N_obs_eff ratio after hourfactor + seasonal_factor:")
            print(ratio.describe())


        print("Total raw NH3 over selected LE cells [Gg/yr]:", emis_cell["nh3_raw"].sum())
        print("Emission cells:", len(emis_cell))

        print("\n========== PAPER-STYLE N PER PASS CHECK ==========")
        print(
            observability_df[
                [
                    "N_obs",
                    "N_obs_eff",
                    "N_obs_signal_eff",
                    "N_raw_per_pass_mean",
                    "N_eff_per_pass_mean",
                    "N_signal_eff_per_pass_mean",
                    "dt_median_days",
                    "M_req_obs",
                    "time_to_detect_days",
                    "time_to_detect_days_paper_rawN",
                    "time_to_detect_days_paper_effN",
                    "time_to_detect_days_paper_signalEffN",
                ]

            ].describe()
        )

        # Save outputs
        observability_df.to_parquet(OUT_DIR / "observability_by_cell.parquet", index=False)
        sampling_summary.to_parquet(OUT_DIR / "sampling_summary_by_cell.parquet", index=False)
        emis_cell.to_parquet(OUT_DIR / "emissions_by_cell.parquet", index=False)

        sampling_summary.to_csv(OUT_DIR / "sampling_summary_by_cell.csv", index=False)
        emis_cell.to_csv(OUT_DIR / "emissions_by_cell.csv", index=False)

        print("Saved observability results to:", OUT_DIR / "observability_by_cell.parquet")
        print("Saved sampling summary to:", OUT_DIR / "sampling_summary_by_cell.parquet")
        print("Saved emissions to:", OUT_DIR / "emissions_by_cell.parquet")


    # Cd sensitivity diagnostics.

    kepler_elements = dep_var_dict.asarray(
        dependent_variable.keplerian_state(
            "Sat-A",
            "Earth"
        )
    )

    final_altitude_km = float(track_coarse["alt_km"].iloc[-1])

    final_semi_major_axis_km = float(
        kepler_elements[-1, 0] / 1000.0
    )

    initial_altitude_km = float(track_coarse["alt_km"].iloc[0])

    initial_semi_major_axis_km = float(
        kepler_elements[0, 0] / 1000.0
    )

    altitude_change_km = (
        final_altitude_km - initial_altitude_km
    )

    semi_major_axis_change_km = (
        final_semi_major_axis_km - initial_semi_major_axis_km
    )

    nl_pass_count = len(candidate_windows)

    print("\n======================================")
    print("Cd SENSITIVITY CHECK METRICS")
    print("======================================")
    print(f"Orbit case: {orbit_name}")

    print(f"Initial altitude [km]       : {initial_altitude_km:.3f}")
    print(f"Final altitude [km]         : {final_altitude_km:.3f}")
    print(f"Altitude change [km]        : {altitude_change_km:.3f}")

    print(f"Initial semi-major axis {initial_semi_major_axis_km:.3f}")
    print(f"Final semi-major axis [km]  : {final_semi_major_axis_km:.3f}")
    print(f"SMA change [km]             : {semi_major_axis_change_km:.3f}")

    print(f"NL pass count               : {nl_pass_count}")
    print("======================================\n")


    time3 = round(time.time(),3)
    print('-------------------------')
    print("Done building polygons...")
    print(f"Time taken (s):", round(time3 - time2,2))
    print("Total NL pixels (IFOV):", len(polys_nl))
    #print(polys_nl.head())

    if SAVE_POLYGON_PARQUET:
        polys_nl.to_parquet(OUT_DIR / f"{orbit_name}_synthetic_swath_IFOV_NL_{YEAR}.parquet", index=False)
        # polys_nl.to_parquet(f"{SAT}_synthetic_swath_IFOV_NL_{YEAR}.parquet", index=False)

    ### Writing pixels/polygons out to a CSO-like format ###
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if polys_nl.empty:
        print("No pixels intersect NL grid for this run. Skipping NetCDF writing. for this orbit")
        return None


    df = polys_nl.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"])

    # Group by day (UTC)
    groups = df.groupby(df["time_utc"].dt.floor("D"))

    n_written = 0
    for day_ts, g in groups:
        day_str = day_ts.strftime("%Y%m%d")
        out_nc = OUT_DIR / f"synthetic_geom_{day_str}.nc"

        written = write_cso_like_geometry_nc(g, out_nc, orbit_tag=day_str)
        if written is None:
            continue

        n_written += 1
    #print(f"Wrote {written} with {len(g)} pixels")

    print(f"Done. Wrote {n_written} daily files to {OUT_DIR}.")
    time4 = round(time.time(),3)
    print(f"Time taken (s):", round(time4 - time3,2))
    print("Total time taken (s):", round(time.time() - time1,2))


    # -------------------------
    # PLOTS FROM TRACK (Tudat)  -- use FINE track!
    # -------------------------

    # Option A: plot full fine track (windowed)
    track_plot = track.sort_values("time_utc").reset_index(drop=True)

    # # Option B: plot NL-only fine track (recommended for comparison)
    # track_nl = track[
    #     (track["lat"] >= LAT_MIN) & (track["lat"] <= LAT_MAX) &
    #     (track["lon"] >= LON_MIN) & (track["lon"] <= LON_MAX)
    # ].copy()

    # track_plot = track_nl.sort_values("time_utc").reset_index(drop=True)

    # Sanity check the cadence
    dt = track_plot["time_utc"].diff().dt.total_seconds()
    print("Track plot points:", len(track_plot))
    print("Median dt [s]:", dt.median(), "min dt [s]:", dt.min(), "max dt [s]:", dt.max())

    # Full (windowed) ground track scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(track_plot["lon"], track_plot["lat"], s=1.0)
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("Tudat (perturbed) centerline ground track (fine track)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "groundtrack_centerline_full.png", dpi=200)
    plt.close()

    # 3-hour subset from the first point in this filtered track
    if not track_plot.empty:
        t0 = track_plot["time_utc"].iloc[0]
        t_end = t0 + pd.Timedelta(hours=3)
        sub = track_plot[(track_plot["time_utc"] >= t0) & (track_plot["time_utc"] <= t_end)]

        plt.figure(figsize=(8, 6))
        plt.scatter(sub["lon"], sub["lat"], s=2)
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("Tudat (perturbed) 3-hour ground track (fine track)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "groundtrack_centerline_3h.png", dpi=200)
        plt.close()

    # Altitude vs time
    if not track_plot.empty:
        plt.figure(figsize=(9, 4))
        plt.scatter(track_plot["time_utc"], track_plot["alt_km"], s=1)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Altitude [km]")
        plt.title("Tudat (perturbed) altitude over time (fine track)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "altitude_vs_time.png", dpi=200)
        plt.close()

    # Save fine track (for later comparisons)
    track.to_csv(PLOT_DIR / "track_centerline_fine.csv", index=False)
    track.to_parquet(PLOT_DIR / "track_centerline_fine.parquet", index=False)


    plt.figure(figsize=(9, 5))
    plt.title("Total acceleration norm on Sat-A over the course of propagation.")
    plt.plot(relative_time_hours, total_acceleration_norm)
    plt.xlabel('Time [hr]')
    plt.ylabel('Total Acceleration [m/s$^2$]')
    plt.xlim([min(relative_time_hours), max(relative_time_hours)])
    plt.grid()
    plt.tight_layout()
    plt.savefig(PLOT_DIR /"Total_acc_norm.png")
    plt.close()


    # Plot Kepler elements as a function of time
    # kepler_elements = dep_var_dict.asarray(dependent_variable.keplerian_state("Sat-A", "Earth"))
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
    fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

    # Semi-major Axis
    semi_major_axis = kepler_elements[:,0] / 1e3
    ax1.plot(relative_time_hours, semi_major_axis)
    ax1.set_ylabel('Semi-major axis [km]')

    # Eccentricity
    eccentricity = kepler_elements[:,1]
    ax2.plot(relative_time_hours, eccentricity)
    ax2.set_ylabel('Eccentricity [-]')

    # Inclination
    inclination = np.rad2deg(kepler_elements[:,2])
    ax3.plot(relative_time_hours, inclination)
    ax3.set_ylabel('Inclination [deg]')

    # Argument of Periapsis
    argument_of_periapsis = np.rad2deg(kepler_elements[:,3])
    ax4.plot(relative_time_hours, argument_of_periapsis)
    ax4.set_ylabel('Argument of Periapsis [deg]')

    # Right Ascension of the Ascending Node
    raan = np.rad2deg(kepler_elements[:,4])
    ax5.plot(relative_time_hours, raan)
    ax5.set_ylabel('RAAN [deg]')

    # True Anomaly
    true_anomaly = np.rad2deg(kepler_elements[:,5])
    ax6.scatter(relative_time_hours, true_anomaly, s=1)
    ax6.set_ylabel('True Anomaly [deg]')
    ax6.set_yticks(np.arange(0, 361, step=60))

    for ax in fig.get_axes():
        ax.set_xlabel('Time [hr]')
        ax.set_xlim([min(relative_time_hours), max(relative_time_hours)])
        ax.grid()
    plt.tight_layout()
    plt.savefig(PLOT_DIR /"Kepler_elements.png")
    plt.close()
    from tudatpy.dynamics.propagation_setup import acceleration

    # map acceleration types to human-readable strings for labeling
    acceleration_type_to_string = {
        acceleration.AvailableAcceleration.point_mass_gravity_type: "PM",
        acceleration.AvailableAcceleration.spherical_harmonic_gravity_type: "SH",
        acceleration.AvailableAcceleration.aerodynamic_type: "Aerodynamic",
        acceleration.AvailableAcceleration.radiation_pressure_type: "Radiation Pressure",
    }

    plt.figure(figsize=(9, 5))

    for acceleration_dep_var_setting in acceleration_dependent_variables_to_save:
        acceleration_norm = dep_var_dict.asarray(acceleration_dep_var_setting)
        label = f"{acceleration_type_to_string[acceleration_dep_var_setting.acceleration_model_type]} {acceleration_dep_var_setting.secondary_body}"

        plt.plot(relative_time_hours, acceleration_norm, label=label)

    plt.xlim([min(relative_time_hours), max(relative_time_hours)])
    plt.xlabel("Time [hr]")
    plt.ylabel("Acceleration Norm [m/s$^2$]")

    plt.legend(bbox_to_anchor=(1.005, 1))
    plt.suptitle(
        "Accelerations norms on Sat-A, distinguished by type and origin, over the course of propagation."
    )
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(PLOT_DIR /"acceleration_over_time.png")
    plt.close()


#Debug of revisit timestamps
    df_pass_debug = (
        polys_nl
        .groupby(["iy", "ix", "pass_id"], as_index=False)
        .agg(time_utc=("time_utc", "min"))
    )
    selected_cells = [
        (120, 80),
        (100, 87),
        (150, 120),
    ]

    for iy0, ix0 in selected_cells:

        cell_passes = (
            df_pass_debug[
                (df_pass_debug["iy"] == iy0)
                & (df_pass_debug["ix"] == ix0)
            ]
            .sort_values("time_utc")
        )

        if len(cell_passes) < 2:
            continue

        dt_hours = (
            cell_passes["time_utc"]
            .diff()
            .dt.total_seconds()
            / 3600.0
        )

        plt.figure(figsize=(8,4))
        plt.hist(dt_hours.dropna(), bins=50)

        plt.xlabel("Revisit gap [hours]")
        plt.ylabel("Count")
        plt.title(f"{orbit_name} : iy={iy0}, ix={ix0}")

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / f"revisit_gap_hist_iy{iy0}_ix{ix0}.png",
            dpi=200
        )

        plt.close()


    # -------------------------
    # PLOTS FROM POLYS_NL (Tudat)
    # -------------------------
    if polys_nl.empty:
        print("No NL pixels -> skipping footprint plots.")
    else:
        # Pixel centers
        plt.figure(figsize=(6, 6))
        plt.scatter(polys_nl["lon_c"], polys_nl["lat_c"], s=2, alpha=0.6)
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("Pixel center locations over NL (Tudat perturbed)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "pixel_centers_over_NL.png", dpi=200)
        plt.close()

        # Off-nadir angle distribution
        plt.figure(figsize=(7, 4))
        plt.hist(polys_nl["theta_deg"], bins=40)
        plt.xlabel("Off-nadir angle θ [deg]")
        plt.ylabel("Pixel count")
        plt.title("Off-nadir viewing angle distribution (Tudat, NL pixels)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "theta_distribution.png", dpi=200)
        plt.close()

        # Pixel size distribution
        plt.figure(figsize=(7, 4))
        plt.hist(polys_nl["w_x_m"], bins=40, alpha=0.7, label="Across-track (w_x)")
        plt.hist(polys_nl["w_y_m"], bins=40, alpha=0.7, label="Along-track (w_y)")
        plt.xlabel("Pixel size [m]")
        plt.ylabel("Pixel count")
        plt.title("Pixel size distribution (Tudat, NL pixels)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "pixel_size_distribution.png", dpi=200)
        plt.close()

        # Polygon outlines (sample to avoid slow plotting)
        plt.figure(figsize=(6, 6))
        NMAX = 5000
        df_poly = polys_nl if len(polys_nl) <= NMAX else polys_nl.sample(NMAX, random_state=0)

        for _, row in df_poly.iterrows():
            lons = [row["lon_ul"], row["lon_ur"], row["lon_lr"], row["lon_ll"], row["lon_ul"]]
            lats = [row["lat_ul"], row["lat_ur"], row["lat_lr"], row["lat_ll"], row["lat_ul"]]
            plt.plot(lons, lats, color="blue", alpha=0.15, linewidth=0.5)

        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title(f"Pixel footprint polygons over NL (Tudat) [sample {len(df_poly)}]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "pixel_polygons_over_NL.png", dpi=200)
        plt.close()

        # Swath centroid over NL (median pixel center per scanline time)
        centroids = (
            polys_nl.groupby("time_utc")[["lat_c", "lon_c"]]
            .median()
            .reset_index()
            .rename(columns={"lat_c": "lat_centroid", "lon_c": "lon_centroid"})
        )

        plt.figure(figsize=(6, 6))
        plt.plot(centroids["lon_centroid"], centroids["lat_centroid"], "-", lw=2)
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("Swath centroid (derived from pixel centers) over NL (Tudat)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "swath_centroid_over_NL.png", dpi=200)
        plt.close()

    if not polys_nl.empty:
        print(sampling_summary.describe()) #Check N_obs not unrealistically large, dt_min_s not full of zeros, dt_median_s in reasonable range (hours–days)
        print(observability_df.head())
        print("Emission cells:", len(emis_cell))
        print("Sampled cells:", len(sampling_summary))
        print("Merged cells:", len(observability_df))


        print("Nonzero emission cells:")
        print((emis_cell["nh3_kg_yr"] > 0).sum())

        print("Merged cells with emissions:")
        print((observability_df["nh3_kg_yr"] > 0).sum())


    # ------------------------------------------------------------
    # Generate all single-orbit Results plots and filter ablations
    # ------------------------------------------------------------
    if GENERATE_RESULTS_PLOTS:
        try:
            generate_single_orbit_results(
                orbit_name=orbit_name,
                track_coarse=track_coarse,
                polys_nl=polys_nl,
                sampling_summary=sampling_summary,
                emis_cell=emis_cell,
                observability_df=observability_df,
                out_dir=OUT_DIR
            )
        except Exception as exc:
            print(
                f"[WARNING] Results plotting failed for "
                f"{orbit_name}: {exc}"
            )
            import traceback
            traceback.print_exc()


        # ------------------------------------------------------------
        # Plot number of CAMS contributions per LE cell
        # ------------------------------------------------------------
        if "n_contrib" in emis_cell.columns:
            contrib_grid = build_full_le_grid(emis_cell, "n_contrib", fill_value=0.0)

            lon_edges_le = GRID_WEST + np.arange(GRID_NX + 1) * GRID_DLON
            lat_edges_le = GRID_SOUTH + np.arange(GRID_NY + 1) * GRID_DLAT

            contrib_grid_plot = np.ma.masked_where(contrib_grid <= 0, contrib_grid)

            plt.figure(figsize=(7, 8))
            pcm = plt.pcolormesh(
                lon_edges_le,
                lat_edges_le,
                contrib_grid_plot,
                shading="flat",
                cmap="viridis",
                edgecolors="none",
                linewidth=0,
                antialiased=False
            )
            plt.colorbar(pcm, label="Number of CAMS cell contributions")
            plt.xlabel("Longitude [deg]")
            plt.ylabel("Latitude [deg]")
            plt.title("CAMS contributions per LE cell")
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "CAMS_contributions_per_LE_cell.png", dpi=300)
            plt.close()

        # ------------------------------------------------------------
        # Binary mask: To check which LE cells receive any CAMS contribution. Diagnostics plot
        # ------------------------------------------------------------
        if "n_contrib" in emis_cell.columns:
            contrib_mask = (contrib_grid > 0).astype(int)

            plt.figure(figsize=(7, 8))
            plt.imshow(
                contrib_mask,
                origin="lower",
                extent=[GRID_WEST, GRID_EAST, GRID_SOUTH, GRID_NORTH],
                aspect="auto",
                interpolation="nearest",
                cmap="Greys"
            )
            plt.xlabel("Longitude [deg]")
            plt.ylabel("Latitude [deg]")
            plt.title("LE cells receiving CAMS contributions (binary mask)")
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "CAMS_contribution_mask.png", dpi=300)
            plt.close()


        #import matplotlib.pyplot as plt

        plt.scatter(emis_cell["ix"], emis_cell["iy"],
                    c=emis_cell["nh3_kg_yr"], s=5)
        plt.colorbar(label="NH3 kg/yr")
        plt.title("Emission distribution")
        plt.savefig(PLOT_DIR /"Emission_distribution.png")
        plt.close()


        # Convert grid indices to lon/lat if you want geographic plotting
        emis_plot = emis_cell.copy()
        emis_plot["lon"] = GRID_WEST + emis_plot["ix"] * GRID_DLON
        emis_plot["lat"] = GRID_SOUTH + emis_plot["iy"] * GRID_DLAT

        # Avoid log(0): keep only strictly positive values for log scaling
        emis_plot = emis_plot[emis_plot["nh3_kg_yr"] > 0].copy()

        plt.figure(figsize=(6, 8))

        sc = plt.scatter(
            emis_plot["lon"],   # or emis_plot["ix"]
            emis_plot["lat"],   # or emis_plot["iy"]
            c=emis_plot["nh3_kg_yr"],
            s=8,
            # norm=mcolors.LogNorm(
            #     vmin=emis_plot["nh3_kg_yr"].min(),
            #     vmax=emis_plot["nh3_kg_yr"].max()
            norm=mcolors.LogNorm(vmin=1, vmax=1e6)  #Just to manually compare with the CAMS plot. Make sure to remove this.

        )

        plt.colorbar(sc, label="NH3 [kg/yr] (log scale)")
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("Emission distribution (log scale)")
        plt.grid(True)
        plt.savefig(PLOT_DIR / "Emission_distribution_logscale.png", dpi=200)
        plt.close()


        emis_plot = emis_cell.copy()
        emis_plot["lon"] = GRID_WEST + (emis_plot["ix"] + 0.5) * GRID_DLON
        emis_plot["lat"] = GRID_SOUTH + (emis_plot["iy"] + 0.5) * GRID_DLAT
        emis_plot = emis_plot[emis_plot["nh3_kg_km2_yr"] > 0].copy()

        vals = emis_plot["nh3_kg_km2_yr"]
        vmin = np.percentile(vals, 5)
        vmax = np.percentile(vals, 99)

        plt.figure(figsize=(6, 8))
        sc = plt.scatter(
            emis_plot["lon"],
            emis_plot["lat"],
            c=vals,
            s=8,
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)
        )
        plt.colorbar(sc, label="NH3 [kg/km²/yr] (log scale)")
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("NH3 emission intensity over LE cells")
        plt.grid(True)
        plt.savefig(PLOT_DIR / "Emission_intensity_kg_km2_yr.png", dpi=200)
        plt.close()


        # Build full LE grid in kg/km²/yr
        emis_grid_km2 = build_full_le_grid(emis_cell, "nh3_kg_km2_yr", fill_value=0.0)

        vals = emis_grid_km2[emis_grid_km2 > 0]
        vmin = np.percentile(vals, 5)
        vmax = np.percentile(vals, 99)

        lon_edges_le = GRID_WEST + np.arange(GRID_NX + 1) * GRID_DLON
        lat_edges_le = GRID_SOUTH + np.arange(GRID_NY + 1) * GRID_DLAT

        emis_grid_plot = np.ma.masked_where(emis_grid_km2 <= 0, emis_grid_km2)

        plt.figure(figsize=(7, 8))
        pcm = plt.pcolormesh(
            lon_edges_le,
            lat_edges_le,
            emis_grid_plot,
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
            shading="flat",
            edgecolors="none",
            linewidth=0,
            antialiased=False
        )
        plt.colorbar(pcm, label="NH3 [kg/km²/yr] (log scale)")
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("NH3 emission intensity over LE grid")
        plt.grid(False)   # IMPORTANT
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "Emission_intensity_LE_conservative.png", dpi=300)
        plt.close()

#Trying with imshowinstead of pmesh


        plt.figure(figsize=(7, 8))
        im = plt.imshow(
            emis_grid_plot,
            origin="lower",
            extent=[GRID_WEST, GRID_EAST, GRID_SOUTH, GRID_NORTH],
            aspect="auto",
            interpolation="nearest",
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
            cmap="viridis"
        )
        plt.colorbar(im, label="NH3 [kg/km²/yr] (log scale)")
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("NH3 emission intensity over LE grid")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "Emission_intensity_LE_conservative_imshow.png", dpi=300)
        plt.close()


        emis_plot = emis_cell.copy()
        emis_plot["lon"] = GRID_WEST + (emis_plot["ix"] + 0.5) * GRID_DLON
        emis_plot["lat"] = GRID_SOUTH + (emis_plot["iy"] + 0.5) * GRID_DLAT

        emis_plot = emis_plot[emis_plot["nh3_kg_km2_yr"] > 0].copy()

        vals = emis_plot["nh3_kg_km2_yr"]
        vmin = np.percentile(vals, 5)
        vmax = np.percentile(vals, 99)

        plt.figure(figsize=(6, 8))
        sc = plt.scatter(
            emis_plot["lon"],
            emis_plot["lat"],
            c=vals,
            s=10,
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)
        )
        plt.colorbar(sc, label="NH3 [kg/km²/yr] (log scale)")
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("NH3 emission intensity over LE cells (CAMS→LE)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "Emission_intensity_CAMS_to_LE.png", dpi=200)
        plt.close()


        # ------------------------------------------------------------
        # FULL LE GRID RASTER PLOT (better than scatter)
        # ------------------------------------------------------------

        # Build full grid in kg/km²/yr
        emis_grid_km2 = build_full_le_grid(emis_cell, "nh3_kg_km2_yr", fill_value=0.0)


        '''
        checking whether the actual grid looks normal
        '''


        mask_grid = (emis_grid_km2 > 0).astype(int)

        plt.figure(figsize=(7, 8))
        plt.imshow(
            mask_grid,
            origin="lower",
            extent=[GRID_WEST, GRID_EAST, GRID_SOUTH, GRID_NORTH],
            aspect="auto",
            interpolation="nearest",
            cmap="Greys"
        )
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("LE grid cells with nonzero emissions (binary mask)")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "Emission_mask_binary.png", dpi=300)
        plt.close()


        # Optional: also build comparable grid in kg/m²/yr
        emis_grid_m2 = build_full_le_grid(emis_cell, "nh3_kg_m2_yr", fill_value=0.0)

        #from scipy.ndimage import gaussian_filter

        # emis_grid_km2_smooth = gaussian_filter(emis_grid_km2, sigma=1)
        # emis_grid_m2_smooth = gaussian_filter(emis_grid_m2, sigma=1)
        # #Then plot emis_grid_km2_smooth instead of emis_grid_km2.


        # Diagnostics
        print("Full LE grid shape:", emis_grid_km2.shape)
        print("Nonzero cells in full grid:", np.count_nonzero(emis_grid_km2 > 0))
        print("Zero cells in full grid:", np.count_nonzero(emis_grid_km2 == 0))

        # -------------------------
        # Plot 1: kg/km²/yr
        # -------------------------
        vals = emis_grid_km2[emis_grid_km2 > 0]

        if len(vals) > 0:
            vmin = np.percentile(vals, 5)
            vmax = np.percentile(vals, 99)

            #From here P_mesh plot was applied. Till plt.close()
            lon_edges_le = GRID_WEST + np.arange(GRID_NX + 1) * GRID_DLON
            lat_edges_le = GRID_SOUTH + np.arange(GRID_NY + 1) * GRID_DLAT

            emis_grid_plot = np.ma.masked_where(emis_grid_km2 <= 0, emis_grid_km2)

            plt.figure(figsize=(7, 8))
            pcm = plt.pcolormesh(
                lon_edges_le,
                lat_edges_le,
                emis_grid_plot,
                norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                shading="flat"
            )
            plt.colorbar(pcm, label="NH3 [kg/km²/yr] (log scale)")
            plt.xlabel("Longitude [deg]")
            plt.ylabel("Latitude [deg]")
            plt.title("NH3 emission intensity over LE grid")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "P_Mesh_Emission_intensity_fullgrid_kg_km2_yr.png", dpi=200)
            plt.close()


        # -------------------------
        # Plot 2: kg/m²/yr (for comparison with supervisor)
        # -------------------------
        vals_m2 = emis_grid_m2[emis_grid_m2 > 0]



        if len(vals_m2) > 0:

            vmin_m2 = np.percentile(vals_m2, 5)
            vmax_m2 = np.percentile(vals_m2, 99)
            #From here P_Mesh
            emis_grid_m2_plot = np.ma.masked_where(emis_grid_m2 <= 0, emis_grid_m2)

            plt.figure(figsize=(7, 8))
            pcm = plt.pcolormesh(
                lon_edges_le,
                lat_edges_le,
                emis_grid_m2_plot,
                norm=mcolors.LogNorm(vmin=vmin_m2, vmax=vmax_m2),
                shading="flat"
            )
            plt.colorbar(pcm, label="NH3 [kg/m²/yr]")
            plt.xlabel("Longitude [deg]")
            plt.ylabel("Latitude [deg]")
            plt.title("NH3 annual emissions over LE grid (kg/m²/yr)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "P_Mesh_Emission_intensity_fullgrid_kg_m2_yr.png", dpi=200)
            plt.close()


        # Convert grid indices to lon/lat if you want geographic plotting
        emis_plot = emis_cell.copy()
        emis_plot["lon"] = GRID_WEST + emis_plot["ix"] * GRID_DLON
        emis_plot["lat"] = GRID_SOUTH + emis_plot["iy"] * GRID_DLAT

        # Avoid log(0): keep only strictly positive values for log scaling
        emis_plot = emis_plot[emis_plot["nh3_kg_yr"] > 0].copy()

        plt.figure(figsize=(6, 8))

        sc = plt.scatter(
            emis_plot["ix"],   # or emis_plot["ix"]
            emis_plot["iy"],   # or emis_plot["iy"]
            c=emis_plot["nh3_kg_km2_yr"],
            #c = df["nh3_kg_km2_yr"],
            s=8,
            norm=mcolors.LogNorm(
                vmin=emis_plot["nh3_kg_km2_yr"].min(),
                vmax=emis_plot["nh3_kg_km2_yr"].max()
            )
        )

        plt.colorbar(sc, label="NH3 [kg/km2/yr] (log scale)")
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title("Emission distribution (log scale)")
        plt.grid(True)
        plt.savefig(PLOT_DIR / "Emission_distribution_logscale_grids.png", dpi=200)
        plt.close()


        plt.scatter(sampling_summary["ix"], sampling_summary["iy"], s=2)
        plt.title("Sampled cells")
        plt.savefig(PLOT_DIR /"Sampled_cells.png")
        plt.close()


        plt.scatter(emis_cell["ix"], emis_cell["iy"], s=2)
        plt.title("Emission cells")
        plt.savefig(PLOT_DIR / "Emission_cells.png")
        plt.close()


        plt.scatter(sampling_summary["ix"], sampling_summary["iy"],
                    c=sampling_summary["N_obs"], s=5)
        plt.colorbar(label="N_obs")
        plt.title("Observation count per grid cell")
        plt.savefig(PLOT_DIR / "Observation_count per grid cell.png")  # How many times each grid cell was observed.
        plt.close()


        plt.scatter(sampling_summary["ix"], sampling_summary["iy"],
                    c=sampling_summary["dt_median_s"]/86400, s=5)
        plt.colorbar(label="Median revisit time [days]")
        plt.title("Median revisit time per grid cell")        #“How often does the orbit come back to each locatio
        plt.savefig(PLOT_DIR / "Median revisits.png")
        plt.close()

        plt.scatter(observability_df["ix"], observability_df["iy"],
                    c=observability_df["time_to_detect_days"], s=5)
        plt.colorbar(label="Time to detect [days]")
        plt.title("NH3 detectability time per grid cell")
        plt.savefig(PLOT_DIR / "Detectability time.png")        #How quickly can this orbit detect NH₃ emissions, spatially across NL
        plt.close()

        if "time_to_detect_days_paper_effN" in observability_df.columns:
            plt.scatter(
                observability_df["ix"],
                observability_df["iy"],
                c=observability_df["time_to_detect_days_paper_effN"],
                s=5
            )
            plt.colorbar(label="Paper-style effective-N time to detect [days]")
            plt.title("NH3 detectability time per grid cell - paper-style effective N")
            plt.savefig(PLOT_DIR / "Detectability_time_paper_effN.png", dpi=200)      # Paper style time_to_detect plots
            plt.close()


        obs_nonzero = observability_df[observability_df["nh3_kg_yr"] > 0].copy()

        # Remove infinite values
        obs_nonzero = obs_nonzero[np.isfinite(obs_nonzero["time_to_detect_days"])]

        if len(obs_nonzero)>0:

            #From here P_Mesh was applied till plt.close()
            # Build full LE detectability grid
            detect_grid = np.full((GRID_NY, GRID_NX), np.nan)

            for _, row in obs_nonzero.iterrows():
                detect_grid[int(row["iy"]), int(row["ix"])] = row["time_to_detect_days"]

            # Cap values for display
            detect_grid = np.where(np.isfinite(detect_grid), np.minimum(detect_grid, 3650), np.nan)

            # Mask NaNs
            detect_grid_plot = np.ma.masked_invalid(detect_grid)

            lon_edges_le = GRID_WEST + np.arange(GRID_NX + 1) * GRID_DLON
            lat_edges_le = GRID_SOUTH + np.arange(GRID_NY + 1) * GRID_DLAT

            plt.figure(figsize=(7, 8))
            pcm = plt.pcolormesh(
                lon_edges_le,
                lat_edges_le,
                detect_grid_plot,
                shading="flat"
            )
            plt.colorbar(pcm, label="Time to detect [days] (capped at 10 years)")
            plt.xlabel("Longitude [deg]")
            plt.ylabel("Latitude [deg]")
            plt.title("NH3 detectability (non-zero emission cells)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "P_Mesh_detectability_nonzero_cells_capped.png", dpi=200)
            plt.close()


            plot_vals = obs_nonzero["time_to_detect_days"].clip(upper=3650)  # 10 years
            plt.figure(figsize=(6, 6))

            sc = plt.scatter(
                obs_nonzero["ix"],
                obs_nonzero["iy"],
                c=plot_vals,
                s=5
            )

            plt.colorbar(sc, label="Time to detect [days] (capped at 10 years)")
            plt.title("NH3 detectability (non-zero emission cells)")
            plt.xlabel("ix (grid)")
            plt.ylabel("iy (grid)")

            plt.savefig(PLOT_DIR / "detectability_nonzero_cells_capped.png", dpi=200)
            plt.close()

            # From here P_Mesh till plt.close()

            plt.figure(figsize=(7, 8))
            pcm = plt.pcolormesh(
                lon_edges_le,
                lat_edges_le,
                detect_grid_plot,
                norm=mcolors.LogNorm(vmin=1, vmax=3650),
                shading="flat"
            )
            plt.colorbar(pcm, label="Time to detect [days] (capped at 10 years)")
            plt.xlabel("Longitude [deg]")
            plt.ylabel("Latitude [deg]")
            plt.title("NH3 detectability logscale (non-zero emission cells)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "P_Mesh_detectability_nonzero_cells_capped_logscale.png", dpi=200)
            plt.close()

            plot_vals = obs_nonzero["time_to_detect_days"].clip(upper=3650)  # 10 years
            plt.figure(figsize=(6, 6))

            sc = plt.scatter(
                obs_nonzero["ix"],
                obs_nonzero["iy"],
                c=plot_vals,
                s=5,
                norm=mcolors.LogNorm(vmin=1, vmax=3650)
            )

            plt.colorbar(sc, label="Time to detect [days] (capped at 10 years)")
            plt.title("NH3 detectability logscale (non-zero emission cells)")
            plt.xlabel("ix (grid)")
            plt.ylabel("iy (grid)")

            plt.savefig(PLOT_DIR / "detectability_nonzero_cells_capped_logscale.png", dpi=200)
            plt.close()


            plt.figure(figsize=(6, 6))

            sc = plt.scatter(
                obs_nonzero["ix"],
                obs_nonzero["iy"],
                c=obs_nonzero["time_to_detect_days"],
                s=10,
                norm=mcolors.LogNorm(vmin=1, vmax=1e4)
            )

            plt.colorbar(sc, label="Time to detect [days] (log scale)")
            plt.title("NH3 detectability (non-zero emission cells, log scale)")
            plt.xlabel("ix")
            plt.ylabel("iy")

            plt.savefig(PLOT_DIR / "detectability_logscale.png", dpi=200)
            plt.close()


            # Sort values
            detect_times = np.sort(obs_nonzero["time_to_detect_days"].values)

            # Cumulative fraction
            cdf = np.arange(1, len(detect_times)+1) / len(detect_times)

            plt.figure(figsize=(6, 4))
            plt.plot(detect_times, cdf)

            plt.xscale("log")
            plt.xlabel("Time to detect [days]")
            plt.ylabel("Fraction of emission cells")
            plt.title("CDF of NH3 detectability")

            plt.grid(True)
            plt.savefig(PLOT_DIR / "detectability_cdf.png", dpi=200)
            plt.close()
        else:
            print(f"{orbit_name}: No finite detectability values for non-zero emission cells")
    # ------------------------------------------------------------
    # 11) Orbit-level summary metrics row
    # ------------------------------------------------------------
    summary_row = summarize_orbit_case(orbit_name, sampling_summary, observability_df)

    # ------------------------------------------------------------
    # Add local-time and hourfactor-time diagnostics to orbit summary
    # ------------------------------------------------------------
    summary_row.update(local_time_metrics)
    summary_row.update(hourfactor_time_metrics)
    summary_row.update(peak_hourfactor_metrics)

    time_total = time.time() - time1

    summary_row["runtime_s"] = float(time_total)
    summary_row["candidate_windows"] = int(len(candidate_windows))
    summary_row["nl_pixels"] = int(len(polys_nl))

    print(f"{orbit_name} completed in {time_total:.1f} s")
    print(summary_row)

    return summary_row


def main():
    """Run configured orbit cases and write comparison outputs."""
    comparison_rows = []

    for case in orbit_cases:
        row = run_orbit_case(case)
        if row is not None:
            comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    if not comparison_df.empty:
        # comparison_df.to_csv("./outputs/orbit_comparison_summary.csv", index=False)
        # comparison_df.to_parquet("./outputs/orbit_comparison_summary.parquet", index=False)

        from datetime import datetime

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        comparison_df.to_csv(
            f"./outputs/orbit_comparison_summary_{run_id}.csv",
            index=False
        )

        comparison_df.to_parquet(
            f"./outputs/orbit_comparison_summary_{run_id}.parquet",
            index=False
        )


        print("\n" + "=" * 80)
        print("ORBIT COMPARISON SUMMARY")
        print("=" * 80)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)

        print(comparison_df)

        comparison_cols = [
            "orbit_name",

            # Sampling / coverage
            "sampled_cells",
            "emission_cells_merged",
            "finite_detectable_emission_cells",
            "frac_finite_emission_cells",

            # Pass-level observation counts
            "median_N_obs_all",
            "median_N_obs_eff_all",
            "median_N_obs_signal_eff_all",
            "median_hourfactor_all",
            "median_signal_to_quality_ratio_all",
            "median_seasonal_factor_all",
            "median_combined_emission_factor_all",

            "median_SZA_reduction_all",

            # Pixel-count-per-pass diagnostics
            "median_N_raw_per_pass_mean",
            "median_N_eff_per_pass_mean",
            "median_N_obs_signal_eff_emission_cells",
            "median_hourfactor_emission_cells",
            "median_signal_to_quality_ratio_emission_cells",
            "median_seasonal_factor_emission_cells",
            "median_combined_emission_factor_emission_cells",

            # SZA / TC / revisit diagnostics
            "median_sza_min_deg_all",
            "median_sza_median_deg_all",

            "median_tc_K_all",
            "median_w_tc_all",
            "median_w_quality_all",

            "median_tcc_all",
            "median_w_cloud_all",
            "median_tcc_emission_cells",
            "median_w_cloud_emission_cells",
            "median_revisit_days_all",

            # Local solar time diagnostics
            "local_time_pixel_peak_hour",
            "local_time_scanline_peak_hour",
            "local_time_pixel_circular_mean_hour",
            "local_time_scanline_circular_mean_hour",
            "local_time_pixel_concentration",
            "local_time_scanline_concentration",
            "frac_pixel_local_06_10",
            "frac_pixel_local_10_14",
            "frac_pixel_local_14_18",
            "frac_scanline_local_06_10",
            "frac_scanline_local_10_14",
            "frac_scanline_local_14_18",

            # Hourfactor by local time diagnostics
            "mean_observed_hourfactor",
            "median_observed_hourfactor",
            "max_hourly_mean_hourfactor",
            "hour_of_max_mean_hourfactor",

            # Relative-to-peak hourfactor diagnostics
            "mean_relative_to_peak_hourfactor",
            "median_relative_to_peak_hourfactor",
            "p10_relative_to_peak_hourfactor",
            "p90_relative_to_peak_hourfactor",
            "frac_obs_relative_to_peak_ge_0p50",
            "frac_obs_relative_to_peak_ge_0p75",
            "frac_obs_relative_to_peak_ge_0p90",
            "mean_daily_peak_hourfactor_sampled_cells",
            "mean_observed_hourfactor_for_peak_diag",

            # Current method: observation-rate method
            "median_detect_days_finite",
            "p10_detect_days_finite",
            "p90_detect_days_finite",
            "weighted_mean_detect_days",
            "frac_detect_365d",

            # Paper-style raw N method
            "median_detect_days_paper_rawN",
            "p10_detect_days_paper_rawN",
            "p90_detect_days_paper_rawN",
            "weighted_mean_detect_days_paper_rawN",
            "frac_detect_365d_paper_rawN",

            # Paper-style effective N method
            "median_detect_days_paper_effN",
            "p10_detect_days_paper_effN",
            "p90_detect_days_paper_effN",
            "weighted_mean_detect_days_paper_effN",
            "frac_detect_365d_paper_effN",

            # Paper-style signal-effective method: SZA + TC + cloud + diurnal emissions
            "median_detect_days_paper_signalEffN",
            "p10_detect_days_paper_signalEffN",
            "p90_detect_days_paper_signalEffN",
            "weighted_mean_detect_days_paper_signalEffN",
            "frac_detect_30d_among_finite_paper_signalEffN",
            "frac_detect_90d_among_finite_paper_signalEffN",
            "frac_detect_365d_among_finite_paper_signalEffN",
        ]

        available_cols = [c for c in comparison_cols if c in comparison_df.columns]

        print("\n" + "=" * 80)
        print("COMPACT ORBIT COMPARISON")
        print("=" * 80)
        print(comparison_df[available_cols].sort_values("median_detect_days_finite"))


    # ============================================================
    # FINAL MULTI-ORBIT RESULTS FIGURES
    # ============================================================

    if GENERATE_MULTI_ORBIT_RESULTS and ORBIT_RESULTS_CACHE:
        try:
            generate_multi_orbit_results(
                output_root="./outputs"
            )
        except Exception as exc:
            print(
                f"[WARNING] Final multi-orbit plotting failed: {exc}"
            )
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
