#!/usr/bin/env python
# coding: utf-8

# # Extract datapoints with correct coordinates uding State Shapefile

# In[ ]:


import os
import re
from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================
INPUT_CSV_DIR = Path(r"H:\shc_data\SHC_data")
STATE_SHP = Path(
    r"H:\Carbon_Data\Shapefile_Admin_India_4Nov2023_PU-20251223T060820Z-3-001"
    r"\Shapefile_Admin_India_4Nov2023_PU\STATE_BOUNDARY.shp"
)
OUTPUT_DIR = Path(r"H:\shc_data\SHC_clean_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 200000   # adjust if needed


# ============================================================
# HELPERS
# ============================================================
def normalize_state_name(name):
    """Standardize state names for matching CSV files with shapefile names."""
    if pd.isna(name):
        return ""

    s = str(name).strip().upper()
    s = re.sub(r"\s+", " ", s)

    alias_map = {
        "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
        "JAMMU & KASHMIR": "JAMMU & KASHMIR",
        "DADRA & NAGAR HAVELI AND DAMAN & DIU": "DADRA & NAGAR HAVELI & DAMAN & DIU",
        "DADRA & NAGAR HAVELI & DAMAN & DIU": "DADRA & NAGAR HAVELI & DAMAN & DIU",
    }

    return alias_map.get(s, s)


def get_state_geometry_map(shp_path):
    """
    Read shapefile, standardize state names, dissolve duplicates,
    and return a dict: normalized_state_name -> merged geometry
    """
    states = gpd.read_file(shp_path)

    if "STATE" not in states.columns:
        raise ValueError("Column 'STATE' not found in shapefile attribute table.")

    states = states[states.geometry.notna() & ~states.geometry.is_empty].copy()
    states["state_std"] = states["STATE"].apply(normalize_state_name)

    # Dissolve duplicate polygons for same state
    states_dissolved = states.dissolve(by="state_std", as_index=False)

    return states_dissolved


# ============================================================
# LOAD AND PREPARE STATE BOUNDARIES
# ============================================================
print("Reading state boundary shapefile...")
states_gdf = get_state_geometry_map(STATE_SHP)

if states_gdf.crs is None:
    raise ValueError("State boundary shapefile has no CRS defined. Please define CRS before running.")

state_geom_map = dict(zip(states_gdf["state_std"], states_gdf.geometry))

print(f"Loaded {len(state_geom_map)} dissolved state geometries.\n")


# ============================================================
# PROCESS EACH STATE CSV
# ============================================================
csv_files = sorted(INPUT_CSV_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_CSV_DIR}")

grand_total = 0
grand_kept = 0
grand_removed_outside = 0
grand_invalid_xy = 0

for csv_path in csv_files:
    state_name_from_file = csv_path.stem
    state_std = normalize_state_name(state_name_from_file)

    print(f"\nProcessing: {csv_path.name}")

    if state_std not in state_geom_map:
        print(f"[WARNING] No matching state polygon found for: {state_name_from_file}")
        continue

    state_geom = state_geom_map[state_std]
    out_csv = OUTPUT_DIR / csv_path.name

    if out_csv.exists():
        out_csv.unlink()

    total_rows = 0
    kept_rows = 0
    removed_outside = 0
    invalid_xy_rows = 0
    first_write = True
    original_columns = None

    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE, low_memory=False):
        if original_columns is None:
            original_columns = list(chunk.columns)

        total_rows += len(chunk)

        if "latitude" not in chunk.columns or "longitude" not in chunk.columns:
            raise ValueError(f"'latitude' or 'longitude' column missing in {csv_path.name}")

        # Force lat/lon numeric
        chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
        chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")

        valid_xy_mask = chunk["latitude"].notna() & chunk["longitude"].notna()
        invalid_xy_rows += (~valid_xy_mask).sum()

        chunk_valid = chunk.loc[valid_xy_mask].copy()

        if chunk_valid.empty:
            continue

        # Create point GeoDataFrame in WGS84
        points_gdf = gpd.GeoDataFrame(
            chunk_valid,
            geometry=gpd.points_from_xy(chunk_valid["longitude"], chunk_valid["latitude"]),
            crs="EPSG:4326"
        )

        # Reproject to shapefile CRS if needed
        if points_gdf.crs != states_gdf.crs:
            points_gdf = points_gdf.to_crs(states_gdf.crs)

        # Keep points that intersect state polygon
        # (includes points exactly on the boundary)
        inside_mask = points_gdf.geometry.intersects(state_geom)

        clean_chunk = points_gdf.loc[inside_mask].drop(columns="geometry")
        outside_count = len(points_gdf) - len(clean_chunk)

        kept_rows += len(clean_chunk)
        removed_outside += outside_count

        if not clean_chunk.empty:
            clean_chunk.to_csv(
                out_csv,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig"
            )
            first_write = False

    # If nothing was written, create an empty CSV with original columns
    if first_write and original_columns is not None:
        pd.DataFrame(columns=original_columns).to_csv(
            out_csv, index=False, encoding="utf-8-sig"
        )

    print(f"[DONE] {state_name_from_file}")
    print(f"       Total rows        : {total_rows}")
    print(f"       Kept rows         : {kept_rows}")
    print(f"       Removed outside   : {removed_outside}")
    print(f"       Invalid lat/lon   : {invalid_xy_rows}")
    print(f"       Output            : {out_csv}")

    grand_total += total_rows
    grand_kept += kept_rows
    grand_removed_outside += removed_outside
    grand_invalid_xy += invalid_xy_rows


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)
print(f"Grand total rows      : {grand_total}")
print(f"Grand kept rows       : {grand_kept}")
print(f"Grand removed outside : {grand_removed_outside}")
print(f"Grand invalid lat/lon : {grand_invalid_xy}")
print(f"Cleaned files saved in: {OUTPUT_DIR}")


# # Extracting only cropland, rangeland, trees and bareground points using LULC raster

# In[ ]:


import numpy as np
import pandas as pd
from pathlib import Path

import rasterio
from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================
INPUT_CSV_DIR = Path(r"H:\shc_data\SHC_clean_data")
LULC_RASTER = Path(r"H:\india_lulc\combined.tif")
OUTPUT_DIR = Path(r"H:\shc_data\SHC_farm_rangeland_trees_bareground")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 200000

# ============================================================
# LULC CLASS MAP
# ============================================================
KEEP_CLASSES = {
    2: "Trees",
    5: "Crops",
    8: "Bare ground",
    11: "Rangeland",
}

# ============================================================
# READ RASTER INFO
# ============================================================
with rasterio.open(LULC_RASTER) as src:
    raster_crs = src.crs
    raster_count = src.count
    raster_nodata = src.nodata

print(f"Raster CRS      : {raster_crs}")
print(f"Raster bands    : {raster_count}")
print(f"Raster nodata   : {raster_nodata}")

# ------------------------------------------------------------
# IMPORTANT:
# If combined.tif has multiple yearly bands, this code uses the
# LAST band by default (usually latest year if stacked that way).
# Change BAND_INDEX manually if needed.
# ------------------------------------------------------------
BAND_INDEX = raster_count if raster_count >= 1 else 1
print(f"Using band      : {BAND_INDEX}")

# Transformer from WGS84 lat/lon to raster CRS
transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

# ============================================================
# PROCESS EACH STATE CSV
# ============================================================
csv_files = sorted(INPUT_CSV_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_CSV_DIR}")

grand_total = 0
grand_kept = 0

with rasterio.open(LULC_RASTER) as src:
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        out_csv = OUTPUT_DIR / csv_file.name
        if out_csv.exists():
            out_csv.unlink()

        total_rows = 0
        kept_rows = 0
        invalid_xy_rows = 0
        first_write = True
        original_columns = None

        for chunk in pd.read_csv(csv_file, chunksize=CHUNKSIZE, low_memory=False):
            if original_columns is None:
                original_columns = list(chunk.columns)

            total_rows += len(chunk)

            if "latitude" not in chunk.columns or "longitude" not in chunk.columns:
                raise ValueError(f"'latitude' or 'longitude' missing in {csv_file.name}")

            chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
            chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")

            valid_mask = chunk["latitude"].notna() & chunk["longitude"].notna()
            invalid_xy_rows += (~valid_mask).sum()

            chunk_valid = chunk.loc[valid_mask].copy()
            if chunk_valid.empty:
                continue

            # Convert lon/lat -> raster CRS
            xs, ys = transformer.transform(
                chunk_valid["longitude"].to_numpy(),
                chunk_valid["latitude"].to_numpy()
            )

            coords = list(zip(xs, ys))

            # Sample raster values
            sampled = np.array([val[0] for val in src.sample(coords, indexes=BAND_INDEX)])

            # Keep only required classes
            keep_mask = np.isin(sampled, list(KEEP_CLASSES.keys()))
            filtered = chunk_valid.loc[keep_mask].copy()

            if filtered.empty:
                continue

            filtered["LULC_VALUE"] = sampled[keep_mask]
            filtered["LULC_CLASS"] = filtered["LULC_VALUE"].map(KEEP_CLASSES)

            # Put LULC columns near the front if possible
            preferred_front = [
                "state", "district", "village", "tehsil",
                "latitude", "longitude", "date", "period",
                "LULC_VALUE", "LULC_CLASS"
            ]
            front_cols = [c for c in preferred_front if c in filtered.columns]
            other_cols = [c for c in filtered.columns if c not in front_cols]
            filtered = filtered[front_cols + other_cols]

            filtered.to_csv(
                out_csv,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig"
            )
            first_write = False

            kept_rows += len(filtered)

        # Create empty output if nothing matched
        if first_write and original_columns is not None:
            empty_df = pd.DataFrame(columns=original_columns + ["LULC_VALUE", "LULC_CLASS"])
            empty_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print(f"[DONE] {csv_file.name}")
        print(f"       Total rows      : {total_rows}")
        print(f"       Kept farm rows  : {kept_rows}")
        print(f"       Invalid lat/lon : {invalid_xy_rows}")
        print(f"       Output          : {out_csv}")

        grand_total += total_rows
        grand_kept += kept_rows

print("\n" + "=" * 60)
print("LULC FILTERING COMPLETED")
print("=" * 60)
print(f"Grand total input rows : {grand_total}")
print(f"Grand kept farm rows   : {grand_kept}")
print(f"Output folder          : {OUTPUT_DIR}")


# # Removing points without date info

# In[ ]:


from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================
INPUT_DIR = Path(r"H:\shc_data\SHC_farm_rangeland_trees_bareground")
OUTPUT_DIR = Path(r"H:\shc_data\SHC_crop_range_bare_with_date")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 200000

# ============================================================
# PROCESS
# ============================================================
csv_files = sorted(INPUT_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {INPUT_DIR}")

grand_total = 0
grand_kept = 0
grand_removed = 0

for csv_file in csv_files:
    print(f"\nProcessing: {csv_file.name}")

    out_csv = OUTPUT_DIR / csv_file.name
    if out_csv.exists():
        out_csv.unlink()

    total_rows = 0
    kept_rows = 0
    removed_rows = 0
    first_write = True
    original_columns = None

    for chunk in pd.read_csv(csv_file, chunksize=CHUNKSIZE, low_memory=False):
        if original_columns is None:
            original_columns = list(chunk.columns)

        total_rows += len(chunk)

        if "date" not in chunk.columns:
            raise ValueError(f"'date' column not found in file: {csv_file.name}")

        # Remove rows where date is NaN, empty, or only whitespace
        date_series = chunk["date"].astype(str).str.strip()
        valid_mask = chunk["date"].notna() & (date_series != "") & (date_series.str.lower() != "nan")

        clean_chunk = chunk.loc[valid_mask].copy()

        kept_rows += len(clean_chunk)
        removed_rows += len(chunk) - len(clean_chunk)

        if not clean_chunk.empty:
            clean_chunk.to_csv(
                out_csv,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig"
            )
            first_write = False

    # If no valid rows remain, create empty CSV with same columns
    if first_write and original_columns is not None:
        pd.DataFrame(columns=original_columns).to_csv(
            out_csv, index=False, encoding="utf-8-sig"
        )

    print(f"[DONE] {csv_file.name}")
    print(f"       Total rows   : {total_rows}")
    print(f"       Kept rows    : {kept_rows}")
    print(f"       Removed rows : {removed_rows}")
    print(f"       Output       : {out_csv}")

    grand_total += total_rows
    grand_kept += kept_rows
    grand_removed += removed_rows

print("\n" + "=" * 60)
print("DATE FILTERING COMPLETED")
print("=" * 60)
print(f"Grand total rows   : {grand_total}")
print(f"Grand kept rows    : {grand_kept}")
print(f"Grand removed rows : {grand_removed}")
print(f"Output folder      : {OUTPUT_DIR}")


# # Shapefile generator for visualisation and geospatial analysis

# In[ ]:


from pathlib import Path
import csv

import numpy as np
import numpy.core.records as rec
np.rec = rec

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# PATHS
# ============================================================
INPUT_DIR = Path(r"H:\shc_data\SHC_crop_range_bare_with_date")
OUTPUT_DIR = Path(r"H:\shc_data\SHC_all_with_date_shapefile")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_GPKG = OUTPUT_DIR / "SHC_all_with_date_all_states.gpkg"
LAYER_NAME = "shc_all_points"

# ============================================================
# SETTINGS
# ============================================================
CHUNKSIZE = 100000
TARGET_CRS = "EPSG:4326"

# GeoPackage reserved/internal field names to avoid
RESERVED_NAMES = {"fid", "ogc_fid"}


# ============================================================
# HELPERS
# ============================================================
def read_csv_header(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return header


def sanitize_column_names(columns):
    """
    Rename reserved GeoPackage field names like FID / OGC_FID.
    Keeps other names unchanged unless duplicate names appear.
    """
    rename_map = {}
    used = set()

    for col in columns:
        col_str = str(col).strip()
        low = col_str.lower()

        if low in RESERVED_NAMES:
            new_name = f"{col_str}_attr"
        else:
            new_name = col_str

        # ensure uniqueness after renaming
        base = new_name
        i = 1
        while new_name.lower() in used:
            new_name = f"{base}_{i}"
            i += 1

        rename_map[col] = new_name
        used.add(new_name.lower())

    return rename_map


# ============================================================
# MAIN
# ============================================================
csv_files = sorted(INPUT_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

if OUT_GPKG.exists():
    OUT_GPKG.unlink()

# ------------------------------------------------------------
# Collect all columns from all CSV files
# ------------------------------------------------------------
all_columns = []
for csv_path in csv_files:
    header = read_csv_header(csv_path)
    all_columns.extend(header)

seen = set()
merged_columns = []
for c in all_columns:
    if c not in seen:
        seen.add(c)
        merged_columns.append(c)

if "longitude" not in merged_columns or "latitude" not in merged_columns:
    raise ValueError("Required columns 'longitude' and/or 'latitude' not found in input CSV files.")

# Rename reserved fields if present
rename_map = sanitize_column_names(merged_columns)

attribute_columns = [c for c in merged_columns if c not in ["longitude", "latitude"]]

first_write = True
total_rows = 0
written_rows = 0
invalid_xy_rows = 0

for csv_path in csv_files:
    print(f"\nProcessing: {csv_path.name}")

    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE, low_memory=False):
        total_rows += len(chunk)

        # ensure all columns exist
        for col in merged_columns:
            if col not in chunk.columns:
                chunk[col] = pd.NA

        chunk = chunk[merged_columns].copy()

        # lon/lat numeric
        chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
        chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")

        valid_mask = (
            chunk["longitude"].notna() &
            chunk["latitude"].notna() &
            np.isfinite(chunk["longitude"]) &
            np.isfinite(chunk["latitude"]) &
            (chunk["longitude"] >= -180) & (chunk["longitude"] <= 180) &
            (chunk["latitude"] >= -90) & (chunk["latitude"] <= 90)
        )

        invalid_xy_rows += (~valid_mask).sum()
        chunk = chunk.loc[valid_mask].copy()

        if chunk.empty:
            continue

        # convert non-coordinate attributes to string for stable schema
        for col in attribute_columns:
            chunk[col] = chunk[col].fillna("").astype(str)

        chunk["longitude"] = chunk["longitude"].astype(float)
        chunk["latitude"] = chunk["latitude"].astype(float)

        # rename reserved columns
        chunk = chunk.rename(columns=rename_map)

        geometry = [Point(xy) for xy in zip(chunk["longitude"], chunk["latitude"])]
        gdf = gpd.GeoDataFrame(chunk, geometry=geometry, crs=TARGET_CRS)

        # write with Fiona engine
        gdf.to_file(
            OUT_GPKG,
            layer=LAYER_NAME,
            driver="GPKG",
            engine="fiona",
            mode="w" if first_write else "a",
            index=False
        )

        first_write = False
        written_rows += len(gdf)
        print(f"  wrote {len(gdf)} features")

print("\n" + "=" * 60)
print("GEOPACKAGE CREATION COMPLETED")
print("=" * 60)
print(f"Total input rows     : {total_rows}")
print(f"Invalid lat/lon rows : {invalid_xy_rows}")
print(f"Written point rows   : {written_rows}")
print(f"Output GeoPackage    : {OUT_GPKG}")
print(f"Layer name           : {LAYER_NAME}")

print("\nReserved-column rename map used:")
for old, new in rename_map.items():
    if old != new:
        print(f"  {old}  -->  {new}")

