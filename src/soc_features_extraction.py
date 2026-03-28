#!/usr/bin/env python
# coding: utf-8

# # Climate Data Extraction

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================
INPUT_DIR = Path(r"H:\shc_data\SHC_crop_range_bare_with_date")
CLIMATE_RASTER = Path(r"H:\Carbon_Data\koppen climate data raw\koppen_geiger_tif\1991_2020\koppen_geiger_0p00833333.tif")
OUTPUT_DIR = Path(r"H:\shc_data\SHC_with_climate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 200000
BAND_INDEX = 1   # Köppen-Geiger raster is expected to be single-band


# ============================================================
# KÖPPEN-GEIGER LOOKUP
# ============================================================
KG_SUBCLASS_MAP = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BWh",
    5: "BWk",
    6: "BSh",
    7: "BSk",
    8: "Csa",
    9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF",
}

KG_BROAD_MAP = {
    1: "Tropical",
    2: "Tropical",
    3: "Tropical",
    4: "Arid",
    5: "Arid",
    6: "Arid",
    7: "Arid",
    8: "Temperate",
    9: "Temperate",
    10: "Temperate",
    11: "Temperate",
    12: "Temperate",
    13: "Temperate",
    14: "Temperate",
    15: "Temperate",
    16: "Temperate",
    17: "Cold",
    18: "Cold",
    19: "Cold",
    20: "Cold",
    21: "Cold",
    22: "Cold",
    23: "Cold",
    24: "Cold",
    25: "Cold",
    26: "Cold",
    27: "Cold",
    28: "Cold",
    29: "Polar",
    30: "Polar",
}


# ============================================================
# MAIN
# ============================================================
csv_files = sorted(INPUT_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {INPUT_DIR}")

with rasterio.open(CLIMATE_RASTER) as src:
    raster_crs = src.crs
    raster_nodata = src.nodata
    raster_count = src.count

print(f"Raster CRS    : {raster_crs}")
print(f"Raster bands  : {raster_count}")
print(f"Raster nodata : {raster_nodata}")

if raster_count < BAND_INDEX:
    raise ValueError(f"Raster does not have band {BAND_INDEX}")

# transformer from WGS84 point coordinates to raster CRS
transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

grand_total = 0
grand_invalid_xy = 0
grand_valid_climate = 0
grand_nodata_climate = 0

with rasterio.open(CLIMATE_RASTER) as src:
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        out_csv = OUTPUT_DIR / csv_file.name
        if out_csv.exists():
            out_csv.unlink()

        total_rows = 0
        invalid_xy_rows = 0
        valid_climate_rows = 0
        nodata_climate_rows = 0
        first_write = True
        original_columns = None

        for chunk in pd.read_csv(csv_file, chunksize=CHUNKSIZE, low_memory=False):
            if original_columns is None:
                original_columns = list(chunk.columns)

            total_rows += len(chunk)

            if "longitude" not in chunk.columns or "latitude" not in chunk.columns:
                raise ValueError(f"'longitude' or 'latitude' missing in {csv_file.name}")

            # ensure numeric lon/lat
            chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
            chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")

            valid_xy_mask = (
                chunk["longitude"].notna() &
                chunk["latitude"].notna() &
                np.isfinite(chunk["longitude"]) &
                np.isfinite(chunk["latitude"]) &
                (chunk["longitude"] >= -180) & (chunk["longitude"] <= 180) &
                (chunk["latitude"] >= -90) & (chunk["latitude"] <= 90)
            )

            invalid_xy_rows += (~valid_xy_mask).sum()

            # default output columns
            chunk["KG_VALUE"] = pd.NA
            chunk["KG_SUBCLASS"] = pd.NA
            chunk["KG_BROAD"] = pd.NA

            chunk_valid = chunk.loc[valid_xy_mask].copy()

            if not chunk_valid.empty:
                xs, ys = transformer.transform(
                    chunk_valid["longitude"].to_numpy(),
                    chunk_valid["latitude"].to_numpy()
                )

                coords = list(zip(xs, ys))
                sampled = np.array([v[0] for v in src.sample(coords, indexes=BAND_INDEX)])

                # treat nodata / invalid codes as missing
                valid_class_mask = np.isin(sampled, list(KG_SUBCLASS_MAP.keys()))

                valid_climate_rows += valid_class_mask.sum()
                nodata_climate_rows += (~valid_class_mask).sum()

                valid_idx = chunk_valid.index[valid_class_mask]
                valid_vals = sampled[valid_class_mask].astype(int)

                chunk.loc[valid_idx, "CLIMATE_VALUE"] = valid_vals
                chunk.loc[valid_idx, "CLIMATE_SUBCLASS"] = [KG_SUBCLASS_MAP[v] for v in valid_vals]
                chunk.loc[valid_idx, "CLIMATE_CLASS"] = [KG_BROAD_MAP[v] for v in valid_vals]

            # reorder columns: place climate columns near date/period if present
            preferred_front = [
                "state", "district", "village", "tehsil",
                "latitude", "longitude", "date", "period",
                "LULC_VALUE", "LULC_CLASS",
                "CLIMATE_VALUE", "CLIMATE_SUBCLASS", "CLIMATE_CLASS"
            ]
            front_cols = [c for c in preferred_front if c in chunk.columns]
            other_cols = [c for c in chunk.columns if c not in front_cols]
            chunk = chunk[front_cols + other_cols]

            chunk.to_csv(
                out_csv,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig"
            )
            first_write = False

        print(f"[DONE] {csv_file.name}")
        print(f"       Total rows         : {total_rows}")
        print(f"       Invalid lat/lon    : {invalid_xy_rows}")
        print(f"       Valid climate rows : {valid_climate_rows}")
        print(f"       Nodata climate     : {nodata_climate_rows}")
        print(f"       Output             : {out_csv}")

        grand_total += total_rows
        grand_invalid_xy += invalid_xy_rows
        grand_valid_climate += valid_climate_rows
        grand_nodata_climate += nodata_climate_rows

print("\n" + "=" * 60)
print("CLIMATE EXTRACTION COMPLETED")
print("=" * 60)
print(f"Grand total rows         : {grand_total}")
print(f"Grand invalid lat/lon    : {grand_invalid_xy}")
print(f"Grand valid climate rows : {grand_valid_climate}")
print(f"Grand nodata climate     : {grand_nodata_climate}")
print(f"Output folder            : {OUTPUT_DIR}")


# # Soil Type Extraction

# In[7]:


from pathlib import Path

# ============================================================
# NUMPY WORKAROUND
# ============================================================
import numpy as np
import numpy.core.records as rec
np.rec = rec

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# force fiona engine to avoid pyogrio GDAL issue
gpd.options.io_engine = "fiona"


# ============================================================
# PATHS
# ============================================================
INPUT_DIR = Path(r"H:\shc_data\SHC_with_climate")
SOIL_SHP = Path(r"H:\Carbon_Data\Soil Map Shapefile\DSMW.shp")
OUTPUT_DIR = Path(r"H:\shc_data\SHC_with_soiltype")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 100000
TARGET_CRS = "EPSG:4326"


# ============================================================
# DOMSOI MAPPING
# ============================================================
SOIL_BROAD_MAP = {
    "A": "ACRISOLS",
    "B": "CAMBISOLS",
    "C": "CHERNOZEMS",
    "D": "PODZOLUVISOLS",
    "E": "RENDZINAS",
    "F": "FERRALSOLS",
    "G": "GLEYSOLS",
    "H": "PHAEOZEMS",
    "I": "LITHOSOLS",
    "J": "FLUVISOLS",
    "K": "KASTANOZEMS",
    "L": "LUVISOLS",
    "M": "GREYZEMS",
    "N": "NITOSOLS",
    "O": "HISTOSOLS",
    "P": "PODZOLS",
    "Q": "ARENOSOLS",
    "R": "REGOSOLS",
    "S": "SOLONETZ",
    "T": "ANDOSOLS",
    "U": "RANKERS",
    "V": "VERTISOLS",
    "W": "PLANOSOLS",
    "X": "XEROSOLS",
    "Y": "YERMOSOLS",
    "Z": "SOLONCHAKS",
}

SOIL_SUBCLASS_MAP = {
    "A": "ACRISOLS",
    "Ao": "Orthic Acrisols",
    "Af": "Ferric Acrisols",
    "Ah": "Humic Acrisols",
    "Ap": "Plinthic Acrisols",
    "Ag": "Gleyic Acrisols",

    "B": "CAMBISOLS",
    "Be": "Eutric Cambisols",
    "Bd": "Dystric Cambisols",
    "Bh": "Humic Cambisols",
    "Bg": "Gleyic Cambisols",
    "Bx": "Gelic Cambisols",
    "Bk": "Calcic Cambisols",
    "Bc": "Chromic Cambisols",
    "Bv": "Vertic Cambisols",
    "Bf": "Ferralic Cambisols",

    "C": "CHERNOZEMS",
    "Ch": "Haplic Chernozems",
    "Ck": "Calcic Chernozems",
    "Cl": "Luvic Chernozems",
    "Cg": "Glossic Chernozems",

    "D": "PODZOLUVISOLS",
    "De": "Eutric Podzoluvisols",
    "Dd": "Dystric Podzoluvisols",
    "Dg": "Gleyic Podzoluvisols",

    "E": "RENDZINAS",

    "F": "FERRALSOLS",
    "Fo": "Orthic Ferralsols",
    "Fx": "Xanthic Ferralsols",
    "Fr": "Rhodic Ferralsols",
    "Fh": "Humic Ferralsols",
    "Fa": "Acric Ferralsols",
    "Fp": "Plinthic Ferralsols",

    "G": "GLEYSOLS",
    "Ge": "Eutric Gleysols",
    "Gc": "Calcaric Gleysols",
    "Gd": "Dystric Gleysols",
    "Gm": "Mollic Gleysols",
    "Gh": "Humic Gleysols",
    "Gp": "Plinthic Gleysols",
    "Gx": "Gelic Gleysols",

    "H": "PHAEOZEMS",
    "Hh": "Haplic Phaeozems",
    "Hc": "Calcaric Phaeozems",
    "Hl": "Luvic Phaeozems",
    "Hg": "Gleyic Phaeozems",

    "I": "LITHOSOLS",

    "J": "FLUVISOLS",
    "Je": "Eutric Fluvisols",
    "Jc": "Calcaric Fluvisols",
    "Jd": "Dystric Fluvisols",
    "Jt": "Thionic Fluvisols",

    "K": "KASTANOZEMS",
    "Kh": "Haplic Kastanozems",
    "Kk": "Calcic Kastanozems",
    "Kl": "Luvic Kastanozems",

    "L": "LUVISOLS",
    "Lo": "Orthic Luvisols",
    "Lc": "Chromic Luvisols",
    "Lk": "Calcic Luvisols",
    "Lv": "Vertic Luvisols",
    "Lf": "Ferric Luvisols",
    "La": "Albic Luvisols",
    "Lp": "Plinthic Luvisols",
    "Lg": "Gleyic Luvisols",

    "M": "GREYZEMS",
    "Mo": "Orthic Greyzems",
    "Mg": "Gleyic Greyzems",

    "N": "NITOSOLS",
    "Ne": "Eutric Nitosols",
    "Nd": "Dystric Nitosols",
    "Nh": "Humic Nitosols",

    "O": "HISTOSOLS",
    "Oe": "Eutric Histosols",
    "Od": "Dystric Histosols",
    "Ox": "Gelic Histosols",

    "P": "PODZOLS",
    "Po": "Orthic Podzols",
    "Pl": "Leptic Podzols",
    "Pf": "Ferric Podzols",
    "Ph": "Humic Podzols",
    "Pp": "Placic Podzols",
    "Pg": "Gleyic Podzols",

    "Q": "ARENOSOLS",
    "Qc": "Cambic Arenosols",
    "Ql": "Luvic Arenosols",
    "Qf": "Ferralic Arenosols",
    "Qa": "Albic Arenosols",

    "R": "REGOSOLS",
    "Re": "Eutric Regosols",
    "Rc": "Calcaric Regosols",
    "Rd": "Dystric Regosols",
    "Rx": "Gelic Regosols",

    "S": "SOLONETZ",
    "So": "Orthic Solonetz",
    "Sm": "Mollic Solonetz",
    "Sg": "Gleyic Solonetz",

    "T": "ANDOSOLS",
    "To": "Ochric Andosols",
    "Tm": "Mollic Andosols",
    "Th": "Humic Andosols",
    "Tv": "Vitric Andosols",

    "U": "RANKERS",

    "V": "VERTISOLS",
    "Vp": "Pellic Vertisols",
    "Vc": "Chromic Vertisols",

    "W": "PLANOSOLS",
    "We": "Eutric Planosols",
    "Wd": "Dystric Planosols",
    "Wm": "Mollic Planosols",
    "Wh": "Humic Planosols",
    "Ws": "Solodic Planosols",
    "Wx": "Gelic Planosols",

    "X": "XEROSOLS",
    "Xh": "Haplic Xerosols",
    "Xk": "Calcic Xerosols",
    "Xy": "Gypsic Xerosols",
    "Xl": "Luvic Xerosols",

    "Y": "YERMOSOLS",
    "Yh": "Haplic Yermosols",
    "Yk": "Calcic Yermosols",
    "Yy": "Gypsic Yermosols",
    "Yl": "Luvic Yermosols",
    "Yt": "Takyric Yermosols",

    "Z": "SOLONCHAKS",
    "Zo": "Orthic Solonchaks",
    "Zm": "Mollic Solonchaks",
    "Zt": "Takyric Solonchaks",
    "Zg": "Gleyic Solonchaks",
}


# ============================================================
# HELPERS
# ============================================================
def normalize_domsoi(val):
    if pd.isna(val):
        return pd.NA

    s = str(val).strip()
    if s == "":
        return pd.NA

    if len(s) == 1:
        return s.upper()

    return s[0].upper() + s[1:]


def domsoi_to_broad(domsoi):
    if pd.isna(domsoi):
        return pd.NA
    return SOIL_BROAD_MAP.get(str(domsoi)[0].upper(), pd.NA)


def domsoi_to_subclass(domsoi):
    if pd.isna(domsoi):
        return pd.NA
    domsoi = str(domsoi)
    if domsoi in SOIL_SUBCLASS_MAP:
        return SOIL_SUBCLASS_MAP[domsoi]
    return SOIL_BROAD_MAP.get(domsoi[0].upper(), pd.NA)


# ============================================================
# LOAD SOIL SHAPEFILE
# ============================================================
print("Reading soil map shapefile...")

soil_gdf = gpd.read_file(SOIL_SHP, engine="fiona", encoding="ISO-8859-1")

if "DOMSOI" not in soil_gdf.columns:
    raise ValueError("Column 'DOMSOI' not found in soil shapefile.")

if soil_gdf.crs is None:
    print("CRS missing in soil shapefile. Assigning EPSG:4326 based on lon/lat extent.")
    soil_gdf = soil_gdf.set_crs(TARGET_CRS, allow_override=True)

soil_gdf = soil_gdf[soil_gdf.geometry.notna() & ~soil_gdf.geometry.is_empty].copy()
soil_gdf["DOMSOI"] = soil_gdf["DOMSOI"].apply(normalize_domsoi)

# rename to unique field to avoid join-name collisions
soil_gdf = soil_gdf[["DOMSOI", "geometry"]].rename(columns={"DOMSOI": "DOMSOI_SOIL"}).copy()

# spatial index for speed
_ = soil_gdf.sindex

print(f"Soil polygons loaded: {len(soil_gdf)}")
print(f"Soil CRS            : {soil_gdf.crs}")


# ============================================================
# MAIN
# ============================================================
csv_files = sorted(INPUT_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

grand_total = 0
grand_written = 0
grand_invalid_xy = 0
grand_matched = 0
grand_unmatched = 0

for csv_file in csv_files:
    print(f"\nProcessing: {csv_file.name}")

    out_csv = OUTPUT_DIR / csv_file.name
    if out_csv.exists():
        out_csv.unlink()

    total_rows = 0
    written_rows = 0
    invalid_xy_rows = 0
    matched_rows = 0
    unmatched_rows = 0
    first_write = True
    original_columns = None

    for chunk in pd.read_csv(csv_file, chunksize=CHUNKSIZE, low_memory=False):
        if original_columns is None:
            original_columns = list(chunk.columns)

        total_rows += len(chunk)

        if "longitude" not in chunk.columns or "latitude" not in chunk.columns:
            raise ValueError(f"'longitude' or 'latitude' missing in {csv_file.name}")

        # remove existing soil columns if present
        for col in ["DOMSOI", "SOIL_TYPE", "SOIL_SUBCLASS"]:
            if col in chunk.columns:
                chunk = chunk.drop(columns=col)

        # numeric lon/lat
        chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
        chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")

        valid_xy_mask = (
            chunk["longitude"].notna() &
            chunk["latitude"].notna() &
            np.isfinite(chunk["longitude"]) &
            np.isfinite(chunk["latitude"]) &
            (chunk["longitude"] >= -180) & (chunk["longitude"] <= 180) &
            (chunk["latitude"] >= -90) & (chunk["latitude"] <= 90)
        )

        invalid_xy_rows += int((~valid_xy_mask).sum())

        chunk["DOMSOI"] = pd.NA
        chunk["SOIL_TYPE"] = pd.NA
        chunk["SOIL_SUBCLASS"] = pd.NA

        chunk_valid = chunk.loc[valid_xy_mask].copy()

        if not chunk_valid.empty:
            geometry = [Point(xy) for xy in zip(chunk_valid["longitude"], chunk_valid["latitude"])]
            pts_gdf = gpd.GeoDataFrame(chunk_valid, geometry=geometry, crs=TARGET_CRS)

            joined = gpd.sjoin(
                pts_gdf,
                soil_gdf,
                how="left",
                predicate="intersects"
            )

            # keep first polygon if multiple matches for same point
            joined = joined.loc[~joined.index.duplicated(keep="first")].copy()

            # safety check
            if "DOMSOI_SOIL" not in joined.columns:
                print("Joined columns were:", list(joined.columns))
                raise KeyError("DOMSOI_SOIL column not found after spatial join.")

            joined["DOMSOI"] = joined["DOMSOI_SOIL"].apply(normalize_domsoi)
            joined["SOIL_TYPE"] = joined["DOMSOI"].apply(domsoi_to_broad)
            joined["SOIL_SUBCLASS"] = joined["DOMSOI"].apply(domsoi_to_subclass)

            matched_mask = joined["DOMSOI"].notna()
            matched_rows += int(matched_mask.sum())
            unmatched_rows += int((~matched_mask).sum())

            chunk.loc[joined.index, ["DOMSOI", "SOIL_TYPE", "SOIL_SUBCLASS"]] = \
                joined[["DOMSOI", "SOIL_TYPE", "SOIL_SUBCLASS"]].values

        preferred_front = [
            "state", "district", "village", "tehsil",
            "latitude", "longitude", "date", "period",
            "LULC_VALUE", "LULC_CLASS",
            "CLIMATE_VALUE", "CLIMATE_SUBCLASS", "CLIMATE_BROAD",
            "DOMSOI", "SOIL_TYPE", "SOIL_SUBCLASS"
        ]
        front_cols = [c for c in preferred_front if c in chunk.columns]
        other_cols = [c for c in chunk.columns if c not in front_cols]
        chunk = chunk[front_cols + other_cols]

        chunk.to_csv(
            out_csv,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig"
        )
        first_write = False
        written_rows += len(chunk)

    if first_write and original_columns is not None:
        empty_cols = original_columns + ["DOMSOI", "SOIL_TYPE", "SOIL_SUBCLASS"]
        pd.DataFrame(columns=empty_cols).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] {csv_file.name}")
    print(f"       Total rows      : {total_rows}")
    print(f"       Written rows    : {written_rows}")
    print(f"       Invalid XY      : {invalid_xy_rows}")
    print(f"       Soil matched    : {matched_rows}")
    print(f"       Soil unmatched  : {unmatched_rows}")
    print(f"       Output          : {out_csv}")

    grand_total += total_rows
    grand_written += written_rows
    grand_invalid_xy += invalid_xy_rows
    grand_matched += matched_rows
    grand_unmatched += unmatched_rows

print("\n" + "=" * 60)
print("SOIL TYPE EXTRACTION COMPLETED")
print("=" * 60)
print(f"Grand total rows     : {grand_total}")
print(f"Grand written rows   : {grand_written}")
print(f"Grand invalid XY     : {grand_invalid_xy}")
print(f"Grand soil matched   : {grand_matched}")
print(f"Grand soil unmatched : {grand_unmatched}")
print(f"Output folder        : {OUTPUT_DIR}")


# # Date format correction for GEE compatibility: YYYY-MM-DD

# In[8]:


from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================
INPUT_DIR = Path(r"H:\shc_data\SHC_with_soiltype")
OUTPUT_DIR = Path(r"H:\shc_data\SHC_date_corrected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 200000

# ============================================================
# HELPERS
# ============================================================
def standardize_date_column(date_series: pd.Series) -> pd.Series:
    """
    Convert mixed date strings like:
    3/15/24, 12:00 AM
    03/15/2024
    2024-03-15
    into YYYY-MM-DD

    Unparseable values become blank.
    """
    s = date_series.astype(str).str.strip()

    # treat common empty placeholders as missing
    missing_mask = (
        date_series.isna() |
        (s == "") |
        (s.str.lower() == "nan") |
        (s.str.lower() == "none") |
        (s.str.lower() == "nat")
    )

    parsed = pd.to_datetime(s, errors="coerce")

    out = parsed.dt.strftime("%Y-%m-%d")
    out[missing_mask] = ""

    # anything still unparsed also becomes blank
    out = out.fillna("")

    return out


# ============================================================
# MAIN
# ============================================================
csv_files = sorted(INPUT_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {INPUT_DIR}")

grand_total = 0
grand_parsed = 0
grand_blank = 0

for csv_file in csv_files:
    print(f"\nProcessing: {csv_file.name}")

    out_csv = OUTPUT_DIR / csv_file.name
    if out_csv.exists():
        out_csv.unlink()

    total_rows = 0
    parsed_rows = 0
    blank_rows = 0
    first_write = True
    original_columns = None

    for chunk in pd.read_csv(csv_file, chunksize=CHUNKSIZE, low_memory=False):
        if original_columns is None:
            original_columns = list(chunk.columns)

        total_rows += len(chunk)

        if "date" not in chunk.columns:
            raise ValueError(f"'date' column not found in file: {csv_file.name}")

        original_date = chunk["date"].copy()
        chunk["date"] = standardize_date_column(chunk["date"])

        parsed_mask = chunk["date"].astype(str).str.strip() != ""
        parsed_rows += int(parsed_mask.sum())
        blank_rows += int((~parsed_mask).sum())

        chunk.to_csv(
            out_csv,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig"
        )
        first_write = False

    # if file was empty, still create an empty csv
    if first_write and original_columns is not None:
        pd.DataFrame(columns=original_columns).to_csv(
            out_csv, index=False, encoding="utf-8-sig"
        )

    print(f"[DONE] {csv_file.name}")
    print(f"       Total rows   : {total_rows}")
    print(f"       Parsed dates : {parsed_rows}")
    print(f"       Blank dates  : {blank_rows}")
    print(f"       Output       : {out_csv}")

    grand_total += total_rows
    grand_parsed += parsed_rows
    grand_blank += blank_rows

print("\n" + "=" * 60)
print("DATE CORRECTION COMPLETED")
print("=" * 60)
print(f"Grand total rows   : {grand_total}")
print(f"Grand parsed dates : {grand_parsed}")
print(f"Grand blank dates  : {grand_blank}")
print(f"Output folder      : {OUTPUT_DIR}")


# In[ ]:




