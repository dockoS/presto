from typing import List, Optional

import torch
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
#from .pipelines.dynamicworld import DynamicWorld2020_2021
# from .pipelines.s1_s2_era5_srtm import (
#     BANDS,
#     ERA5_BANDS,
#     NORMED_BANDS,
#     REMOVED_BANDS,
#     S1_BANDS,
#     S1_S2_ERA5_SRTM,
#     S2_BANDS,
#     SRTM_BANDS,
# )
S1_BANDS = ["VV", "VH"]
# EarthEngine estimates Sentinel-1 values range from -50 to 1
S1_SHIFT_VALUES = [25.0, 25.0]
S1_DIV_VALUES = [25.0, 25.0]
S2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
S2_SHIFT_VALUES = [float(0.0)] * len(S2_BANDS)
S2_DIV_VALUES = [float(1e4)] * len(S2_BANDS)
ERA5_BANDS = ["temperature_2m", "total_precipitation"]
# for temperature, shift to celcius and then divide by 35 based on notebook (ranges from)
# 37 to -22 degrees celcius
# For rainfall, based on
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
ERA5_SHIFT_VALUES = [-272.15, 0.0]
ERA5_DIV_VALUES = [35.0, 0.03]
SRTM_BANDS = ["elevation", "slope"]
# visually gauged 90th percentile from
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
SRTM_SHIFT_VALUES = [0.0, 0.0]
SRTM_DIV_VALUES = [2000.0, 50.0]

DYNAMIC_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS
STATIC_BANDS = SRTM_BANDS

DYNAMIC_BANDS_SHIFT = S1_SHIFT_VALUES + S2_SHIFT_VALUES + ERA5_SHIFT_VALUES
DYNAMIC_BANDS_DIV = S1_DIV_VALUES + S2_DIV_VALUES + ERA5_DIV_VALUES

STATIC_BANDS_SHIFT = SRTM_SHIFT_VALUES
STATIC_BANDS_DIV = SRTM_DIV_VALUES

# These bands are what is created by the Engineer. If the engineer changes, the bands
# here will need to change (and vice versa)
REMOVED_BANDS = ["B1", "B10"]
RAW_BANDS = DYNAMIC_BANDS + STATIC_BANDS

BANDS = [x for x in DYNAMIC_BANDS if x not in REMOVED_BANDS] + STATIC_BANDS + ["NDVI"]
# NDVI is between 0 and 1
ADD_BY = (
    [DYNAMIC_BANDS_SHIFT[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + STATIC_BANDS_SHIFT
    + [0.0]
)
DIVIDE_BY = (
    [DYNAMIC_BANDS_DIV[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + STATIC_BANDS_DIV
    + [1.0]
)

NUM_TIMESTEPS = 12
NUM_ORG_BANDS = len(BANDS)
TIMESTEPS_IDX = list(range(NUM_TIMESTEPS))

NORMED_BANDS = [x for x in BANDS if x != "B9"]
NUM_BANDS = len(NORMED_BANDS)
BANDS_IDX = list(range(NUM_BANDS))
BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "S1": [NORMED_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [NORMED_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [NORMED_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [NORMED_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [NORMED_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [NORMED_BANDS.index(b) for b in ["B11", "B12"]],  # Include B10?
        "ERA5": [NORMED_BANDS.index(b) for b in ERA5_BANDS],
        "SRTM": [NORMED_BANDS.index(b) for b in SRTM_BANDS],
        "NDVI": [NORMED_BANDS.index("NDVI")],
    }
)



def construct_single_presto_input(
    s1: Optional[torch.Tensor] = None,
    s1_bands: Optional[List[str]] = None,
    s2: Optional[torch.Tensor] = None,
    s2_bands: Optional[List[str]] = None,
    era5: Optional[torch.Tensor] = None,
    era5_bands: Optional[List[str]] = None,
    srtm: Optional[torch.Tensor] = None,
    srtm_bands: Optional[List[str]] = None,
    dynamic_world: Optional[torch.Tensor] = None,
    normalize: bool = True,
):
    """
    Inputs are paired into a tensor input <X> and a list <X>_bands, which describes <X>.

    <X> should have shape (num_timesteps, len(<X>_bands)), with the following bands possible for
    each input:

    s1: ["VV", "VH"]
    s2: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    era5: ["temperature_2m", "total_precipitation"]
        "temperature_2m": Temperature of air at 2m above the surface of land,
            sea or in-land waters in Kelvin (K)
        "total_precipitation": Accumulated liquid and frozen water, including rain and snow,
            that falls to the Earth's surface. Measured in metres (m)
    srtm: ["elevation", "slope"]

    dynamic_world is a 1d input of shape (num_timesteps,) representing the dynamic world classes
        of each timestep for that pixel
    """
    num_timesteps_list = [x.shape[0] for x in [s1, s2, era5, srtm] if x is not None]
    if dynamic_world is not None:
        num_timesteps_list.append(len(dynamic_world))

    assert len(num_timesteps_list) > 0


    assert all(num_timesteps_list[0] == timestep for timestep in num_timesteps_list)
    num_timesteps = num_timesteps_list[0]
    mask, x = torch.ones(num_timesteps, len(BANDS)), torch.zeros(num_timesteps, len(BANDS))

    for band_group in [
        (s1, s1_bands, S1_BANDS),
        (s2, s2_bands, S2_BANDS),
        (era5, era5_bands, ERA5_BANDS),
        (srtm, srtm_bands, SRTM_BANDS),
    ]:
        data, input_bands, output_bands = band_group
        if data is not None:
            assert input_bands is not None
        else:
            continue

        kept_output_bands = [x for x in output_bands if x not in REMOVED_BANDS]
        # construct a mapping from the input bands to the expected bands
        kept_input_band_idxs = [i for i, val in enumerate(input_bands) if val in kept_output_bands]
        kept_input_band_names = [val for val in input_bands if val in kept_output_bands]

        input_to_output_mapping = [BANDS.index(val) for val in kept_input_band_names]

        x[:, input_to_output_mapping] = data[:, kept_input_band_idxs]
        mask[:, input_to_output_mapping] = 0


    keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
    mask = mask[:, keep_indices]

    # if normalize:
    #     # normalize includes x = x[:, keep_indices]
    #     x = S1_S2_ERA5_SRTM.normalize(x)
    #     if s2_bands is not None:
    #         if ("B8" in s2_bands) and ("B4" in s2_bands):
    #             mask[:, NORMED_BANDS.index("NDVI")] = 0
    # else:
    x = x[:, keep_indices]
    return x, mask, dynamic_world
