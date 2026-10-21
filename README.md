# Spherical Fibonacci Lattice Pixels (SFLP) for VR Images/Videos and Uniform Spherical CNN Toolkit

A Python package for creating Spherical Fibonacci Lattice Pixel (SFLP) VR Images and Videos as well as Uniform Spherical Convolutional Kernels. This tool allows for generating the pixel layout in a 2D rectangular format, and conversion from other popular VR encoding methods to the SFLP encoding method.

## Features

- Generating the SFLP layout in a 2D rectangular format.
- Conversion from popular VR encoding methods using projection methods such as ERP, EAP, CMP, or EAC, to the SFLP encoding method.
- Evaluation of different VR encoding methods based on S-PSNR, using BD-rate and bitrate savings calculations.
- Comprehensive error handling and validation.
- Uniform spherical convolutional kernels: SFLP layout preserves spatial locality in 2D rectangular format and can be efficiently used to run SFL points through convolutional neural network (CNN) modules.
- CPP code for real-time SFLP re-encoding from ERP and decoding/viewport sampling.

## Installation

### Requirements

- Python 3.9 or higher
- CUDA 12.x
- numpy
- pandas
- ffmpeg (for video conversion)
- cupy and GPU for accelerated encoding/decoding
- matplotlib and pyvista (visualization)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Chtun/SFLP_toolkit.git
cd SFLP_toolkit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

Here's a simple example to get you started.

First, generate the layout tables:

```python
from pathlib import Path
from SFLP_toolkit.utilities.pixel_arrangement_utils import generate_SFLP_pixel_arrangement, SFLPArrangementType
import pandas as pd

output_dir = Path("../output/layouts")
input_dir = Path("../input")
output_dir.mkdir(parents=True, exist_ok=True)

SFLP_num_points = 21660001

arrangement_type = SFLPArrangementType.STANDARD

arranged_pixel_indices_df = generate_SFLP_pixel_arrangement(SFLP_num_points, arrangement_type)
SFLP_arrangement_save_path = output_dir / f"SFLP_arrangement_{arrangement_type.name}-360_No_Tiling-{SFLP_num_points}.csv"
arranged_pixel_indices_df.to_csv(SFLP_arrangement_save_path, index=False, header=False)

print(f"SFLP layout saved to : {SFLP_arrangement_save_path}")

num_pixels_horizontal = 1900
num_pixels_vertical = num_pixels_horizontal
Cube_num_points = num_pixels_vertical * num_pixels_horizontal * 6

Cube_frame_df = generate_CubeMap_pixel_arrangement(num_pixels_horizontal=num_pixels_horizontal, num_pixels_vertical=num_pixels_vertical)
Cube_arrangement_save_path = output_dir / f"CubeMap_pixel_arrangement-360_No_Tiling-{Cube_num_points}.csv"
Cube_frame_df.to_csv(Cube_arrangement_save_path, index=False, header=False)

print(f"CubeMap layout saved to : {Cube_arrangement_save_path}")

# For some video conversion with H.265, videos need to have height and width which are multiples of 2. The following creates a padded version of SFLP for video conversion which complies with the requirement.

padding_size = 2

H, W = arranged_pixel_indices_df.shape
pad_H = (H + padding_size - 1) // padding_size * padding_size   # round up to multiple of padding_size
pad_W = (W + padding_size - 1) // padding_size * padding_size   # round up to multiple of padding_size

if pad_H != H or pad_W != W:
    # Create a padded DataFrame
    padded_df = arranged_pixel_indices_df.copy()

    # Pad columns (extend last column to the right)
    if pad_W > W:
        last_col = arranged_pixel_indices_df.iloc[:, -1:]
        for _ in range(W, pad_W):
            padded_df[_] = last_col.values

    # Pad rows (extend last row downward)
    if pad_H > H:
        last_row = padded_df.iloc[-1:, :].copy()
        for _ in range(H, pad_H):
            padded_df = pd.concat([padded_df, last_row], axis=0)

    arranged_pixel_indices_df = padded_df.reset_index(drop=True)

output_path = output_dir / f"SFLP_arrangement_{arrangement_type.name}-360_No_Tiling-{num_points}_padded.csv"

# Save padded CSV
arranged_pixel_indices_df.to_csv(
    output_path,
    index=False, header=False
)
```

Now, convert an example image in ERP to SFLP, CMP, EAC, and EAP:

```python
from pathlib import Path
from PIL import Image
import math

from SFLP_toolkit.utilities.file_conversion_utils import *

input_dir = Path("../input")
layout_input_dir = Path("../output/layouts")

output_dir = Path("../output/images")
# Ensure the directory exists
output_dir.mkdir(parents=True, exist_ok=True)

input_image_name = "ERP-CableCar-snapshot"
input_image_path = input_dir / f"{input_image_name}.jpg"

# ===
# Image names and metadata
# ===

# SFLP

SFLP_num_pixels = 21660001
arrangement_type = SFLPArrangementType.STANDARD

SFLP_layout_csv_path = layout_input_dir / f"SFLP_arrangement_{arrangement_type.name}-360_No_Tiling-{SFLP_num_pixels}.csv"
SFLP_output_path = output_dir / f"SFLP-{input_image_name}-{SFLP_num_pixels}-{arrangement_type.name}.jpeg"

# CubeMap (CMP and EAC)

CubeMap_num_pixels = 21660000

if math.sqrt(CubeMap_num_pixels / 6) != int(math.sqrt(CubeMap_num_pixels / 6)):
    raise ValueError("Invalid number of CMP pixels")

face_size = int(math.sqrt(CubeMap_num_pixels / 6))

CubeMap_layout_csv_path = layout_input_dir / f"CubeMap_pixel_arrangement-360_No_Tiling-{CubeMap_num_pixels}.csv"
CMP_output_path = output_dir / f"CMP-{input_image_name}-{CubeMap_num_pixels}.jpeg"
EAC_output_path = output_dir / f"EAC-{input_image_name}-{CubeMap_num_pixels}.jpeg"

# ERP and EAP

ERP_size = (6600, 3300)
ERP_num_pixels = (ERP_size[0] * ERP_size[1])
ERP_output_path = output_dir / f"Reencoded_ERP_image-{ERP_num_pixels}.jpeg"
EAP_output_path = output_dir / f"EAP-{input_image_name}-{ERP_num_pixels}.jpeg"

# ===
# Sampling Parameters
# ===

sample_type = Image.Resampling.BILINEAR


# ===
# Image Conversion
# ===


ERP_to_SFLP_image_conversion(
    num_pixels=SFLP_num_pixels,
    layout_csv_path=SFLP_layout_csv_path,
    input_image_path=input_image_path,
    output_path=SFLP_output_path,
    sample_type=sample_type
)

ERP_to_CMP_image_conversion(
    face_size=face_size,
    layout_csv_path=CubeMap_layout_csv_path,
    input_image_path=input_image_path,
    output_path=CMP_output_path,
    sample_type=sample_type
)

ERP_to_EAC_image_conversion(
    face_size=face_size,
    layout_csv_path=CubeMap_layout_csv_path,
    input_image_path=input_image_path,
    output_path=EAC_output_path,
    sample_type=sample_type
)

ERP_to_EAP_image_conversion(
    output_size=ERP_size,
    input_image_path=input_image_path,
    output_path=EAP_output_path,
    sample_type=sample_type
)

reencode_ERP_image(
    input_image_path=input_image_path,
    output_path=ERP_output_path,
    output_size=ERP_size,
    sample_type=sample_type
)

```

## Usage

### Basic Usage

A basic way to convert a video in ERP to another encoding method (EAP, CMP, EAC, SFLP):

```python
from pathlib import Path
from PIL import Image
import math
from config import FFMPEG, FFMPEG_GPU_PATH
from SFLP_toolkit.utilities.file_conversion_utils import (
  ERP_to_SFLP_video_conversion,
  ERP_to_CMP_video_conversion,
  ERP_to_EAC_video_conversion,
  ERP_to_EAP_video_conversion,
  reencode_ERP_video,
  VideoEncoder,
  SFLPArrangementType,
)

output_dir = Path("..\\output\\reencoded_videos")
# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

input_dir = Path("..\\input") # Change this directory to where your high-resolution videos reside
layout_input_dir = Path("..\\output\\layouts") # This directory holds the layouts for SFLP and CubeMap arrangements.

# New Encoding arrangement details
SFLP_num_pixels = 21660001
SFLP_arrangement_type = SFLPArrangementType.STANDARD

CubeMap_num_pixels = 21660000
face_size = int(math.sqrt(CubeMap_num_pixels // 6))
CubeMap_layout_csv_path = layout_input_dir / f"CubeMap_pixel_arrangement-360_No_Tiling-{CubeMap_num_pixels}.csv"

ERP_reencoded_output_size = (6600, 3300)
ERP_num_pixels = ERP_reencoded_output_size[0] * ERP_reencoded_output_size[1]

sample_type = Image.Resampling.BILINEAR
video_container = VideoEncoder.H265

# Input Videos
input_video_names = [
    # Add the names of ERP videos to convert
]

# Bitrates
bitrates = [
    5000,
    10000,
    25000,
    40000,
    60000,
]

def process_video(input_video_name: str, bitrates: list[int], SFLP_num_pixels: int):
    input_video_path = input_dir / f"{input_video_name}.mp4"

    
    SFLP_layout_csv_path = layout_input_dir / f"SFLP_arrangement_{SFLP_arrangement_type.name}-360_No_Tiling-{SFLP_num_pixels}_padded.csv"
    SFLP_output_name = f"SFLP-360_no_tiling-{input_video_name}-{SFLP_num_pixels}-SFLP_arrangement_{SFLP_arrangement_type.name}"
    ERP_to_SFLP_video_conversion(
        num_pixels=SFLP_num_pixels,
        csv_path=SFLP_layout_csv_path,
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=SFLP_output_name,
        output_video_type=video_container,
        ffmpeg_path=FFMPEG_GPU_PATH,
        sample_type=sample_type,
        bitrates=bitrates
    )

    CMP_output_name = f"CMP-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}"
    ERP_to_CMP_video_conversion(
        face_size=face_size,
        csv_path=CubeMap_layout_csv_path,
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=CMP_output_name,
        output_video_type=video_container,
        ffmpeg_path=FFMPEG_GPU_PATH,
        sample_type=sample_type,
        bitrates=bitrates
    )

    EAC_output_name = f"EAC-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}"
    ERP_to_EAC_video_conversion(
        face_size=face_size,
        csv_path=CubeMap_layout_csv_path,
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=EAC_output_name,
        output_video_type=video_container,
        ffmpeg_path=FFMPEG_GPU_PATH,
        sample_type=sample_type,
        bitrates=bitrates
    )

    EAP_output_name = f"EAP-360_no_tiling-{input_video_name}-{ERP_num_pixels}"
    ERP_to_EAP_video_conversion(
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=EAP_output_name,
        output_size=ERP_reencoded_output_size,
        output_video_type=video_container,
        ffmpeg_path=FFMPEG_GPU_PATH,
        sample_type=sample_type,
        bitrates=bitrates
    )

    ERP_output_name = f"Reencoded_ERP-{input_video_name}-{ERP_num_pixels}"
    reencode_ERP_video(
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=ERP_output_name,
        output_size=ERP_reencoded_output_size,
        output_video_type=video_container,
        ffmpeg_path=FFMPEG_GPU_PATH,
        sample_type=sample_type,
        bitrates=bitrates
    )

if __name__ == "__main__":
    for input_video_name in input_video_names:
        process_video(
            input_video_name=input_video_name,
            bitrates=bitrates,
            SFLP_num_pixels=SFLP_num_pixels
        )
```

### Encoding Method Evaluation (Video)

To evaluate the different encoding methods, first compute the visual quality (S-PSNR) of each video, compared to the ground truth (original high-resolution videos):
```python
from pathlib import Path
from PIL import Image
import math
import csv

from SFLP_toolkit.utilities.evaluation_utils import (
  evaluate_video_quality,
  generator_vec_geodesic_distance,
  ProjectionMetadata,
  EncodingProjection,
  SFLPMetadata,
  VideoEncoder,
  WeightingType,
  SFLPArrangementType,
)


sample_type = Image.Resampling.BILINEAR
weighting_type = WeightingType.VIEWING_BEHAVIOR
output_video_container = VideoEncoder.H265
presets_and_pixels = "Preset_Slow-21.6M_Pixels"

# Input directories
ground_truth_dir = Path("..\\input") # Replace with the folder to your original, high-resolution videos.
reencoded_dir = Path("..\\output\\reencoded_videos")

# Output directory
output_dir = Path(f"..\\output\\SPSNR\\weighting-{weighting_type.name}")
output_dir.mkdir(parents=True, exist_ok=True)


# Number of Points on Sphere to Sample for S-PSNR calculations
num_evaluation_points = 100001

# Metadata
ground_truth_metadata = ProjectionMetadata(EncodingProjection.ERP)

SFLP_num_pixels = 21660001
SFLP_arrangement_type = SFLPArrangementType.STANDARD
# Hint: use generate_SFLP_pixel_arrangement() to generate the pixel arrangement to help derive these values.
optimal_index_hops = [4181, 6765] # 2584 10946  # Uses the zig and zag column generator vector index hops:
                                    # zig and zag -> arr[0][0] - arr[0][1] and arr[0][1] - arr[0][2], zig < zag.
first_element_SFLP_index = 21656007 # Uses the first element of the first column in the generated pixel arrangement: arr[0][0].
last_element_SFLP_index = 1242 # Uses the last element of the first column in the generated pixel arrangement: arr[0][col_length - 1].

optimal_index_hops_metadata = [(k, generator_vec_geodesic_distance(k, SFLP_num_pixels)) for k in optimal_index_hops]
optimal_index_hops_metadata = [(k, val[0], val[1], val[2]) for k, val in optimal_index_hops_metadata]

SFLP_metadata = SFLPMetadata(
    num_pixels=SFLP_num_pixels,
    arrangement_type=SFLP_arrangement_type,
    optimal_index_hops=optimal_index_hops_metadata,
    first_element_SFLP_index=first_element_SFLP_index,
    last_element_SFLP_index=last_element_SFLP_index,
)


CubeMap_num_pixels = 21660000
face_size = int(math.sqrt(CubeMap_num_pixels // 6))
CMP_metadata = ProjectionMetadata(EncodingProjection.CMP)
EAC_metadata =  ProjectionMetadata(EncodingProjection.EAC)

ERP_reencoded_output_size = (6600, 3300)
ERP_num_pixels = ERP_reencoded_output_size[0] * ERP_reencoded_output_size[1]
ERP_metadata = ProjectionMetadata(EncodingProjection.ERP)
EAP_metadata = ProjectionMetadata(EncodingProjection.EAP)

# Input Videos
input_video_names = [
    # Add the names of ERP videos to convert
]

# Bitrates
bitrates = [
    5000,
    10000,
    25000,
    40000,
    60000,
]

if __name__ == "__main__":
    
    for input_video_name in input_video_names:
        for bitrate in bitrates:
            encoding_info = f"{SFLP_arrangement_type.name}-{output_video_container.name}-Target_{bitrate}k-{presets_and_pixels}-{weighting_type.name}"
            compared_subfolder = f"{output_video_container.name}-Target_{bitrate}k-Preset_Slow"
            compared_dir = Path(reencoded_dir, f"{compared_subfolder}")

            ground_truth_video_name = input_video_name + ".mp4"
            ground_truth_video_path = ground_truth_dir / ground_truth_video_name

            SFLP_video_name = f"SFLP-360_no_tiling-{input_video_name}-{SFLP_num_pixels}-SFLP_arrangement_{SFLP_arrangement_type.name}-bitrate_{bitrate}.mp4"
            SFLP_compared_video_path = compared_dir / SFLP_video_name

            SFLP_SPSNR = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=SFLP_compared_video_path,
                compared_metadata=SFLP_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
            )

            ERP_video_name = f"Reencoded_ERP-{input_video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            ERP_video_path = compared_dir / ERP_video_name

            ERP_SPSNR = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=ERP_video_path,
                compared_metadata=ERP_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
            )

            EAP_video_name = f"EAP-360_no_tiling-{input_video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            EAP_video_path = compared_dir / EAP_video_name

            EAP_SPSNR = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=EAP_video_path,
                compared_metadata=EAP_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
            )

            CMP_video_name = f"CMP-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            CMP_video_path = compared_dir / CMP_video_name

            CMP_SPSNR = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=CMP_video_path,
                compared_metadata=CMP_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
            )

            EAC_video_name = f"EAC-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            EAC_video_path = compared_dir / EAC_video_name

            EAC_SPSNR = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=EAC_video_path,
                compared_metadata=EAC_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
            )
            
            # Write to CSV
            SPSNR_csv_name = f"SPSNR-{input_video_name}-{encoding_info}.csv"
            SPSNR_path = output_dir / SPSNR_csv_name
            with open(SPSNR_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Frame", f"ERP ({ERP_num_pixels}) S-PSNR", f"EAP ({ERP_num_pixels}) S-PSNR", f"CMP ({CubeMap_num_pixels}) S-PSNR", f"EAC ({CubeMap_num_pixels}) S-PSNR", f"SFLP ({SFLP_num_pixels}) S-PSNR"])  # Header
                for frame in SFLP_SPSNR.keys():
                    ERP_SPSNR_val = ERP_SPSNR[frame]
                    EAP_SPSNR_val = EAP_SPSNR[frame]
                    CMP_SPSNR_val = CMP_SPSNR[frame]
                    EAC_SPSNR_val = EAC_SPSNR[frame]
                    SFLP_SPSNR_val = SFLP_SPSNR[frame]

                    writer.writerow([frame, ERP_SPSNR_val, EAP_SPSNR_val, CMP_SPSNR_val, EAC_SPSNR_val, SFLP_SPSNR_val])

```

Next, compute the BD-rate and bitrate savings:

```python
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import defaultdict
import numpy as np

from SFLP_toolkit.utilities.evaluation_utils import (
  ComparisonMetric,
  VideoEncoder,
  extract_average_quality,
  save_average_quality,
  generate_rd_curve,
  compute_bd_rate,
  get_exact_bitrate,
  matches_categories,
  WeightingType,
  SFLPArrangementType,
)

from config import *


sample_type = Image.Resampling.BILINEAR
output_video_container = VideoEncoder.H265
weighting_type = WeightingType.EQUAL
SFLP_arrangement_type = SFLPArrangementType.STANDARD
comparison_metric = ComparisonMetric.SPSNR
preset = "Preset_Slow"
pixels = "21.6M_Pixels"

temporal_levels = [low_temporal_name, medium_temporal_name, high_temporal_name]
spatial_levels = [low_spatial_name, medium_spatial_name, high_spatial_name]

average_names = [f"{t} Average" for t in temporal_levels] + [f"{s} Average" for s in spatial_levels]
average_names.append(macro_average_name)

# Whether a video must contain all categories, or just one, in order to be a part of the averaging group.
match_all = False


# Input directory
input_dir = Path(f"../output/{comparison_metric.name}/weighting-{weighting_type.name}")
reencoded_dir = Path("D:/GitHub_Projects/VR_Encoding_Dataset/Output")

# Output directory
output_dir = Path(f"../output/{comparison_metric.name}/Averages")
output_dir.mkdir(parents=True, exist_ok=True)

SFLP_num_pixels = 21660001
CubeMap_num_pixels = 21660000
ERP_num_pixels = 21780000

method_names = {
    "ERP": f"ERP ({ERP_num_pixels}) S-PSNR",
    "EAP": f"EAP ({ERP_num_pixels}) S-PSNR",
    "CMP": f"CMP ({CubeMap_num_pixels}) S-PSNR",
    "EAC": f"EAC ({CubeMap_num_pixels}) S-PSNR",
    "SFLP": f"SFLP ({SFLP_num_pixels}) S-PSNR",
    
    "ERP vs. SFLP": f"SFLP ({SFLP_num_pixels}) bitrate saving against ERP ({ERP_num_pixels})",
    "EAP vs. SFLP": f"SFLP ({SFLP_num_pixels}) bitrate saving against EAP ({ERP_num_pixels})",
    "CMP vs. SFLP": f"SFLP ({SFLP_num_pixels}) bitrate saving against CMP ({CubeMap_num_pixels})",
    "EAC vs. SFLP": f"SFLP ({SFLP_num_pixels}) bitrate saving against EAC ({CubeMap_num_pixels})",
    
    "ERP vs. EAP": f"EAP ({ERP_num_pixels}) bitrate saving against ERP ({ERP_num_pixels})",
    "ERP vs. CMP": f"CMP ({CubeMap_num_pixels}) bitrate saving against ERP ({ERP_num_pixels})",
    "ERP vs. EAC": f"EAC ({CubeMap_num_pixels}) bitrate saving against ERP ({ERP_num_pixels})",
}

# Input Videos
input_video_names = [
    # Add the names of ERP videos to convert
]

video_categories = VIDEO_CATEGORIES

# Bitrates
bitrates = [
    5000,
    10000,
    25000,
    40000,
    60000,
]

if __name__ == "__main__":
    
    quality_per_video = {}

    for average_name in average_names:
        quality_per_video[average_name] = {}

    for input_video_name in input_video_names:
        quality_per_video[input_video_name] = {}

    for bitrate in bitrates:

        average_quality = {}
        for average_name in average_names:
            average_quality[average_name] = {}
            average_quality[average_name]["num_vids"] = 0

        for input_video_name in input_video_names:
            quality_info = f"{SFLP_arrangement_type.name}-{output_video_container.name}-Target_{bitrate}k-{preset}-{pixels}-{weighting_type.name}"
            quality_csv_name = f"{comparison_metric.name}-{input_video_name}-{quality_info}.csv"
            quality_path = input_dir / quality_csv_name

            averages = extract_average_quality(quality_path)
            quality_per_video[input_video_name][bitrate] = averages

            average_quality[macro_average_name]["num_vids"] += 1
            for average_name in average_names:
                    if matches_categories(average_name, video_categories[input_video_name], match_all=match_all):
                        average_quality[average_name]["num_vids"] += 1

            for method, avg_val in averages.items():
                if method not in average_quality[macro_average_name]:
                    for average_name in average_names:
                        average_quality[average_name][method] = 0

                average_quality[macro_average_name][method] += avg_val
                for average_name in average_names:
                    if matches_categories(average_name, video_categories[input_video_name], match_all=match_all):
                        average_quality[average_name][method] += avg_val
        
        for average_name in average_names:
            for method in average_quality[average_name]:
                if method != "num_vids" and average_quality[average_name]["num_vids"] != 0:
                    average_quality[average_name][method] = average_quality[average_name][method] / average_quality[average_name]["num_vids"]
            
            # Remove the num_vids value after using for normalization.
            average_quality[average_name].pop("num_vids", None)

            quality_per_video[average_name][bitrate] = average_quality[average_name]


    # Restructure the data so that each encoding method holds a list of bitrates.
    bitrates = sorted(list(next(iter(quality_per_video.values())).keys()))
    restructured_data = defaultdict(lambda: defaultdict(list))

    for video_name, bitrates_data in quality_per_video.items():
        if not ("Average" in video_name):
            for bitrate in sorted(bitrates_data.keys()):
                for method, value in bitrates_data[bitrate].items():
                    restructured_data[video_name][method].append(value)

    bitrate_savings = {}

    for average_name in average_names:
        bitrate_savings[average_name] = {}
        bitrate_savings[average_name]
        bitrate_savings[average_name][method_names["ERP vs. SFLP"]] = 0
        bitrate_savings[average_name][method_names["EAP vs. SFLP"]] = 0
        bitrate_savings[average_name][method_names["CMP vs. SFLP"]] = 0
        bitrate_savings[average_name][method_names["EAC vs. SFLP"]] = 0
        bitrate_savings[average_name][method_names["ERP vs. EAP"]] = 0
        bitrate_savings[average_name][method_names["ERP vs. CMP"]] = 0
        bitrate_savings[average_name][method_names["ERP vs. EAC"]] = 0

    videos_calculated = {}
    for average_name in average_names:
        videos_calculated[average_name] =  {}
        videos_calculated[average_name][method_names["ERP vs. SFLP"]] = 0
        videos_calculated[average_name][method_names["EAP vs. SFLP"]] = 0
        videos_calculated[average_name][method_names["CMP vs. SFLP"]] = 0
        videos_calculated[average_name][method_names["EAC vs. SFLP"]] = 0
        videos_calculated[average_name][method_names["ERP vs. EAP"]] = 0
        videos_calculated[average_name][method_names["ERP vs. CMP"]] = 0
        videos_calculated[average_name][method_names["ERP vs. EAC"]] = 0

    # Generate the RD curves.
    encoding_info_no_bitrate = f"{SFLP_arrangement_type.name}-{output_video_container.name}-{preset}"
    quality_info = f"{encoding_info_no_bitrate}-{pixels}-{weighting_type.name}"
    for video_name, video_data in restructured_data.items():
        
        # Extract the exact bitrates for each encoding type for this video.
        exact_bitrates = {}
        exact_ERP_bitrates = []
        exact_EAP_bitrates = []
        exact_CMP_bitrates = []
        exact_EAC_bitrates = []
        exact_SFLP_bitrates = []

        for bitrate in bitrates:
            encoding_info = f"{output_video_container.name}-Target_{bitrate}k-{preset}"
            video_dir = reencoded_dir / f"{encoding_info}"
            
            ERP_video_name = f"Reencoded_ERP-{video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            ERP_video_path = video_dir / ERP_video_name

            exact_ERP_bitrates.append(get_exact_bitrate(ERP_video_path))

            EAP_video_name = f"EAP-360_no_tiling-{video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            EAP_video_path = video_dir / EAP_video_name

            exact_EAP_bitrates.append(get_exact_bitrate(EAP_video_path))

            CMP_video_name = f"CMP-360_no_tiling-{video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            CMP_video_path = video_dir / CMP_video_name

            exact_CMP_bitrates.append(get_exact_bitrate(CMP_video_path))

            EAC_video_name = f"EAC-360_no_tiling-{video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            EAC_video_path = video_dir / EAC_video_name

            exact_EAC_bitrates.append(get_exact_bitrate(EAC_video_path))

            
            SFLP_video_name = f"SFLP-360_no_tiling-{video_name}-{SFLP_num_pixels}-SFLP_arrangement_{SFLP_arrangement_type.name}-bitrate_{bitrate}.mp4"
            SFLP_video_path = video_dir / SFLP_video_name

            exact_SFLP_bitrates.append(get_exact_bitrate(SFLP_video_path))


        for method in video_data:
            if "ERP" in method:
                exact_bitrates[method] = exact_ERP_bitrates
            elif "EAP" in method:
                exact_bitrates[method] = exact_EAP_bitrates
            elif "CMP" in method:
                exact_bitrates[method] = exact_CMP_bitrates
            elif "EAC" in method:
                exact_bitrates[method] = exact_EAC_bitrates
            elif "SFLP" in method:
                exact_bitrates[method] = exact_SFLP_bitrates

        print(exact_bitrates)

        exit()
        
        graph_output_filename = f"rd_curve-{video_name}-{quality_info}.png"
        graph_output_file_path = output_dir / graph_output_filename
        modified_video_name = video_name.split("_pano")[0]
        if "original_ERP-" in modified_video_name:
            modified_video_name = video_name.split("original_ERP-")[1]
        generate_rd_curve(video_data, exact_bitrates, modified_video_name, graph_output_file_path)

        table_output_filename = f"average_{comparison_metric.name}-{video_name}-{quality_info}.csv"
        table_output_file_path = output_dir / table_output_filename
        save_average_quality(video_data=video_data, bitrates=bitrates, output_path=table_output_file_path)

        print()

        ERP_quality = video_data[method_names["ERP"]]
        EAP_quality = video_data[method_names["EAP"]]
        CMP_quality = video_data[method_names["CMP"]]
        EAC_quality = video_data[method_names["EAC"]]
        SFLP_quality = video_data[method_names["SFLP"]]

        # Compute BD-Rate savings
        bd_ERP_vs_SFLP = compute_bd_rate(exact_ERP_bitrates, ERP_quality, exact_SFLP_bitrates, SFLP_quality)
        bd_EAP_vs_SFLP = compute_bd_rate(exact_EAP_bitrates, EAP_quality, exact_SFLP_bitrates, SFLP_quality)
        bd_CMP_vs_SFLP = compute_bd_rate(exact_CMP_bitrates, CMP_quality, exact_SFLP_bitrates, SFLP_quality)
        bd_EAC_vs_SFLP = compute_bd_rate(exact_EAC_bitrates, EAC_quality, exact_SFLP_bitrates, SFLP_quality)

        bd_ERP_vs_EAP = compute_bd_rate(exact_ERP_bitrates, ERP_quality, exact_EAP_bitrates, EAP_quality)
        bd_ERP_vs_CMP = compute_bd_rate(exact_ERP_bitrates, ERP_quality, exact_CMP_bitrates, CMP_quality)
        bd_ERP_vs_EAC = compute_bd_rate(exact_ERP_bitrates, ERP_quality, exact_EAC_bitrates, EAC_quality)


        bitrate_savings[video_name] = {}
        bitrate_savings[video_name][method_names["ERP vs. SFLP"]] = bd_ERP_vs_SFLP
        bitrate_savings[video_name][method_names["EAP vs. SFLP"]] = bd_EAP_vs_SFLP
        bitrate_savings[video_name][method_names["CMP vs. SFLP"]] = bd_CMP_vs_SFLP
        bitrate_savings[video_name][method_names["EAC vs. SFLP"]] = bd_EAC_vs_SFLP

        bitrate_savings[video_name][method_names["ERP vs. EAP"]] = bd_ERP_vs_EAP
        bitrate_savings[video_name][method_names["ERP vs. CMP"]] = bd_ERP_vs_CMP
        bitrate_savings[video_name][method_names["ERP vs. EAC"]] = bd_ERP_vs_EAC
        
        if bd_ERP_vs_SFLP != np.inf:
            bitrate_savings[macro_average_name][method_names["ERP vs. SFLP"]] += bd_ERP_vs_SFLP
            videos_calculated[macro_average_name][method_names["ERP vs. SFLP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["ERP vs. SFLP"]] += bd_ERP_vs_SFLP
                        videos_calculated[average_name][method_names["ERP vs. SFLP"]] += 1
        
        if bd_EAP_vs_SFLP != np.inf:
            bitrate_savings[macro_average_name][method_names["EAP vs. SFLP"]] += bd_EAP_vs_SFLP
            videos_calculated[macro_average_name][method_names["EAP vs. SFLP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["EAP vs. SFLP"]] += bd_EAP_vs_SFLP
                        videos_calculated[average_name][method_names["EAP vs. SFLP"]] += 1

        if bd_CMP_vs_SFLP != np.inf:
            bitrate_savings[macro_average_name][method_names["CMP vs. SFLP"]] += bd_CMP_vs_SFLP
            videos_calculated[macro_average_name][method_names["CMP vs. SFLP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["CMP vs. SFLP"]] += bd_CMP_vs_SFLP
                        videos_calculated[average_name][method_names["CMP vs. SFLP"]] += 1
        
        if bd_ERP_vs_EAP != np.inf:
            bitrate_savings[macro_average_name][method_names["ERP vs. EAP"]] += bd_ERP_vs_EAP
            videos_calculated[macro_average_name][method_names["ERP vs. EAP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["ERP vs. EAP"]] += bd_ERP_vs_EAP
                        videos_calculated[average_name][method_names["ERP vs. EAP"]] += 1

        if bd_ERP_vs_CMP != np.inf:
            bitrate_savings[macro_average_name][method_names["ERP vs. CMP"]] += bd_ERP_vs_CMP
            videos_calculated[macro_average_name][method_names["ERP vs. CMP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["ERP vs. CMP"]] += bd_ERP_vs_CMP
                        videos_calculated[average_name][method_names["ERP vs. CMP"]] += 1
        
        if bd_ERP_vs_EAC != np.inf:
            bitrate_savings[macro_average_name][method_names["ERP vs. EAC"]] += bd_ERP_vs_EAC
            videos_calculated[macro_average_name][method_names["ERP vs. EAC"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["ERP vs. EAC"]] += bd_ERP_vs_EAC
                        videos_calculated[average_name][method_names["ERP vs. EAC"]] += 1
    
    for average_name in average_names:
        if videos_calculated[average_name][method_names["ERP vs. SFLP"]] != 0:
            bitrate_savings[average_name][method_names["ERP vs. SFLP"]] /= videos_calculated[average_name][method_names["ERP vs. SFLP"]]
        else:
            bitrate_savings[average_name][method_names["ERP vs. SFLP"]] = "N/A"

        if videos_calculated[average_name][method_names["EAP vs. SFLP"]] != 0:
            bitrate_savings[average_name][method_names["EAP vs. SFLP"]] /= videos_calculated[average_name][method_names["EAP vs. SFLP"]]
        else:
            bitrate_savings[average_name][method_names["EAP vs. SFLP"]] = "N/A"
        
        if videos_calculated[average_name][method_names["CMP vs. SFLP"]] != 0:
            bitrate_savings[average_name][method_names["CMP vs. SFLP"]] /= videos_calculated[average_name][method_names["CMP vs. SFLP"]]
        else:
            bitrate_savings[average_name][method_names["CMP vs. SFLP"]] = "N/A"

        if videos_calculated[average_name][method_names["EAC vs. SFLP"]] != 0:
            bitrate_savings[average_name][method_names["EAC vs. SFLP"]] /= videos_calculated[average_name][method_names["EAC vs. SFLP"]]
        else:
            bitrate_savings[average_name][method_names["EAC vs. SFLP"]] = "N/A"

        if videos_calculated[average_name][method_names["ERP vs. EAP"]] != 0:
            bitrate_savings[average_name][method_names["ERP vs. EAP"]] /= videos_calculated[average_name][method_names["ERP vs. EAP"]]
        else:
            bitrate_savings[average_name][method_names["ERP vs. EAP"]] = "N/A"
        
        if videos_calculated[average_name][method_names["ERP vs. CMP"]] != 0:
            bitrate_savings[average_name][method_names["ERP vs. CMP"]] /= videos_calculated[average_name][method_names["ERP vs. CMP"]]
        else:
            bitrate_savings[average_name][method_names["ERP vs. CMP"]] = "N/A"
        
        if videos_calculated[average_name][method_names["ERP vs. EAC"]] != 0:
            bitrate_savings[average_name][method_names["ERP vs. EAC"]] /= videos_calculated[average_name][method_names["ERP vs. EAC"]]
        else:
            bitrate_savings[average_name][method_names["ERP vs. EAC"]] = "N/A"

    df_savings = pd.DataFrame.from_dict(bitrate_savings, orient="index")

    df_videos = df_savings[~df_savings.index.str.contains("Average|Macro")]

    # Now compute P25, P50, P75
    percentiles = df_videos.quantile([0.25, 0.5, 0.75])
    percentiles.index = ["P25", "P50", "P75"]

    # Append percentiles back to the original df_savings
    df_savings_with_percentiles = pd.concat([df_savings, percentiles])


    bitrate_savings_path = output_dir / f"bitrate_savings-{quality_info}.csv"
    df_savings_with_percentiles.to_csv(bitrate_savings_path)

    print(f"Bitrate savings saved to: {bitrate_savings_path}")
    print()

```

### Example Scripts

Under the 'examples' folder, the aforementioned functions and scripts are presented. To download the dataset used in the paper, download from the following links: [STJU](https://medialab.sjtu.edu.cn/post/sjtu-8k-360-video-sequences/), [Kandao-CableCar](https://drive.google.com/drive/folders/1kAZ2uefVQY_k3qmD3wo8JUQqcDEdw0ZV), [Kandao-DroneInDesert](https://drive.google.com/drive/folders/1rrmzGB9_G6FmU43g3ZMVKj4lTPixZJIj), [Kandao-BridgeBikeRide](https://drive.google.com/drive/folders/10T9IbzoVsZ7bqFFLyecw6Y9CAky12fok), [Kandao-Fireworks](https://drive.google.com/drive/folders/1lYAu255lBpCscPC-JH5lE7sE1UD4ZM3l).

## Project Structure

```
SFLP_toolkit/
├── src/      # Main package
│   └── utilities/      # Utility functions
├── tests/    # Test suite
├── docs/     # Documentation
├── examples/ # Example scripts
├── input/    # Sample input images
└── output/   # Output folder
    ├── images/         # Output images
    ├── layouts/        # Layouts for pixel arrangements
    └── SPSNR/          # Visual quality evaluation
```

## Custom Data Classes

Vector: This data class represents a 3-dimensional vector, where x is the front-back axis,
        y is the left-right axis, and z is the up-down axis.

GeographicPoint: This data class represents a spherical coordinate, where lat is the latitudinal coordinate,
             and lon is the longitudinal coordinate.

## Output

This toolkit includes functions to create:
- Create SFLP layout tables.
- Re-encoded images in .jpeg conversion.
- Re-encoded videos using H.264 (.mp4), H.265 (.mp4), or .AV1 (.webm) visual compression.
- Analysis tables on S-PSNR values for re-encoded videos.
- Analysis tables on BD-rate and bitrate savings based on S-PSNR values.
- Images of RD curves for re-encoded videos, plotting S-PSNR over bitrate.
- Viewport sampling for SFLP, ERP, EAP, CMP, and EAC.

## Configuration Options

To use the scripts in the 'examples' folder, configurations can be made in the 'config.py' script.

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## License

This project is licensed under a proprietary license - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{SFLP_toolkit,
  title = {SFLP Toolkit},
  author = {Chitsein Htun},
  year = {2026},
  url = {https://github.com/Chtun/SFLP_toolkit}
}
```

## Contact

For questions and support:
- Create an issue on GitHub
- Email: chtun@live.com

## Acknowledgments

This project is based on research by Chitsein Htun and builds upon the concepts presented in "Improving 360-degree Video Storage and Transmission: Spherical Fibonacci Lattice Pixels" published in the Proceedings of the ACM International Conference on Multimedia (MM) 2026.
