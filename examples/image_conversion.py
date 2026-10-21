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

SFLP_num_pixels = 18000001
arrangement_type = SFLPArrangementType.STANDARD

SFLP_layout_csv_path = layout_input_dir / f"SFLP_arrangement_{arrangement_type.name}-360_No_Tiling-{SFLP_num_pixels}.csv"
SFLP_output_path = output_dir / f"SFLP-{input_image_name}-{SFLP_num_pixels}-{arrangement_type.name}.jpeg"

# CubeMap (CMP and EAC)

CubeMap_num_pixels = 17998944

if math.sqrt(CubeMap_num_pixels / 6) != int(math.sqrt(CubeMap_num_pixels / 6)):
    raise ValueError("Invalid number of CMP pixels")

face_size = int(math.sqrt(CubeMap_num_pixels / 6))

CubeMap_layout_csv_path = layout_input_dir / f"CubeMap_pixel_arrangement-360_No_Tiling-{CubeMap_num_pixels}.csv"
CMP_output_path = output_dir / f"CMP-{input_image_name}-{CubeMap_num_pixels}.jpeg"
EAC_output_path = output_dir / f"EAC-{input_image_name}-{CubeMap_num_pixels}.jpeg"

# ERP and EAP

ERP_size = (6000, 3000)
ERP_num_pixels = (ERP_size[0] * ERP_size[1])
ERP_output_path = output_dir / f"Reencoded-{input_image_name}-{ERP_num_pixels}.jpeg"
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