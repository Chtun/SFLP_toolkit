import numpy as np
import pandas as pd
from pathlib import Path

from SFLP_toolkit.utilities.pixel_arrangement_utils import generate_CubeMap_pixel_arrangement

output_dir = Path("../output/layouts")
input_dir = Path("../input")

num_pixels_horizontal = 1732
num_pixels_vertical = num_pixels_horizontal
Cube_num_points = num_pixels_vertical * num_pixels_horizontal * 6

Cube_frame_df = generate_CubeMap_pixel_arrangement(num_pixels_horizontal=num_pixels_horizontal, num_pixels_vertical=num_pixels_vertical)
Cube_arrangement_save_path = output_dir / f"CubeMap_pixel_arrangement-360_No_Tiling-{Cube_num_points}.csv"
Cube_frame_df.to_csv(Cube_arrangement_save_path, index=False, header=False)

print(f"CubeMap layout saved to : {Cube_arrangement_save_path}")
