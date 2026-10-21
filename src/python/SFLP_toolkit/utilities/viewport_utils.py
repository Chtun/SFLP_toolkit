
"""Viewport Sampling module.

This module includes functions to evaluate the visual quality of different VR encoding methods (SFLP, ERP, CMP, EAC) for Image and Video.

Methods:

    sample_viewport: Samples a set of points on a viewport for a given input frame

"""

from enum import Enum
from pathlib import Path
import cupy as cp
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad

from SFLP_toolkit.utilities.file_conversion_utils import *

class WeightingType(Enum):
    EQUAL = "EQUAL"
    VIEWING_BEHAVIOR = "VIEWING_BEHAVIOR" # VIEWING_BEHAVIOR for weights based on where users typically look

class ComparisonMetric(Enum):
    SPSNR = "SPSNR"
    SSSIM = "SSSIM"


#
# region Viewport Sampling Functions
#

def get_viewport_geographic_points(
    viewport_direction: Tuple[float, float],
    viewport_angular_resolution: Tuple[float, float],
    viewport_resolution: Tuple[int, int],
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Computes the (lat, lon) geographic coordinates for every pixel in a viewport
    using a rectilinear planar projection.
    
    Args:
        viewport_direction: The latitude and longitude of the center of the viewport.
        viewport_angular_resolution: The angular resolution (total FOV in degrees) of the viewport.
        viewport_resolution: The resolution of the viewport (image width by height).

    Returns:
        A tuple of cp.ndarrays (new_lat, new_lon) of shape (height, width).
    """
    lat_center, lon_center = viewport_direction
    ang_res_h, ang_res_v = viewport_angular_resolution
    res_w, res_h = viewport_resolution

    # 1. Convert total field of view to radians
    fov_h_rad = cp.deg2rad(ang_res_h)
    fov_v_rad = cp.deg2rad(ang_res_v)

    # 2. Compute focal lengths based on image dimensions and FOV
    f_x = (res_w - 1 if res_w > 1 else 1.0) / (2.0 * cp.tan(fov_h_rad * 0.5))
    f_y = (res_h - 1 if res_h > 1 else 1.0) / (2.0 * cp.tan(fov_v_rad * 0.5))

    c_x = (res_w - 1) / 2.0
    c_y = (res_h - 1) / 2.0

    # Create pixel coordinate grid
    x_idx = cp.arange(res_w, dtype=cp.float64)
    y_idx = cp.arange(res_h, dtype=cp.float64)
    xv, yv = cp.meshgrid(x_idx, y_idx)

    # 3. Map pixel indices to normalized camera local ray directions (pinhole camera model)
    u = (xv - c_x) / f_x
    v = (c_y - yv) / f_y  # Invert Y so screen top maps to +Z

    x_3d = cp.ones_like(u)
    y_3d = u
    z_3d = v

    # Normalize local ray vectors
    inv_len = 1.0 / cp.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
    x_3d *= inv_len
    y_3d *= inv_len
    z_3d *= inv_len

    # 4. Apply Rotations to move center to (lat_center, lon_center)
    lat_rad = cp.deg2rad(lat_center)
    lon_rad = cp.deg2rad(lon_center)

    # Rotate by latitude (pitch around Y axis)
    x_3d_rot1 = x_3d * cp.cos(lat_rad) - z_3d * cp.sin(lat_rad)
    y_3d_rot1 = y_3d
    z_3d_rot1 = x_3d * cp.sin(lat_rad) + z_3d * cp.cos(lat_rad)

    # Rotate by longitude (yaw around Z axis)
    final_x = x_3d_rot1 * cp.cos(lon_rad) - y_3d_rot1 * cp.sin(lon_rad)
    final_y = x_3d_rot1 * cp.sin(lon_rad) + y_3d_rot1 * cp.cos(lon_rad)
    final_z = z_3d_rot1

    # 5. Convert back to Geographic (Lat, Lon) in degrees
    new_lat = cp.rad2deg(cp.arcsin(cp.clip(final_z, -1.0, 1.0)))
    new_lon = cp.rad2deg(cp.arctan2(final_y, final_x))

    return new_lat, new_lon

def sample_viewport(
    frame: cp.ndarray,
    frame_metadata: ProjectionMetadata,
    viewport_direction: Tuple[float, float],
    viewport_angular_resolution: Tuple[float, float],
    viewport_resolution: Tuple[int, int],
    sample_type: Image.Resampling = Image.Resampling.BILINEAR,
) -> cp.ndarray:
    """
    Samples a viewport on a single video frame.

    Args:
        frame: The 3D CuPy array for the RGB channels of the input frame.
        frame_metadata: The metadata of the input frame.
        viewport_direction: The latitude and longitude of the center of the viewport.
        viewport_angular_resolution: The angular resolution of the viewport.
        viewport_resolution: The resolution of the viewport (image width by height).
        sample_type: The type of sampling to conduct (nearest-neighbor or bilinear).

    Returns:
        A CuPy 3D array of RGB values for the selected viewport on the specified frame.
    """

    res_w, res_h = viewport_resolution

    geo_points_raw = get_viewport_geographic_points(
        viewport_direction, 
        viewport_angular_resolution, 
        viewport_resolution
    )

    geo_points = (
        geo_points_raw[0].flatten(), 
        geo_points_raw[1].flatten()
    )

    proj_type = frame_metadata.projection_type

    if proj_type == EncodingProjection.ERP:
        R, G, B = sample_ERP(geo_points, frame, sample_type)
    elif proj_type == EncodingProjection.EAP:
        R, G, B = sample_EAP(geo_points, frame, sample_type)
    elif proj_type == EncodingProjection.CMP:
        R, G, B = sample_CMP(geo_points, frame, sample_type)
    elif proj_type == EncodingProjection.EAC:
        R, G, B = sample_EAC(geo_points, frame, sample_type)
    elif proj_type == EncodingProjection.SFLP:
        if not isinstance(frame_metadata, SFLPMetadata):
            raise TypeError("Expected SFLPMetadata for SFLP projection")
        SFLP_metadata: SFLPMetadata = frame_metadata
        R, G, B = sample_SFLP(
            geographic_points=geo_points,
            input_image=frame,
            num_pixels=SFLP_metadata.num_pixels,
            arrangement_type=SFLP_metadata.arrangement_type,
            optimal_index_hops=SFLP_metadata.optimal_index_hops,
            first_element_SFLP_index=SFLP_metadata.first_element_SFLP_index,
            last_element_SFLP_index=SFLP_metadata.last_element_SFLP_index,
            non_padded_col_length=SFLP_metadata.non_padded_col_length,
            non_padded_row_length=SFLP_metadata.non_padded_row_length,
            sample_type=sample_type
        )        
    else:
        raise ValueError(f"Projection {proj_type} is not yet supported for viewport sampling.")

    viewport_rgb_combined = cp.stack([R, G, B], axis=-1)

    # Reshape back into the 2D viewport structure
    viewport_rgb = viewport_rgb_combined.reshape((res_h, res_w, 3))

    del geo_points_raw, geo_points
    del viewport_rgb_combined, R, G, B

    return viewport_rgb

#
# endregion Viewport Sampling Functions
#

#
# region Video/Image Interface Functions
#

def sample_video_frame_viewport(
    video_path: Path,
    video_metadata: ProjectionMetadata,
    frame_number: int,
    viewport_direction: Tuple[float, float],
    viewport_angular_resolution: Tuple[float, float],
    viewport_resolution: Tuple[int, int],
    sample_type: Image.Resampling = Image.Resampling.BILINEAR,
) -> cp.ndarray:
    """
    Samples a viewport on a single video frame.

    Args:
        video_path: The path to the input video.
        video_metadata: The metadata of the input video.
        frame_number: The frame number to sample the viewport on.
        viewport_direction: The latitude and longitude of the center of the viewport.
        viewport_angular_resolution: The angular resolution of the viewport.
        viewport_resolution: The resolution of the viewport (image width by height).
        sample_type: The type of sampling to conduct (nearest-neighbor or bilinear).

    Returns:
        A cp 3D array of RGB values for the selected viewport on the specified frame.
    """
    
    # Load Input Video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validation
    if frame_number >= frame_count:
        raise ValueError(f"Frame number {frame_number} exceeds video length ({frame_count} frames).")
    if frame_number < 0:
        raise ValueError(f"Frame number must be non-negative.")

    # Seek to Frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        raise IOError(f"Failed to read frame {frame_number} from {video_path}")

    # Pre-process Frame
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_cp = cp.asarray(frame_rgb)

    # Perform Sampling
    viewport_frame_cp = sample_viewport(
        frame=frame_cp,
        frame_metadata=video_metadata,
        viewport_direction=viewport_direction,
        viewport_angular_resolution=viewport_angular_resolution,
        viewport_resolution=viewport_resolution,
        sample_type=sample_type
    )

    return viewport_frame_cp

def save_video_frame_viewport(
    video_path: Path,
    video_metadata: ProjectionMetadata,
    frame_number: int,
    viewport_direction: Tuple[float, float],
    viewport_angular_resolution: Tuple[float, float],
    viewport_resolution: Tuple[int, int],
    output_image_path: Path,
    sample_type: Image.Resampling = Image.Resampling.BILINEAR,
) -> cp.ndarray:

    # Perform Sampling
    viewport_frame_cp = sample_video_frame_viewport(
        video_path=video_path,
        video_metadata=video_metadata,
        frame_number=frame_number,
        viewport_direction=viewport_direction,
        viewport_angular_resolution=viewport_angular_resolution,
        viewport_resolution=viewport_resolution,
        sample_type=sample_type,
    )

    viewport_frame_np = cp.asnumpy(viewport_frame_cp)

    # Clip and cast to uint8
    viewport_frame_uint8 = np.clip(viewport_frame_np, 0, 255).astype(np.uint8)

    # Save using Pillow
    img_out = Image.fromarray(viewport_frame_uint8, mode="RGB")
    img_out.save(output_image_path)
    
    print(f"Viewport frame {frame_number} saved to {output_image_path}")

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


#
# endregion Video/Image Interface Functions
#