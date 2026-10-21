"""File Conversion module.

This module includes functions to convert from popular VR encoding methods (ERP, CMP, EAC) to the SFLP method for Image and Video.

Methods:
    get_CubeMap_geographic_points: Converts CubeMap layout tokens to geographic coordinates (latitude, longitude).
    get_EAP_geographic_points: Grabs the geographic coordinates (latitude, longitude) for EAP pixel centers.

    geographic_to_ERP_index: Convert lat/lon to floating point ERP image index.
    geographic_to_ERP_index_float: Convert lat/lon to floating point ERP image coordinates.
    map_vectors_to_SFLP_indices: GPU-parallelized function that maps each input vector to its nearest SFLP index (or four, if bilinear sampling).

    sample_ERP: Samples from a ERP image given a set of input geographic points.
    sample_CMP: Samples from a CMP image given a set of input geographic points.
    sample_EAC: Samples from an EAC image given a set of input geographic points.
    sample_EAP: Samples from an EAC image given a set of input geographic points.
    sample_SFLP: Samples from a SFLP image given a set of input geographic points.
    
    ERP_to_SFLP_image_conversion: Converts an ERP image to an SFLP image.
    ERP_to_CMP_image_conversion: Converts an ERP image to a CMP image.
    ERP_to_EAC_image_conversion: Converts an ERP image to an EAC image.
    ERP_to_EAP_image_conversion: Converts an ERP image to an EAP image.
    CMP_to_SFLP_image_conversion: Converts a CMP image to an SFLP image.
    EAC_to_SFLP_image_conversion: Converts an EAC image to an SFLP image.
    EAP_to_SFLP_image_conversion: Converts an EAP image to an SFLP image.
    SFLP_to_CMP_image_conversion: Converts an SFLP image to an CMP image.
    reencode_ERP_image: Resizes and re-encodes an ERP image to a specified size.

    ERP_to_SFLP_video_conversion: Converts an ERP video to an SFLP video.
    ERP_to_CMP_video_conversion: Converts an ERP video to a CMP video.
    ERP_to_EAC_video_conversion: Converts an ERP video to a EAC video.
    ERP_to_EAP_video_conversion: Converts an ERP video to an EAP video.
    reencode_ERP_video: Re-encodes an ERP video to a standardized format and resolution.

"""

from enum import Enum
from pathlib import Path
from PIL import Image
import cupy as cp
import pandas as pd
import numpy as np
import cv2
from typing import Optional, Tuple, List
import subprocess
import time

from SFLP_toolkit.utilities.pixel_arrangement_utils import *


class EncodingProjection(Enum):
    ERP = "EQUIRECTANGULAR_PROJECTION"
    CMP = "CUBEMAP_PROJECTION"
    EAC = "EQUIANGULAR_CUBEMAP"
    EAP = "CYLINDRICAL_EQUAL_AREA_PROJECTION"
    SFLP = "SPHERICAL_FIBONACCI_LATTICE"

class VideoEncoder(Enum):
    AV1 = "AV1"
    H264 = "H264"
    H265 = "H265"

class ProjectionMetadata():
    def __init__(self, projection_type: EncodingProjection):
        self.projection_type = projection_type

class SFLPMetadata(ProjectionMetadata):
    def __init__(self, num_pixels, arrangement_type, optimal_index_hops, first_element_SFLP_index, last_element_SFLP_index, padding_size: int = -1):
        super().__init__(EncodingProjection.SFLP)
        self.num_pixels = num_pixels
        self.arrangement_type = arrangement_type
        self.optimal_index_hops = optimal_index_hops
        self.first_element_SFLP_index = first_element_SFLP_index
        self.last_element_SFLP_index = last_element_SFLP_index

        if self.arrangement_type == SFLPArrangementType.STANDARD:
            if padding_size < 0:
                raise ValueError("Padding size must be provided for STANDARD arrangement!")
            self.non_padded_row_length = self.optimal_index_hops[1][0]
            self.non_padded_col_length = math.floor(self.num_pixels / self.non_padded_row_length)
            self.non_padded_col_length = self.non_padded_col_length + (padding_size - (self.non_padded_col_length % padding_size))
        elif self.arrangement_type == SFLPArrangementType.SEAM_REMOVED:
            self.non_padded_row_length = self.optimal_index_hops[1][0]
            self.non_padded_col_length = math.ceil(self.num_pixels / self.non_padded_row_length)
        elif self.arrangement_type == SFLPArrangementType.ZIGZAG:
            self.non_padded_row_length = math.ceil((self.optimal_index_hops[0][0] + self.optimal_index_hops[1][0]) / 2)
            self.non_padded_col_length = 2 * math.ceil(self.num_pixels / self.non_padded_row_length)
        else:
            raise ValueError(f"SFLP Arrangement: {self.arrangement_type} is not supported!")


    def __str__(self):

        str_output = f"Number of pixels: {self.num_pixels}\n"
        str_output += f"Arrangement Type: {self.arrangement_type}\n"
        str_output += f"Optimal index hops: {self.optimal_index_hops}\n"
        str_output += f"SFLP index of first element in first column: {self.first_element_SFLP_index}\n"
        str_output += f"SFLP index of last element in first column: {self.last_element_SFLP_index}\n"
        str_output += f"Length of longest column (excluding padded row): {self.non_padded_col_length}\n"
        str_output += f"Length of longest row (excluding padded column): {self.non_padded_row_length}\n"
        
        return str_output
    

#
# region Sampling Functions
#

# region Geographic Coordinates to Image Data

def geographic_points_to_image_info(
    geographic_points: tuple[cp.ndarray, cp.ndarray],
    input_image: cp.ndarray,
    input_metadata: ProjectionMetadata,
    sample_type: Image.Resampling = Image.Resampling.NEAREST
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Vectorized, GPU-parallel function to convert geographic points to ERP image RGB values.

    Args:
        geographic_points: (N,) tuple of two CuPy arrays of latitude and longitude for set of geographic points.
        input_image: (H, W, 3) CuPy array
        input_metadata: The metadata for the input image.
        first_element_SFLP_index, last_element_SFLP_index: SFLP index of first element and last element of first column.

    Returns:
        R, G, B: CuPy arrays of shape (N,)
    """
    input_projection_type = input_metadata.projection_type

    R, G, B = [], [] ,[]

    # Handle ERP
    if input_projection_type == EncodingProjection.ERP:
        R, G, B = sample_ERP(geographic_points=geographic_points, input_image=input_image, sample_type=sample_type)
    elif input_projection_type == EncodingProjection.CMP:
        R, G, B = sample_CMP(geographic_points=geographic_points, input_image=input_image, sample_type=sample_type)
    elif input_projection_type == EncodingProjection.EAC:
        R, G, B = sample_EAC(geographic_points=geographic_points, input_image=input_image, sample_type=sample_type)
    elif input_projection_type == EncodingProjection.EAP:
        R, G, B = sample_EAP(geographic_points=geographic_points, input_image=input_image, sample_type=sample_type)
    elif input_projection_type == EncodingProjection.SFLP:
        if not isinstance(input_metadata, SFLPMetadata):
            raise TypeError("Expected SFLPMetadata for SFLP projection")
        SFLP_metadata: SFLPMetadata = input_metadata
        R, G, B = sample_SFLP(
            geographic_points=geographic_points,
            input_image=input_image,
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
        raise ValueError(f"Input Projection Type {input_projection_type} is not supported for conversion to SFLP!")

    return R, G, B

def sample_ERP(
    geographic_points: Tuple[cp.ndarray, cp.ndarray],
    input_image: cp.ndarray,
    sample_type: Image.Resampling = Image.Resampling.NEAREST
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Samples from a ERP image given a set of input geographic points.
    
    Args:
        geographic_points (N, 2): CuPy arrays of latitudes and longitudes for a set of points.
        input_image: (H, W, 3) CuPy ERP image.
        sample_type: The type of sampling to do on the ERP image data.

    Returns:
        R, G, B: CuPy arrays of shape (N,)
    """
    H, W = input_image.shape[:2]

    R, G, B = [], [], []

    # Convert to ERP image coordinates
    if sample_type == Image.Resampling.NEAREST:
        lat, lon = geographic_points
        row_idx, col_idx = geographic_to_ERP_index(lat, lon, H, W)

        # Sample the image (N, 3)
        pixels = input_image[row_idx, col_idx]
        R = pixels[:, 0]
        G = pixels[:, 1]
        B = pixels[:, 2]
    elif sample_type == Image.Resampling.BILINEAR:
        R, G, B = sample_ERP_bilinear(geographic_points=geographic_points, input_image=input_image)
    else:
        raise ValueError(f"Sample type ({sample_type}) not supported!")
    
    return R, G, B

#endregion Geographic Coordinates to Image Data

# region ERP Coordinates

def get_ERP_geographic_points(
    frame_dimensions: Tuple[int, int]
) -> Tuple[cp.ndarray, cp.ndarray]:
    W, H = frame_dimensions

    # 1. Generate 1D vectors for the center of each pixel column/row
    # Longitude: Linear from -180 to 180
    longs = cp.linspace(-180, 180, W, endpoint=False)
    
    # Latitude: Linear from 90 to -90 (Top to Bottom)
    lats = cp.linspace(90, -90, H, endpoint=False)

    # 2. Use broadcasting to create the flattened arrays
    # longitudes_deg repeats the row for every latitude
    # latitudes_deg repeats the column for every longitude
    longitudes_deg = cp.tile(longs, H)
    latitudes_deg = cp.repeat(lats, W)

    return latitudes_deg, longitudes_deg

# endregion ERP Coordinates

# region CMP Coordinates

def get_CubeMap_geographic_points(
    CubeMap_tokens: np.ndarray,
    face_size: int,
    is_CMP: bool
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Converts CubeMap layout tokens to geographic coordinates (latitude, longitude).

    Args:
        CubeMap_tokens (np.ndarray): Flattened NumPy array of strings like 'Z=-1_4_0'.
        face_size (int): Horizontal/Vertical resolution of a face.
        is_CMP (bool): True if CMP, False if EAC.

    Returns:
        Tuple[cp.ndarray, cp.ndarray]: (latitudes, longitudes) in degrees.
    """
    face_ids, row_local, col_local = [], [], []

    # Parse the EAC tokens
    for token in CubeMap_tokens:
        face_part, row_str, col_str = token.split("_")
        face_ids.append(face_part)
        row_local.append(int(row_str))
        col_local.append(int(col_str))

    row_local = cp.asarray(row_local)
    col_local = cp.asarray(col_local)

    # Convert pixel indices to angular positions (Cubemap Proejction/Equi-Angular Cubemap mapping)
    if is_CMP:
        u, v = CMP_indices_to_uv(row_local=row_local, col_local=col_local, face_size=face_size)
    else:
        u, v = EAC_indices_to_uv(row_local, col_local, face_size)

    u_cpu = cp.asnumpy(u)
    v_cpu = cp.asnumpy(v)

    # Once the face side and each u, v coordinate is known, get the x, y, and z coordinates.
    x, y, z = [], [], []
    for i, face in enumerate(face_ids):
        xi, yi, zi = CubeMap_id_to_vector(face, u_cpu[i], v_cpu[i])
        x.append(xi)
        y.append(yi)
        z.append(zi)

    x = cp.asarray(x)
    y = cp.asarray(y)
    z = cp.asarray(z)

    norm = cp.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm

    # convert the x, y, and z coordinates to longitude and latitude coordinates.
    lon = cp.arctan2(y, x)
    lat = cp.arcsin(z)       

    return cp.degrees(lat), cp.degrees(lon)

def CubeMap_id_to_vector(face: str, u: float, v: float) -> Tuple[float, float, float]:
    """
    Converts (face index, u, v) to 3D unit vectors on the sphere based on custom CubeMap layout.

    Args:
        face: string for face ID.
        u, v: normalized coordinates in [-1, 1]

    Returns:
        (N, 3) CuPy array of unit vectors
    """
    if face == "X=+1":
        return (1, -u, v)
    elif face == "X=-1":
        return (-1, u, v)
    elif face == "Y=+1":
        return (u, 1, v)
    elif face == "Y=-1":
        return (v, -1, -u)
    elif face == "Z=+1":
        return (v, -u, 1)
    elif face == "Z=-1":
        return (v, u, -1)
    else:
        raise ValueError(f"Invalid face: {face}")

def CubeMap_ids_to_vectors(face_indices: cp.ndarray, u: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
    """
    Converts (face index, u, v) to 3D unit vectors on the sphere based on custom CubeMap layout (GPU-compatible).

    Args:
        face_indices: (N,) CuPy array with values 0 to 5
        u, v: (N,) CuPy arrays of normalized coordinates in [-1, 1]

    Returns:
        (N, 3) CuPy array of unit vectors
    """
    x = cp.zeros_like(u)
    y = cp.zeros_like(u)
    z = cp.zeros_like(u)

    # Face 2 → "X=+1" → (1, u, v)
    mask = face_indices == 0
    x[mask] = 1
    y[mask] = -u[mask]
    z[mask] = v[mask]

    # Face 0 → "X=-1" → (-1, -u, v)
    mask = face_indices == 2
    x[mask] = -1
    y[mask] = u[mask]
    z[mask] = v[mask]

    # Face 1 → "Y=+1" → (-u, 1, v)
    mask = face_indices == 1
    x[mask] = u[mask]
    y[mask] = 1
    z[mask] = v[mask]

    # Face 4 → "Y=-1" → (v, -1, -u)
    mask = face_indices == 4
    x[mask] = v[mask]
    y[mask] = -1
    z[mask] = -u[mask]

    # Face 5 → "Z=+1" → (v, -u, 1)
    mask = face_indices == 5
    x[mask] = v[mask]
    y[mask] = -u[mask]
    z[mask] = 1

    # Face 3 → "Z=-1" → (v, u, -1)
    mask = face_indices == 3
    x[mask] = v[mask]
    y[mask] = u[mask]
    z[mask] = -1

    # Normalize to unit vectors
    vec = cp.stack([x, y, z], axis=1)
    norm = cp.linalg.norm(vec, axis=1, keepdims=True)
    return vec / (norm + 1e-8)

def CMP_indices_to_uv(row_local: cp.ndarray, col_local: cp.ndarray, face_size: int) -> cp.ndarray:
    """
    Converts CMP indices to uv coordinates.

    Args:
        row_local: CuPy array of the row index for each pair.
        col_local: CuPy array of the column index for each pair.
        face_size: The number of pixels in a row/column of the face.

    Returns:
        u, v: The u, v coordinates for the CMP indices.
    """
    u = -(col_local + 0.5) / face_size * 2 + 1
    v = -(row_local + 0.5) / face_size * 2 + 1

    return u, v

# endregion CMP

# region EAC Coordinates

def EAC_indices_to_uv(row_local: cp.ndarray, col_local: cp.ndarray, face_size: int) -> cp.ndarray:
    """
    Converts EAC indices to uv coordinates.

    Args:
        row_local: CuPy array of the row index for each pair.
        col_local: CuPy array of the column index for each pair.
        face_size: The number of pixels in a row/column of the face.

    Returns:
        u, v: The u, v coordinates for the EAC indices.
    """
    pi_quarter = cp.pi / 4
    u = cp.tan((-(col_local + 0.5) / face_size * 2 + 1) * pi_quarter)
    v = cp.tan((-(row_local + 0.5) / face_size * 2 + 1) * pi_quarter)

    return u, v

# endregion EAC

# region EAP Coordinates

def get_EAP_geographic_points(
    frame_dimensions: Tuple[int, int]
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Grabs the geographic coordinates (latitude, longitude) for EAP pixel centers.

    Args:
        frame_dimensions: The width and height of the frame.

    Returns:
        Tuple[cp.ndarray, cp.ndarray]: (latitudes, longitudes) in degrees.
    """
    W, H = frame_dimensions

    # Generate pixel coordinates for the EAP frame
    x = cp.arange(W)
    y = cp.arange(H)
    
    # Create 2D grids for the x and y coordinates
    xx, yy = cp.meshgrid(x, y)
    
    # Normalize coordinates to the range [-1, 1]
    # The y-axis is inverted to match standard latitude orientation (bottom to top)
    normalized_x = (2.0 * xx + 1.0) / W - 1.0
    normalized_y = 1.0 - (2.0 * yy + 1.0) / H
    
    # Perform the inverse EAP projection
    longitudes_rad = cp.pi * normalized_x
    latitudes_rad = cp.arcsin(normalized_y)
    
    # Convert radians to degrees and flatten the arrays
    latitudes_deg = cp.degrees(latitudes_rad.flatten())
    longitudes_deg = cp.degrees(longitudes_rad.flatten())
    
    return latitudes_deg, longitudes_deg

# endregion EAP

# region Sampling From ERP Helpers

def geographic_to_ERP_index(lat_deg: cp.ndarray, lon_deg: cp.ndarray, height: int, width: int):
    """
    Convert lat/lon to floating point ERP image index.

    Args:
        lat: latitude in radians, shape (N,)
        lon: longitude in radians, shape (N,)
        height: image height
        width: image width

    Returns:
        row_idx, col_idx: The nearest (floor) row and column indices into the ERP image
    """
    row = (90.0 - lat_deg) / 180.0 * height
    col = (lon_deg + 180.0) / 360.0 * width

    row_idx = cp.clip(cp.floor(row).astype(cp.int32), 0, height - 1)
    col_idx = cp.clip(cp.floor(col).astype(cp.int32), 0, width - 1)

    return row_idx, col_idx

def geographic_to_ERP_index_float(lat_deg: cp.ndarray, lon_deg: cp.ndarray, height: int, width: int):
    """
    Convert lat/lon to floating point ERP image coordinates.

    Args:
        lat: latitude in radians, shape (N,)
        lon: longitude in radians, shape (N,)
        height: image height
        width: image width

    Returns:
        row_f, col_f: float row and column indices into the ERP image
    """
    row_f = (90.0 - lat_deg) / 180.0 * height
    col_f = (lon_deg + 180.0) / 360.0 * width
    return row_f, col_f

def sample_ERP_bilinear_helper(input_image: cp.ndarray, row_f: cp.ndarray, col_f: cp.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """
    Performs bilinear sampling from a CuPy image of an ERP image from row and column floats.

    Args:
        input_image: (H, W, 3) CuPy image array of ERP image.
        row_f: (N,) float row indices.
        col_f: (N,) float col indices.

    Returns:
        R, G, B: (N,) interpolated channels.
    """
    H, W = input_image.shape[:2]
    N = row_f.shape[0]

    time_one = time.perf_counter()

    # Get integer floor and clipped ceil indices
    row0 = cp.clip(cp.floor(row_f).astype(cp.int32), 0, H - 1)
    row1 = cp.clip(row0 + 1, 0, H - 1)
    col0 = cp.clip(cp.floor(col_f).astype(cp.int32), 0, W - 1)
    col1 = cp.clip(col0 + 1, 0, W - 1)

    # Stack corners for vectorized processing: shape (N,4)
    rows = cp.stack([row0, row0, row1, row1], axis=1)
    cols = cp.stack([col0, col1, col0, col1], axis=1)

    time_two = time.perf_counter()

    # Sample pixel colors for all corners at once: (N,4,3)
    pixels = input_image[rows, cols]  # advanced indexing on GPU

    time_three = time.perf_counter()

    # Convert the samples to phi and theta
    phi_samples, theta_samples = ERP_index_to_polar_azimuthal(row_f, col_f, H, W)

    # Convert the corners to phi and theta
    phi_corners, theta_corners = ERP_index_to_polar_azimuthal(rows, cols, H, W)

    time_four = time.perf_counter()

    # Broadcast and compute cosine theta:
    cos_theta = (
        cp.sin(phi_samples[:, None]) * cp.sin(phi_corners) +
        cp.cos(phi_samples[:, None]) * cp.cos(phi_corners) *
        cp.cos(theta_samples[:, None] - theta_corners)
    )

    # Clamp and get angular distances
    thetas = cp.arccos(cp.clip(cos_theta, -1, 1))

    # mean = cp.mean(thetas)
    # std = cp.std(thetas)

    # print(f"Mean of theta: {mean}")
    # print(f"Standard deviation of theta: {std}")

    time_five = time.perf_counter()

    eps = 1e-6
    inv_thetas = 1.0 / (thetas + eps)

    # Normalize weights (N,4)
    weights = inv_thetas / cp.sum(inv_thetas, axis=1, keepdims=True)

    time_six = time.perf_counter()

    # Interpolate RGB: multiply weights with pixel colors and sum over corners
    neighbors = cp.sum(pixels * weights[:, :, None], axis=1)  # (N,3)

    time_seven = time.perf_counter()

    # Clamp RGB output
    final = cp.clip(neighbors, 0, 255)

    time_eight = time.perf_counter()

    # print(f"Time between time_two and time_one: {time_two - time_one}")
    # print(f"Time between time_three and time_two: {time_three - time_two}")
    # print(f"Time between time_four and time_three: {time_four - time_three}")
    # print(f"Time between time_five and time_four: {time_five - time_four}")
    # print(f"Time between time_six and time_five: {time_six - time_five}")
    # print(f"Time between time_seven and time_six: {time_seven - time_six}")
    # print(f"Time between time_eight and time_seven: {time_eight - time_seven}")

    return final[:, 0], final[:, 1], final[:, 2]

def ERP_index_to_polar_azimuthal(row: cp.ndarray, col: cp.ndarray, height: int, width: int) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Converts ERP image row/column indices to spherical polar (phi) and azimuthal (theta) angles.
    
    Args:
        row: Row index(es) from the ERP image.
        col: Column index(es) from the ERP image.
        height: Height of the ERP image.
        width: Width of the ERP image.
        
    Returns:
        Tuple[cp.ndarray, cp.ndarray]: (phi, theta) in radians.
                                      phi: polar angle (0 to pi, from top to bottom)
                                      theta: azimuthal angle (0 to 2*pi, around the equator)
    """
    # Calculate phi (polar angle)
    # Maps row from [0, H] to [0, pi]
    phi = cp.pi * (1 - row / height)

    # Calculate theta (azimuthal angle)
    # Maps col from [0, W] to [0, 2*pi]
    theta = 2 * cp.pi * col / width
    
    return phi, theta

def sample_ERP_bilinear(geographic_points: cp.ndarray, input_image: cp.ndarray):
    """
    Samples from an ERP image using bilinear sampling given a set of input geographic points.

    Args:
        geographic_points: The geographic points to sample on.
        input_image: The input ERP image to sample from.
    
    Returns:
        R, G, B: (N,) interpolated channels.
    """
    
    start_time = time.perf_counter()

    H, W = input_image.shape[:2]
    
    lat_deg, lon_deg = geographic_points

    row_f, col_f = geographic_to_ERP_index_float(lat_deg=lat_deg, lon_deg=lon_deg, height=H, width=W)

    middle_pre_delete_time = time.perf_counter()

    del lat_deg, lon_deg, geographic_points

    middle_time = time.perf_counter()

    R, G, B = sample_ERP_bilinear_helper(input_image=input_image, row_f=row_f, col_f=col_f)

    end_time = time.perf_counter()

    # print(f"Time to grab ERP index: {middle_pre_delete_time - start_time} seconds")
    # print(f"Time to mark old memory as rewritable: {middle_time - middle_pre_delete_time}")
    # print(f"Time to sample ERP bilinear: {end_time - start_time} seconds")

    return R, G, B

# endregion ERP Sampling Helpers

# region Sampling From CMP Helpers

def get_CubeMap_face_and_uv(vectors: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Converts 3D unit vectors to CubeMap face index and face-local (u, v) ∈ [-1, 1].
    Follows the custom face convention from face_uv_to_spherical_coords.
    
    Args:
        vectors: (N, 3) A list of vectors to get the CubeMap face and (u, v) for.

    Returns:
        face_indices: (N,) in [0,5]
        u: (N,) ∈ [-1, 1]
        v: (N,) ∈ [-1, 1]
    """
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    abs_x, abs_y, abs_z = cp.abs(x), cp.abs(y), cp.abs(z)

    is_x_major = (abs_x >= abs_y) & (abs_x >= abs_z)
    is_y_major = (abs_y > abs_x) & (abs_y >= abs_z)
    is_z_major = ~(is_x_major | is_y_major)

    face_indices = cp.full(x.shape, -1, dtype=cp.int32)
    u = cp.zeros_like(x)
    v = cp.zeros_like(x)

    # === X faces ===
    x_pos = is_x_major & (x > 0)
    x_neg = is_x_major & (x < 0)

    face_indices = cp.where(x_pos, 0, face_indices)  # X = +1
    u = cp.where(x_pos, -y / abs_x, u)
    v = cp.where(x_pos, z / abs_x, v)

    face_indices = cp.where(x_neg, 2, face_indices)  # X = -1
    u = cp.where(x_neg, y / abs_x, u)
    v = cp.where(x_neg, z / abs_x, v)

    # === Y faces ===
    y_pos = is_y_major & (y > 0)
    y_neg = is_y_major & (y < 0)

    face_indices = cp.where(y_pos, 1, face_indices)  # Y = +1
    u = cp.where(y_pos, x / abs_y, u)
    v = cp.where(y_pos, z / abs_y, v)

    face_indices = cp.where(y_neg, 4, face_indices)  # Y = -1
    u = cp.where(y_neg, -z / abs_y, u)
    v = cp.where(y_neg, x / abs_y, v)

    # === Z faces ===
    z_pos = is_z_major & (z > 0)
    z_neg = is_z_major & (z < 0)

    face_indices = cp.where(z_pos, 5, face_indices)  # Z = +1
    u = cp.where(z_pos, -y / abs_z, u)
    v = cp.where(z_pos, x / abs_z, v)

    face_indices = cp.where(z_neg, 3, face_indices)  # Z = -1
    u = cp.where(z_neg, y / abs_z, u)
    v = cp.where(z_neg, x / abs_z, v)

    return face_indices, u, v

def map_face_uv_to_CMP_row_col(
    face_indices: cp.ndarray,
    u: cp.ndarray,
    v: cp.ndarray,
    face_size: int
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Maps (face_index, u, v) in [-1, 1] to global CMP (row, col) image coordinates.

    Args:
        face_indices: (N,) array of face indices ∈ [0,5]
        u, v: (N,) arrays of local face coordinates ∈ [-1, 1]
        face_size: int, size (in pixels) of one cube face

    Returns:
        (row_f, col_f): (N,) arrays of image-space coordinates
    """
    # CubeMap layout:
    #  [ (0,0) X=+1 , (0,1) Y=+1 , (0,2) X=-1 ]
    #  [ (1,0) Z=-1 , (1,1) Y=-1 , (1,2) Z=+1 ]
    face_row = cp.array([0, 0, 0, 1, 1, 1], dtype=cp.int32)
    face_col = cp.array([0, 1, 2, 0, 1, 2], dtype=cp.int32)

    row_offsets = face_row[face_indices] * face_size
    col_offsets = face_col[face_indices] * face_size

    # Rescale from [-1, 1] to pixel index range [0, face_size - 1]
    col_local = (u - 1.0) * -0.5 * (face_size) - 0.5
    row_local = (v - 1.0) * -0.5 * (face_size) - 0.5

    col_indices = col_offsets + col_local
    row_indices = row_offsets + row_local

    return row_indices, col_indices

def CMP_pixel_to_cartesian_coords(row: cp.ndarray, col: cp.ndarray, face_size: int) -> cp.ndarray:
    """
    Converts CMP row/column indices to cartesian coordinates.
    
    Args:
        row: (N,) CuPy array of row indices.
        col: (N,) CuPy array of column indices.
        face_size: The number of pixels in a row/col of a face.
    
    Returns:
        (N, 3) CuPy array of unit vectors.
    """
    face_row = row // face_size
    face_col = col // face_size
    face_indices = face_row * 3 + face_col

    row_local = (row % face_size).astype(cp.float32)
    col_local = (col % face_size).astype(cp.float32)
    
    u_local, v_local = CMP_indices_to_uv(row_local, col_local, face_size)

    return CubeMap_ids_to_vectors(face_indices, u_local, v_local)

def EAC_pixel_to_cartesian_coords(row: cp.ndarray, col: cp.ndarray, face_size: int) -> cp.ndarray:
    """
    Converts EAC row/column indices to cartesian coordinates.
    
    Args:
        row: (N,) CuPy array of row indices.
        col: (N,) CuPy array of column indices.
        face_size: The number of pixels in a row/col of a face.
    
    Returns:
        (N, 3) CuPy array of unit vectors.
    """
    face_row = row // face_size
    face_col = col // face_size
    face_indices = face_row * 3 + face_col

    # Local pixel coordinates within the face
    row_local = (row % face_size).astype(cp.float32)
    col_local = (col % face_size).astype(cp.float32)

    u_local, v_local = EAC_indices_to_uv(row_local, col_local, face_size)

    return CubeMap_ids_to_vectors(face_indices, u_local, v_local)

def sample_CubeMap_bilinear_helper(
    row_f: cp.ndarray,
    col_f: cp.ndarray,
    vectors: cp.ndarray,
    input_image: cp.ndarray,
    face_size: int,
    is_CMP: bool
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Performs spherical-aware bilinear sampling on a CMP or EAC image using angular distances.

    Args:
        row_f, col_f: (N,) fractional row/col pixel indices into CMP or EAC image
        vectors: (N, 3) unit vectors on the sphere corresponding to sampling points
        input_image: (H, W, 3) CMP or EAC image
        face_size: int, size of one cube face in pixels
        is_CMP: True if the input image is in CMP, False if in EAC.

    Returns:
        result: (N, 3) interpolated RGB values
    """
    H, W = input_image.shape[:2]

    # Nearest 4 surrounding pixels
    row0 = cp.floor(row_f).astype(cp.int32)
    row1 = cp.clip(row0 + 1, 0, H - 1)
    row0 = cp.clip(row0, 0, H - 1)

    col0 = cp.floor(col_f).astype(cp.int32)
    col1 = cp.clip(col0 + 1, 0, W - 1)
    col0 = cp.clip(col0, 0, W - 1)

    # Sample corner pixels (N, 3)
    top_left     = input_image[row0, col0]
    top_right    = input_image[row0, col1]
    bottom_left  = input_image[row1, col0]
    bottom_right = input_image[row1, col1]

    # Get sphere points for each corner
    if is_CMP:
        p_tl = CMP_pixel_to_cartesian_coords(row0, col0, face_size)
        p_tr = CMP_pixel_to_cartesian_coords(row0, col1, face_size)
        p_bl = CMP_pixel_to_cartesian_coords(row1, col0, face_size)
        p_br = CMP_pixel_to_cartesian_coords(row1, col1, face_size)
    else:
        p_tl = EAC_pixel_to_cartesian_coords(row0, col0, face_size)
        p_tr = EAC_pixel_to_cartesian_coords(row0, col1, face_size)
        p_bl = EAC_pixel_to_cartesian_coords(row1, col0, face_size)
        p_br = EAC_pixel_to_cartesian_coords(row1, col1, face_size)

    # Compute angular distances using 1 - cos(θ)
    dot_tl = cp.einsum('ij,ij->i', vectors, p_tl)
    dot_tr = cp.einsum('ij,ij->i', vectors, p_tr)
    dot_bl = cp.einsum('ij,ij->i', vectors, p_bl)
    dot_br = cp.einsum('ij,ij->i', vectors, p_br)

    # Clamp to prevent NaNs from floating-point imprecision
    dot_tl = cp.clip(dot_tl, -1.0, 1.0)
    dot_tr = cp.clip(dot_tr, -1.0, 1.0)
    dot_bl = cp.clip(dot_bl, -1.0, 1.0)
    dot_br = cp.clip(dot_br, -1.0, 1.0)

    # Angular distances in radians
    theta_tl = cp.arccos(dot_tl)
    theta_tr = cp.arccos(dot_tr)
    theta_bl = cp.arccos(dot_bl)
    theta_br = cp.arccos(dot_br)

    # Stack into shape (4, N) then take min across axis 0 → result shape (N,)
    thetas = cp.stack([theta_tl, theta_tr, theta_bl, theta_br])
    min_thetas = cp.min(thetas, axis=0)

    # print("Smallest angular distance: ")
    # print(min_thetas)

    # avg = cp.mean(min_thetas)
    # std = cp.std(min_thetas)
    # print("Average angular distance:", avg.get())
    # print("Std of angular dist: ", std.get())

    # Inverse-distance weighting (avoid div-by-zero)
    eps = 1e-6
    inv_tl = 1.0 / (theta_tl + eps)
    inv_tr = 1.0 / (theta_tr + eps)
    inv_bl = 1.0 / (theta_bl + eps)
    inv_br = 1.0 / (theta_br + eps)

    # Normalize weights
    weight_sum = inv_tl + inv_tr + inv_bl + inv_br
    w_tl = inv_tl / weight_sum
    w_tr = inv_tr / weight_sum
    w_bl = inv_bl / weight_sum
    w_br = inv_br / weight_sum

    # Interpolate
    neighbors = (
        top_left     * w_tl[:, None] +
        top_right    * w_tr[:, None] +
        bottom_left  * w_bl[:, None] +
        bottom_right * w_br[:, None]
    )

    final = cp.clip(neighbors, 0, 255)
    R, G, B = final[:, 0], final[:, 1], final[:, 2]

    return R, G, B

def sample_CubeMap_bilinear_chunked(
    row_f: cp.ndarray,
    col_f: cp.ndarray,
    vectors: cp.ndarray,
    input_image: cp.ndarray,
    face_size: int,
    is_CMP: bool,
    chunk_size: int = 10000,   # safe default for 100K points
):
    """
    Chunked version of CubeMap bilinear sampling to avoid long GPU kernels.

    Args:
        row_f, col_f: (N,) fractional pixel coords
        vectors: (N, 3) unit vectors
        input_image: (H, W, 3) CMP or EAC image
        face_size: cube face size
        is_CMP: True for CMP, False for EAC
        chunk_size: number of points per kernel launch

    Returns:
        R, G, B: (N,) arrays
    """

    N = row_f.size
    R_list, G_list, B_list = [], [], []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        R_chunk, G_chunk, B_chunk = sample_CubeMap_bilinear_helper(
            row_f=row_f[start:end],
            col_f=col_f[start:end],
            vectors=vectors[start:end],
            input_image=input_image,
            face_size=face_size,
            is_CMP=is_CMP,
        )

        R_list.append(R_chunk)
        G_list.append(G_chunk)
        B_list.append(B_chunk)

    R = cp.concatenate(R_list)
    G = cp.concatenate(G_list)
    B = cp.concatenate(B_list)
    return R, G, B

def sample_CMP(geographic_points: cp.ndarray, input_image: cp.ndarray, sample_type: Image.Resampling):
    """
    Samples from a CMP image given a set of input geographic points.

    Args:
        geographic_points: The geographic points to sample on.
        input_image: The input CMP image to sample from.
        sample_type: The type of sampling to conduct.
    
    Returns:
        R, G, B: (N,) interpolated channels.
    """

    H, W = input_image.shape[:2]

    face_size = W / 3

    if face_size != int(W / 3) and face_size != (H / 2):
        raise ValueError("The input CMP image was not formatted correctly!")

    # Convert to unit vectors.
    lat_deg, lon_deg = geographic_points
    vectors = geographic_to_vector_gpu(lat_deg=lat_deg, lon_deg=lon_deg)
    vectors = cp.stack(vectors, axis=1)

    # Grab the face index (based on face map) and (u, v) coordinates for each unit vector.
    face_indices, u, v = get_CubeMap_face_and_uv(vectors=vectors)

    # grab the row and col index in the CMP image for each point.
    row_f, col_f = map_face_uv_to_CMP_row_col(face_indices, u, v, face_size)

    if sample_type == Image.Resampling.NEAREST:
        # Round to nearest integer indices
        row_nn = cp.round(row_f).astype(cp.int32)
        col_nn = cp.round(col_f).astype(cp.int32)

        # Clamp indices to valid range [0, face_size-1]
        row_nn = cp.clip(row_nn, 0, H - 1)
        col_nn = cp.clip(col_nn, 0, W - 1)
        
        R = input_image[row_nn, col_nn, 0]
        G = input_image[row_nn, col_nn, 1]
        B = input_image[row_nn, col_nn, 2]
    elif sample_type == Image.Resampling.BILINEAR:
        R, G, B = sample_CubeMap_bilinear_helper(
            row_f=row_f,
            col_f=col_f,
            vectors=vectors,
            input_image=input_image,
            face_size=face_size,
            is_CMP=True,
        )
    else:
        raise ValueError(f"Sampling type ({sample_type}) is not supported!")
    

    return R, G, B

# endregion Sampling From CMP Helpers

# region Sampling From EAC Helpers

def map_face_uv_to_EAC_row_col(
    face_indices: cp.ndarray,
    u: cp.ndarray,
    v: cp.ndarray,
    face_size: int
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Maps (face_index, u, v) in [-1, 1] to global CMP (row, col) image coordinates.

    Args:
        face_indices: (N,) array of face indices ∈ [0,5]
        u, v: (N,) arrays of local face coordinates ∈ [-1, 1]
        face_size: int, size (in pixels) of one cube face

    Returns:
        (row_f, col_f): (N,) arrays of image-space coordinates
    """
    # CubeMap layout:
    #  [ (0,0) X=+1 , (0,1) Y=+1 , (0,2) X=-1 ]
    #  [ (1,0) Z=-1 , (1,1) Y=-1 , (1,2) Z=+1 ]
    face_row = cp.array([0, 0, 0, 1, 1, 1], dtype=cp.int32)
    face_col = cp.array([0, 1, 2, 0, 1, 2], dtype=cp.int32)

    row_offsets = face_row[face_indices] * face_size
    col_offsets = face_col[face_indices] * face_size

    # Rescale from [-1, 1] to pixel index range [0, face_size - 1]
    pi_quarter = cp.pi / 4
    col_local = ((cp.arctan(u) / pi_quarter) - 1.0) * -0.5 * (face_size) - 0.5
    row_local = ((cp.arctan(v) / pi_quarter) - 1.0) * -0.5 * (face_size) - 0.5

    col_indices = col_offsets + col_local
    row_indices = row_offsets + row_local

    return row_indices, col_indices

def sample_EAC(geographic_points: cp.ndarray, input_image: cp.ndarray, sample_type: Image.Resampling):
    """
    Samples from an EAC image given a set of input geographic points.

    Args:
        geographic_points: The geographic points to sample on.
        input_image: The input EAC image to sample from.
        sample_type: The type of sampling to conduct.
    
    Returns:
        R, G, B: (N,) interpolated channels.
    """

    H, W = input_image.shape[:2]

    face_size = W / 3

    if face_size != int(W / 3) and face_size != (H / 2):
        raise ValueError("The input CMP image was not formatted correctly!")

    # Convert to unit vectors.
    lat_deg, lon_deg = geographic_points
    vectors = geographic_to_vector_gpu(lat_deg=lat_deg, lon_deg=lon_deg)
    vectors = cp.stack(vectors, axis=1)

    # Grab the face index (based on face map) and (u, v) coordinates for each unit vector.
    face_indices, u, v = get_CubeMap_face_and_uv(vectors=vectors)

    # grab the row and col index in the CMP image for each point.
    row_f, col_f = map_face_uv_to_EAC_row_col(face_indices, u, v, face_size)

    if sample_type == Image.Resampling.NEAREST:
        # Round to nearest integer indices
        row_nn = cp.round(row_f).astype(cp.int32)
        col_nn = cp.round(col_f).astype(cp.int32)

        # Clamp indices to valid range [0, face_size-1]
        row_nn = cp.clip(row_nn, 0, H - 1)
        col_nn = cp.clip(col_nn, 0, W - 1)
        
        R = input_image[row_nn, col_nn, 0]
        G = input_image[row_nn, col_nn, 1]
        B = input_image[row_nn, col_nn, 2]

    elif sample_type == Image.Resampling.BILINEAR:
        R, G, B = sample_CubeMap_bilinear_helper(
            row_f=row_f,
            col_f=col_f,
            vectors=vectors,
            input_image=input_image,
            face_size=face_size,
            is_CMP=False,
        )
    else:
        raise ValueError(f"Sampling type ({sample_type}) is not supported!")
    
    return R, G, B

# endregion Sampling From EAC Helpers

# region Sampling From EAP Helpers

def geographic_to_EAP_index_float(lat_deg: cp.ndarray, lon_deg: cp.ndarray, height: int, width: int):
    """
    Convert lat/lon to floating point EAP image coordinates.

    Args:
        lat: latitude in radians, shape (N,)
        lon: longitude in radians, shape (N,)
        H: image height
        W: image width

    Returns:
        row_f, col_f: float row and column indices into the EAP image
    """

    lat_rad = cp.radians(lat_deg)

    # Latitude is mapped using the sine function from [-pi/2, pi/2] to [0, height]
    row_f_corners = (1.0 - cp.sin(lat_rad)) / 2.0 * height

    # Longitude is mapped linearly from [-pi, pi] to [0, width]
    col_f_corners = (lon_deg + 180.0) / 360.0 * width

    row_f = row_f_corners - 0.5
    col_f = col_f_corners - 0.5
    
    return row_f, col_f

def sample_EAP_bilinear_helper(input_image: cp.ndarray, row_f: cp.ndarray, col_f: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Performs bilinear sampling from a CuPy image of an EAP image from row and column floats.

    Args:
        input_image: (H, W, 3) CuPy image array of EAP image.
        row_f: (N,) float row indices.
        col_f: (N,) float col indices.

    Returns:
        R, G, B: (N,) interpolated channels.
    """
    H, W = input_image.shape[:2]

    time_one = time.perf_counter()

    # Get integer floor and clipped ceil indices
    row0 = cp.clip(cp.floor(row_f).astype(cp.int32), 0, H - 1)
    row1 = cp.clip(row0 + 1, 0, H - 1)
    col0 = cp.clip(cp.floor(col_f).astype(cp.int32), 0, W - 1)
    col1 = cp.clip(col0 + 1, 0, W - 1)

    # Stack corners for vectorized processing: shape (N,4)
    rows = cp.stack([row0, row0, row1, row1], axis=1)
    cols = cp.stack([col0, col1, col0, col1], axis=1)

    time_two = time.perf_counter()

    # Sample pixel colors for all corners at once: (N,4,3)
    pixels = input_image[rows, cols]  # advanced indexing on GPU

    time_three = time.perf_counter()

    # Convert the samples to phi and theta
    phi_samples, theta_samples = EAP_index_to_polar_azimuthal(row_f, col_f, H, W)

    # Convert the corners to phi and theta
    phi_corners, theta_corners = EAP_index_to_polar_azimuthal(rows, cols, H, W)

    time_four = time.perf_counter()

    # Broadcast and compute cosine theta:
    cos_theta = (
        cp.sin(phi_samples[:, None]) * cp.sin(phi_corners) +
        cp.cos(phi_samples[:, None]) * cp.cos(phi_corners) *
        cp.cos(theta_samples[:, None] - theta_corners)
    )

    # Clamp and get angular distances
    thetas = cp.arccos(cp.clip(cos_theta, -1, 1))

    # mean = cp.mean(thetas)
    # std = cp.std(thetas)

    # print(f"Mean of theta: {mean}")
    # print(f"Standard deviation of theta: {std}")

    time_five = time.perf_counter()

    eps = 1e-6
    inv_thetas = 1.0 / (thetas + eps)

    # Normalize weights (N,4)
    weights = inv_thetas / cp.sum(inv_thetas, axis=1, keepdims=True)

    time_six = time.perf_counter()

    # Interpolate RGB: multiply weights with pixel colors and sum over corners
    neighbors = cp.sum(pixels * weights[:, :, None], axis=1)  # (N,3)

    time_seven = time.perf_counter()

    # Clamp RGB output
    final = cp.clip(neighbors, 0, 255)

    time_eight = time.perf_counter()

    # print(f"Time between time_two and time_one: {time_two - time_one}")
    # print(f"Time between time_three and time_two: {time_three - time_two}")
    # print(f"Time between time_four and time_three: {time_four - time_three}")
    # print(f"Time between time_five and time_four: {time_five - time_four}")
    # print(f"Time between time_six and time_five: {time_six - time_five}")
    # print(f"Time between time_seven and time_six: {time_seven - time_six}")
    # print(f"Time between time_eight and time_seven: {time_eight - time_seven}")

    return final[:, 0], final[:, 1], final[:, 2]

def EAP_index_to_polar_azimuthal(row: cp.ndarray, col: cp.ndarray, height: int, width: int) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Converts EAP image row/column indices to spherical polar (phi) and azimuthal (theta) angles.
    
    Args:
        row: Row index(es) from the EAP image.
        col: Column index(es) from the EAP image.
        height: Height of the EAP image.
        width: Width of the EAP image.
        
    Returns:
        Tuple[cp.ndarray, cp.ndarray]: (phi, theta) in radians.
                                      phi: polar angle (0 to pi, from top to bottom, 0 at North Pole)
                                      theta: azimuthal angle (0 to 2*pi, around the equator)
    """
    # Adjust row and col to represent the center of the pixel
    row_centers = row + 0.5
    col_centers = col + 0.5

    # Maps col from [0, W] to [0, 2*pi]
    theta = 2 * cp.pi * col_centers / width

    # 2. Calculate polar angle (phi) - EAP-specific non-linear mapping
    # First, normalize the row index to a range equivalent to sin(latitude) or a 'z' component.
    # The normalized_y goes from approx 1.0 (top row) to approx -1.0 (bottom row).
    normalized_y = (2.0 * row_centers) / height - 1.0
    normalized_y_clipped = cp.clip(normalized_y, -1.0, 1.0)
    
    phi = cp.pi / 2 - cp.arcsin(normalized_y_clipped)
    
    return phi, theta

def sample_EAP(geographic_points: cp.ndarray, input_image: cp.ndarray, sample_type: Image.Resampling):
    """
    Samples from an EAC image given a set of input geographic points.

    Args:
        geographic_points: The geographic points to sample on.
        input_image: The input EAC image to sample from.
        sample_type: The type of sampling to conduct.
    
    Returns:
        R, G, B: (N,) interpolated channels.
    """
    
    H, W = input_image.shape[:2]

    lat_deg, lon_deg = geographic_points

    # grab the row and col index in the EAP image for each point.
    row_f, col_f = geographic_to_EAP_index_float(lat_deg, lon_deg, H, W)

    if sample_type == Image.Resampling.NEAREST:
        # Round to nearest integer indices
        row_nn = cp.round(row_f).astype(cp.int32)
        col_nn = cp.round(col_f).astype(cp.int32)

        # Clamp indices to valid range [0, face_size-1]
        row_nn = cp.clip(row_nn, 0, H - 1)
        col_nn = cp.clip(col_nn, 0, W - 1)
        
        R = input_image[row_nn, col_nn, 0]
        G = input_image[row_nn, col_nn, 1]
        B = input_image[row_nn, col_nn, 2]

    elif sample_type == Image.Resampling.BILINEAR:
        R, G, B = sample_EAP_bilinear_helper(
            input_image=input_image,
            row_f=row_f,
            col_f=col_f,
        )
    else:
        raise ValueError(f"Sampling type ({sample_type}) is not supported!")
    
    return R, G, B

# endregion Sampling From EAP Helpers

# region Sampling From SFLP Helpers

def map_vectors_to_SFLP_indices(vectors: cp.ndarray, num_points: int, sample_type: Image.Resampling) -> cp.ndarray:
    """
    GPU-parallelized function that maps each input vector to its four nearest SFLP indices.

    Args:
        vectors: (N, 3) CuPy array of unit vectors.
        num_points: int, number of SFLP points.
        sample_type: The type of sampling to do (Only nearest and bilinear supported).

    Returns:
        indices: (N, X) CuPy array of the X nearest SFLP indices for each vector (depending on whether sampling nearest or bilinear).
    """
    golden_ratio = golden_ratio_cp()

    def madfrac(a, b):
        return a * b - cp.floor(a * b)
    
    def normalize_theta(unnormalized_theta):
        return cp.mod(unnormalized_theta + cp.pi, 2 * cp.pi) - cp.pi

    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    theta = cp.arctan2(y, x)

    del x, y

    k = cp.maximum(2, cp.floor(
        cp.log(num_points * cp.pi * cp.sqrt(5) * (1 - z * z)) /
        cp.log(golden_ratio * golden_ratio)
    )).astype(cp.int32)

    Fk = golden_ratio**k / cp.sqrt(5)
    F0 = cp.round(Fk).astype(cp.int32)
    F1 = cp.round(Fk * golden_ratio).astype(cp.int32)

    # Broadcast B and invB for each row
    B_TOP_ROW = cp.stack([
        2 * cp.pi * madfrac(F0 + 1, golden_ratio - 1) - 2 * cp.pi * (golden_ratio - 1),
        2 * cp.pi * madfrac(F1 + 1, golden_ratio - 1) - 2 * cp.pi * (golden_ratio - 1)
    ], axis=-1)

    B_BOTTOM_ROW = cp.stack([
        2 * F0 / num_points,
        2 * F1 / num_points
    ], axis=-1)

    B = cp.stack([B_TOP_ROW, B_BOTTOM_ROW], axis=1)  # (N, 2, 2)
    invB = cp.linalg.inv(B)

    del k, Fk, F0, F1, B_TOP_ROW, B_BOTTOM_ROW

    c_u_init = normalize_theta(2 * cp.pi * -1 * cp.floor((num_points) / 2) * (golden_ratio - 1))
    c_v_init = (1 / num_points - 1)
    c_u = normalize_theta(theta - c_u_init)
    c_v = z - c_v_init
    target = cp.stack([c_u, c_v], axis=-1)  # (N, 2)
    c = cp.floor(cp.einsum("nij,nj->ni", invB, target))  # (N, 2)

    del c_u_init, c_v_init, c_u, c_v, invB

    offsets = cp.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # shape (4, 2)
    offsets = offsets[None, :, :]  # shape (1, 4, 2)
    c_broadcasted = c[:, None, :]  # shape (N, 1, 2)
    c_with_offsets = offsets + c_broadcasted  # (N, 4, 2)

    del c, c_broadcasted, offsets, theta

    # B @ offset.T + (1 - 1/N) => (N, 4)
    z_approx = cp.einsum("nij,nkj->nki", B, c_with_offsets)[:, :, 1] + (1 / num_points - 1)
    z_clipped = cp.clip(z_approx, -1, 1)
    z_reflected = z_clipped * 2 - z_approx

    nearest_four_indices = cp.floor(num_points * 0.5 + z_reflected * num_points * 0.5).astype(cp.int32)
    nearest_four_indices = cp.clip(nearest_four_indices, 0, num_points - 1)  # (N, 4)

    del z_approx, z_clipped, z_reflected

    nearest_four_theta = normalize_theta(2 * cp.pi * (nearest_four_indices - cp.floor((num_points) / 2)) * (golden_ratio - 1))
    nearest_four_cos_phi = (2 * nearest_four_indices + 1) / num_points - 1

    nearest_four_sin_phi = cp.sqrt(1 - nearest_four_cos_phi * nearest_four_cos_phi)

    q = cp.stack(
        [
            cp.cos(nearest_four_theta) * nearest_four_sin_phi,
            cp.sin(nearest_four_theta) * nearest_four_sin_phi,
            nearest_four_cos_phi
        ],
        axis=-1
    )

    del nearest_four_theta, nearest_four_cos_phi, nearest_four_sin_phi

    diff = q - vectors[:, None, :]  # Shape: (N, 4, 3)
    squared_distances = cp.sum(diff ** 2, axis=-1)

    del diff, q

    min_indices = cp.argmin(squared_distances, axis=1) # Which of the four indexes
    final_indices = nearest_four_indices[cp.arange(len(nearest_four_indices)), min_indices]

    del squared_distances

    if sample_type == Image.Resampling.NEAREST:
        return final_indices[:, None]
    elif sample_type == Image.Resampling.BILINEAR:
        return nearest_four_indices
    elif sample_type == "Smoothing Filter":
        final_c = c_with_offsets[cp.arange(len(c_with_offsets)), min_indices]

        del c_with_offsets, final_indices, min_indices, nearest_four_indices

        smooth_filter_offsets = cp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]])
        smooth_filter_offsets = smooth_filter_offsets[None, :, :]  # shape (1, 4, 2)
        smooth_c_broadcasted = final_c[:, None, :]  # shape (N, 1, 2)
        smooth_c_with_offsets = smooth_filter_offsets + smooth_c_broadcasted

        del smooth_c_broadcasted, smooth_filter_offsets

        smooth_z_approx = cp.einsum("nij,nkj->nki", B, smooth_c_with_offsets)[:, :, 1] + (1 / num_points - 1)

        del B

        smooth_z_clipped = cp.clip(smooth_z_approx, -1, 1)
        smooth_z_reflected = smooth_z_clipped * 2 - smooth_z_approx

        del smooth_c_with_offsets, smooth_z_approx, smooth_z_clipped

        smooth_indices = cp.floor(num_points * 0.5 + smooth_z_reflected * num_points * 0.5).astype(cp.int32)
        smooth_indices = cp.clip(smooth_indices, 0, num_points - 1)

        return smooth_indices
    else:
        raise ValueError(f"Sampling type: {sample_type} is not supported for vectors to SFLP indices!")

def sample_SFLP_helper(
    vectors: cp.ndarray,
    sampled_indices: cp.ndarray,
    input_image: cp.ndarray,
    num_pixels: int,
    arrangement_type: SFLPArrangementType,
    optimal_index_hops: list,
    first_element_SFLP_index: int,
    last_element_SFLP_index: int,
    non_padded_col_length: int,
    non_padded_row_length: int,

) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Performs spherical-aware bilinear sampling on a SFLP image using angular distances.

    Args:
        vectors: (N, 3) Unit vectors on the sphere corresponding to sampling points.
        nearest_four_indices: (N, 4) The nearest SFLP indices around each vector.
        input_image: (H, W, 3) The input SFLP image.
        num_pixels (int): The number of SFLP pixels in the input image.
        optimal_index_hops: The two optimal k-th index hops.
        first_element_SFLP_index: SFLP index of first element of first column.
        last_element_SFLP_index: SFLP index of last element of first column.
        non_padded_col_length: The length of the longest column, excluding padding.
        non_padded_row_length: The length of the longest row, excluding padding.
        
    Returns:
        result: (N, 3) interpolated RGB values
    """
    input_image_dtype = input_image.dtype

    num_sampled_per_pixel = sampled_indices.shape[1] # Grab the number of indices to compute the weight matrix for.

    L = sampled_indices.shape[0]
    D = input_image.shape[2]
    all_pixels = cp.empty((num_sampled_per_pixel, L, D), dtype=input_image_dtype)
    angular_distances = cp.empty((num_sampled_per_pixel, L), dtype=float)

    start_time = time.perf_counter()

    for i in range(num_sampled_per_pixel):
        
        # Grab the pixel values for each index.
        next_indices = sampled_indices[:, i]

        if arrangement_type == SFLPArrangementType.STANDARD:
            first_nearest_row_col = get_row_col_index_gpu(
                SFLP_indices=next_indices,
                num_points=num_pixels,
                optimal_index_hops=optimal_index_hops,
                last_element_SFLP_index=last_element_SFLP_index,
                row_length=non_padded_row_length,
                col_length=non_padded_col_length
            )
        elif arrangement_type == SFLPArrangementType.SEAM_REMOVED:
            first_nearest_row_col = get_row_col_index_seam_removed_gpu(
                SFLP_indices=next_indices,
                num_points=num_pixels,
                optimal_index_hops=optimal_index_hops,
                num_tiles_x=1,
                num_tiles_y=1,
                first_element_SFLP_index=first_element_SFLP_index,
                last_element_SFLP_index=last_element_SFLP_index,
                row_length=non_padded_row_length,
                col_length=non_padded_col_length
            )
        elif arrangement_type == SFLPArrangementType.ZIGZAG:
            first_nearest_row_col = get_row_col_index_zigzag_gpu(
                SFLP_indices=next_indices,
                num_points=num_pixels,
                optimal_index_hops=optimal_index_hops,
                num_tiles_x=1,
                num_tiles_y=1,
                first_element_SFLP_index=first_element_SFLP_index,
                last_element_SFLP_index=last_element_SFLP_index,
                row_length=non_padded_row_length,
                col_length=non_padded_col_length
            )
        else:
            raise ValueError(f"Arrangement type: {arrangement_type} is not supported for sampling on!")


        next_pixels = input_image[first_nearest_row_col[:, 0], first_nearest_row_col[:, 1]]
        all_pixels[i, :] = next_pixels

        # Convert each nearest index to vector.
        next_lat, next_lon = get_SFL_geographic_points(next_indices, num_pixels)
        next_vectors = geographic_to_vector_gpu(next_lat, next_lon)
        next_vectors = cp.stack(next_vectors, axis=1)

        # Find the spherical distance between the vector and each nearest vector to get weights.
        # Compute angular distances using 1 - cos(θ)
        spherical_distance = cp.einsum('ij,ij->i', vectors, next_vectors)
        spherical_distance = cp.clip(spherical_distance, -1.0, 1.0)

        # Angular distances in radians
        angular_distance = cp.arccos(spherical_distance)
        angular_distances[i, :] = angular_distance

    del input_image, sampled_indices, vectors
    
    # print("Smallest angular distance: ")
    # print(angular_distances[0])
    
    # avg = cp.mean(angular_distances, axis=1)
    # std = cp.std(angular_distances, axis=1)

    # print("Average angular distance:", avg.get())
    # print("Standard deviation of angular distance:", std.get())

    end_time = time.perf_counter()

    # print(f"Time to get pixels and angular distances: {end_time - start_time:.6f} seconds")

    # Inverse-distance weighting (avoid div-by-zero)
    eps = 1e-6
    inv_weights = 1.0 / (angular_distances + eps)

    # Normalize weights
    weights_sum = cp.sum(inv_weights, axis=0)
    normalized_weights = (inv_weights / weights_sum).astype(cp.float32)

    del inv_weights, weights_sum

    # Interpolate
    neighbors = cp.zeros((L, D), dtype=float)  # (N, D)

    # Accumulate weighted contributions from each neighbor
    for i in range(num_sampled_per_pixel):
        neighbors += normalized_weights[i, :, None] * all_pixels[i]

    final = cp.clip(neighbors, 0, 255)
    R, G, B = final[:, 0], final[:, 1], final[:, 2]

    return R, G, B

def sample_SFLP(
    geographic_points: tuple[cp.ndarray, cp.ndarray],
    input_image: cp.ndarray,
    num_pixels: int,
    arrangement_type: SFLPArrangementType,
    optimal_index_hops: list,
    first_element_SFLP_index: int,
    last_element_SFLP_index: int,
    non_padded_col_length: int,
    non_padded_row_length: int,
    sample_type: Image.Resampling,
):
    """
    Samples from a SFLP image given a set of input geographic points.

    Args:
        geographic_points: The geographic points to sample on.
        input_image: The input ERP SFLP to sample from.
        num_pixels: The number of SFLP pixels in the input image.
        arrangement_type: The type of SFLP arrangement (STANDARD, SEAM_REMOVED, or ZIGZAG).
        optimal_index_hops: The two optimal k-th index hops.
        first_element_SFLP_index, last_element_SFLP_index: SFLP index of first element and last element of first column.
        non_padded_col_length: The length of the longest column, excluding padding.
        non_padded_row_length: The length of the longest row, excluding padding.
        sample_type: The type of sampling to conduct (nearest-neighbor or bilinear).

    Returns:
        R, G, B: (N,) interpolated channels.
    """

    lat_deg, lon_deg = geographic_points
    vectors = geographic_to_vector_gpu(lat_deg=lat_deg, lon_deg=lon_deg)
    vectors = cp.stack(vectors, axis=1)

    del lat_deg, lon_deg, geographic_points

    # Grab the four nearest SFLP indices for each vector.
    start_time = time.perf_counter()
    sampled_indices = map_vectors_to_SFLP_indices(vectors=vectors, num_points=num_pixels, sample_type=sample_type)
    end_time = time.perf_counter()

    # print(f"Time to grab SFLP indices: {end_time - start_time:.6f} seconds")

    # Compute the samples given the nearest index/indices (nearest vs. bilinear) for each vector.
    R, G, B = sample_SFLP_helper(
        vectors=vectors,
        sampled_indices=sampled_indices,
        input_image=input_image,
        num_pixels=num_pixels,
        arrangement_type=arrangement_type,
        optimal_index_hops=optimal_index_hops,
        first_element_SFLP_index=first_element_SFLP_index,
        last_element_SFLP_index=last_element_SFLP_index,
        non_padded_col_length=non_padded_col_length,
        non_padded_row_length=non_padded_row_length,
    )
    
    return R, G, B

# endregion Sampling From SFLP Helpers

#
# endregion Sampling Functions
#

#
# region File Type Conversion Functions
#

def convert_to_H264(input_path: Path, output_path: Path, ffmpeg_path: str, bitrate: int=50000):
    """
    Converts an input video file to MP4 format using FFmpeg with H.264 codec.

    Args:
        input_path (Path): Path to the input video file.
        output_path (Path): Path to save the output MP4 file.
        ffmpeg_path (str): Path to the FFmpeg executable.
    """

    ffmpeg_cmd = [
        ffmpeg_path, '-y',
        '-i', str(input_path),

        # --- Video Encoding Settings ---
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-b:v', f'{bitrate}k',           # Target bitrate
        '-maxrate', f'{bitrate}k',       # Constrain bitrate peaks
        '-bufsize', f'{2 * bitrate}k',      # Bitrate buffer
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',

        # --- Audio Encoding Settings ---
        # '-c:a', 'aac',
        # '-b:a', '192k',

        '-threads', '8',
        str(output_path)
    ]

    subprocess.run(ffmpeg_cmd, check=True, shell=True)

def convert_to_H265(input_path: Path, output_path: Path, ffmpeg_path: str, bitrate: int=50000):
    """
    Converts an input video file to MP4 format using FFmpeg with H.265 codec.

    Args:
        input_path (Path): Path to the input video file.
        output_path (Path): Path to save the output MP4 file.
        ffmpeg_path (str): Path to the FFmpeg executable.
    """

    ffmpeg_cmd = [
        ffmpeg_path, '-y',
        '-i', str(input_path),

        # --- Video Encoding Settings ---
        '-c:v', 'libx265',
        '-preset', 'slow',
        '-b:v', f'{bitrate}k',           # Target bitrate
        '-maxrate', f'{bitrate}k',       # Constrain bitrate peaks
        '-bufsize', f'{10 * bitrate}k',      # Bitrate buffer
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',

        '-threads', '8',
        str(output_path)
    ]

    subprocess.run(ffmpeg_cmd, check=True, shell=True)

def convert_to_av1(input_path: Path, output_path: Path, ffmpeg_path: str, bitrate: int = 50000):
    """
    Converts an input AVI file to AV1 format using FFmpeg.

    Args:
        input_path (str): Path to the input AVI file.
        output_path (str): Path to save the output AV1 file.
    """


    ffmpeg_cmd = [
        ffmpeg_path, '-y',
        '-i', str(input_path),

        # --- Video Encoding Settings ---
        '-c:v', 'libsvtav1',
        '-preset', '4',
        '-rc', '2',
        # Pass SVT-AV1 encoder params here
        '-b:v', f'{bitrate}k',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',

        # --- Audio Encoding Settings ---
        # '-c:a', 'aac',
        # '-b:a', '192k',

        '-threads', '2',
        str(output_path)
    ]

    
    subprocess.run(ffmpeg_cmd, check=True, shell=True)

#
# endregion File Type Conversion Functions
#

#
# region VR Projection Type Conversions
#

# region Image Conversion

def get_SFLP_geographic_points_from_layout(
    num_pixels: int,
    layout_csv_path: Path,
):
    # === Load Layout ===
    arranged_pixel_indices_df = pd.read_csv(layout_csv_path, header=None)
    arranged_pixels_indices = arranged_pixel_indices_df.to_numpy()
    H, W = arranged_pixels_indices.shape[:2]

    dimensions = W, H

    indices_cp = cp.asarray(arranged_pixels_indices.flatten())
    SFLP_geographic_points = get_SFL_geographic_points(indices=indices_cp, num_points=num_pixels)

    return SFLP_geographic_points, dimensions

def ERP_to_SFLP_image_conversion_preloaded_layout(
  SFLP_geographic_points: Tuple[cp.ndarray, cp.ndarray],
  dimensions: Tuple[int, int],
  input_image_path: Path,
  output_path: Path,
  sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an ERP image to an SFLP image.

    Args:
        num_pixels (int): Number of pixels to sample on.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing ERP image.
        output_path (Path): Path to save SFLP image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """
    W, H = dimensions

    # === Load ERP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    ERP_image = np.array(img)
    ERP_image_cp = cp.asarray(ERP_image)

    # === Convert ERP to SFLP ===
    input_metadata = ProjectionMetadata(EncodingProjection.ERP)
    R, G, B = geographic_points_to_image_info(
        SFLP_geographic_points,
        ERP_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    arranged_image_gpu = cp.stack([R, G, B], axis=1).reshape(H, W, 3)

    # Transfer the whole image at once
    arranged_image_cpu = cp.asnumpy(arranged_image_gpu)

    # Then clip and convert
    arranged_uint8 = np.clip(arranged_image_cpu, 0, 255).astype(np.uint8)

    # === Save SFLP Image ===
    img_out = Image.fromarray(arranged_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved SFLP image to {output_path}")

    del ERP_image_cp, R, G, B
    del arranged_image_gpu

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def ERP_to_SFLP_image_conversion(
  num_pixels: int,
  layout_csv_path: Path,
  input_image_path: Path,
  output_path: Path,
  sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an ERP image to an SFLP image.

    Args:
        num_pixels (int): Number of pixels to sample on.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing ERP image.
        output_path (Path): Path to save SFLP image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """

    # === Load Layout ===
    arranged_pixel_indices_df = pd.read_csv(layout_csv_path, header=None)
    arranged_pixels_indices = arranged_pixel_indices_df.to_numpy()
    H, W = arranged_pixels_indices.shape[:2]

    indices_cp = cp.asarray(arranged_pixels_indices.flatten())
    SFLP_geographic_points = get_SFL_geographic_points(indices=indices_cp, num_points=num_pixels)

    # === Load ERP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    ERP_image = np.array(img)
    ERP_image_cp = cp.asarray(ERP_image)

    # === Convert ERP to SFLP ===
    input_metadata = ProjectionMetadata(EncodingProjection.ERP)
    R, G, B = geographic_points_to_image_info(
        SFLP_geographic_points,
        ERP_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    arranged_image_gpu = cp.stack([R, G, B], axis=1).reshape(H, W, 3)

    # Transfer the whole image at once
    arranged_image_cpu = cp.asnumpy(arranged_image_gpu)

    # Then clip and convert
    arranged_uint8 = np.clip(arranged_image_cpu, 0, 255).astype(np.uint8)

    # === Save SFLP Image ===
    img_out = Image.fromarray(arranged_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved SFLP image to {output_path}")

    del indices_cp, SFLP_geographic_points
    del ERP_image_cp, R, G, B
    del arranged_image_gpu

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def get_Cube_geographic_points_from_layout(
    face_size: int,
    layout_csv_path: Path,
    is_CMP: bool,
):
    # === Load Layout ===
    arranged_pixels_tokens_df = pd.read_csv(layout_csv_path, header=None)
    arranged_pixels_tokens = arranged_pixels_tokens_df.to_numpy()
    H, W = arranged_pixels_tokens.shape[:2]

    dimensions = W, H

    tokens = arranged_pixels_tokens.flatten()
    Cube_geographic_points = get_CubeMap_geographic_points(tokens, face_size, is_CMP=is_CMP)

    return Cube_geographic_points, dimensions

def ERP_to_CMP_image_conversion_preloaded_layout(
  CMP_geographic_points: Tuple[cp.ndarray, cp.ndarray],
  dimensions: Tuple[int, int],
  input_image_path: Path,
  output_path: Path,
  sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an ERP image to a CMP image.

    Args:
        face_size (int): The number of pixels in a row or column on a face of the cube.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing ERP image.
        output_path (Path): Path to save CMP image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """
    W, H = dimensions

    # === Load ERP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    ERP_image = np.array(img)
    ERP_image_cp = cp.asarray(ERP_image)

    # === Convert ERP to CMP ===
    input_metadata = ProjectionMetadata(EncodingProjection.ERP)
    R, G, B = geographic_points_to_image_info(
        geographic_points=CMP_geographic_points,
        input_image=ERP_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    R_np = cp.asnumpy(R)
    G_np = cp.asnumpy(G)
    B_np = cp.asnumpy(B)

    arranged_image_info = np.stack(
        [R_np.reshape(H, W), G_np.reshape(H, W), B_np.reshape(H, W)], axis=2
    )

    arranged_image_info_uint8 = np.clip(arranged_image_info, 0, 255).astype(np.uint8)

    # === Save CMP Image ===
    img_out = Image.fromarray(arranged_image_info_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved CMP image to {output_path}")

    del ERP_image_cp, R, G, B
    del arranged_image_info

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def ERP_to_CMP_image_conversion(
  face_size: int,
  layout_csv_path: Path,
  input_image_path: Path,
  output_path: Path,
  sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an ERP image to a CMP image.

    Args:
        face_size (int): The number of pixels in a row or column on a face of the cube.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing ERP image.
        output_path (Path): Path to save CMP image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """

    # === Load Layout ===
    arranged_pixels_tokens_df = pd.read_csv(layout_csv_path, header=None)
    arranged_pixels_tokens = arranged_pixels_tokens_df.to_numpy()
    H, W = arranged_pixels_tokens.shape[:2]

    tokens = arranged_pixels_tokens.flatten()
    CMP_geographic_points = get_CubeMap_geographic_points(tokens, face_size, is_CMP=True)

    # === Load ERP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    ERP_image = np.array(img)
    ERP_image_cp = cp.asarray(ERP_image)

    # === Convert ERP to CMP ===
    input_metadata = ProjectionMetadata(EncodingProjection.ERP)
    R, G, B = geographic_points_to_image_info(
        geographic_points=CMP_geographic_points,
        input_image=ERP_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    R_np = cp.asnumpy(R)
    G_np = cp.asnumpy(G)
    B_np = cp.asnumpy(B)

    arranged_image_info = np.stack(
        [R_np.reshape(H, W), G_np.reshape(H, W), B_np.reshape(H, W)], axis=2
    )

    arranged_image_info_uint8 = np.clip(arranged_image_info, 0, 255).astype(np.uint8)

    # === Save CMP Image ===
    img_out = Image.fromarray(arranged_image_info_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved CMP image to {output_path}")

    del arranged_pixels_tokens, CMP_geographic_points
    del ERP_image_cp, R, G, B
    del arranged_image_info

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def ERP_to_EAC_image_conversion_preloaded_layout(
    EAC_geographic_points: Tuple[cp.ndarray, cp.ndarray],
    dimensions: Tuple[int, int],
    input_image_path: Path,
    output_path: Path,
    sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an ERP image to an EAC image.

    Args:
        face_size (int): The number of pixels in a row or column on a face of the cube.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing ERP image.
        output_path (Path): Path to save EAC image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """
    W, H = dimensions

    # === Load ERP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    ERP_image = np.array(img)
    ERP_image_cp = cp.asarray(ERP_image)

    # === Convert ERP to EAC ===
    input_metadata = ProjectionMetadata(EncodingProjection.ERP)
    R, G, B = geographic_points_to_image_info(
        geographic_points=EAC_geographic_points,
        input_image=ERP_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    R_np = cp.asnumpy(R)
    G_np = cp.asnumpy(G)
    B_np = cp.asnumpy(B)

    arranged_image_info = np.stack(
        [R_np.reshape(H, W), G_np.reshape(H, W), B_np.reshape(H, W)], axis=2
    )

    arranged_image_info_uint8 = np.clip(arranged_image_info, 0, 255).astype(np.uint8)

    # === Save EAC Image ===
    img_out = Image.fromarray(arranged_image_info_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved EAC image to {output_path}")

    del ERP_image_cp, R, G, B
    del arranged_image_info

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def ERP_to_EAC_image_conversion(
  face_size: int,
  layout_csv_path: Path,
  input_image_path: Path,
  output_path: Path,
  sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an ERP image to an EAC image.

    Args:
        face_size (int): The number of pixels in a row or column on a face of the cube.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing ERP image.
        output_path (Path): Path to save EAC image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """

    # === Load Layout ===
    arranged_pixels_tokens_df = pd.read_csv(layout_csv_path, header=None)
    arranged_pixels_tokens = arranged_pixels_tokens_df.to_numpy()
    H, W = arranged_pixels_tokens.shape[:2]

    tokens = arranged_pixels_tokens.flatten()

    EAC_geographic_points = get_CubeMap_geographic_points(tokens, face_size, is_CMP=False)

    # === Load ERP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    ERP_image = np.array(img)
    ERP_image_cp = cp.asarray(ERP_image)

    # === Convert ERP to EAC ===
    input_metadata = ProjectionMetadata(EncodingProjection.ERP)
    R, G, B = geographic_points_to_image_info(
        geographic_points=EAC_geographic_points,
        input_image=ERP_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    R_np = cp.asnumpy(R)
    G_np = cp.asnumpy(G)
    B_np = cp.asnumpy(B)

    arranged_image_info = np.stack(
        [R_np.reshape(H, W), G_np.reshape(H, W), B_np.reshape(H, W)], axis=2
    )

    arranged_image_info_uint8 = np.clip(arranged_image_info, 0, 255).astype(np.uint8)

    # === Save EAC Image ===
    img_out = Image.fromarray(arranged_image_info_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved EAC image to {output_path}")

    del arranged_pixels_tokens, EAC_geographic_points
    del ERP_image_cp, R, G, B
    del arranged_image_info

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def ERP_to_EAP_image_conversion(
  output_size: Tuple[int, int],
  input_image_path: Path,
  output_path: Path,
  sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an ERP image to an EAP image.

    Args:
        output_size (int): The width and height of the image.
        input_image_path (Path): Path to image file containing ERP image.
        output_path (Path): Path to save EAP image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """

    W, H = output_size
    EAP_geographic_points = get_EAP_geographic_points(frame_dimensions=output_size)

    # === Load ERP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    ERP_image = np.array(img)
    ERP_image_cp = cp.asarray(ERP_image)

    # === Convert ERP to EAC ===
    input_metadata = ProjectionMetadata(EncodingProjection.ERP)
    R, G, B = geographic_points_to_image_info(
        geographic_points=EAP_geographic_points,
        input_image=ERP_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    R_np = cp.asnumpy(R)
    G_np = cp.asnumpy(G)
    B_np = cp.asnumpy(B)

    arranged_image_info = np.stack(
        [R_np.reshape(H, W), G_np.reshape(H, W), B_np.reshape(H, W)], axis=2
    )

    arranged_image_info_uint8 = np.clip(arranged_image_info, 0, 255).astype(np.uint8)

    # === Save EAP Image ===
    img_out = Image.fromarray(arranged_image_info_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved EAP image to {output_path}")

    del EAP_geographic_points
    del ERP_image_cp, R, G, B
    del arranged_image_info

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def CMP_to_SFLP_image_conversion(
    num_pixels: int,
    layout_csv_path: Path,
    input_image_path: Path,
    output_path: Path,
    sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts a CMP image to an SFLP image.

    Args:
        num_pixels (int): Number of pixels to sample on.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing CMP image.
        output_path (Path): Path to save SFLP image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """

    # === Load Layout ===
    arranged_pixel_indices_df = pd.read_csv(layout_csv_path, header=None)
    arranged_pixels_indices = arranged_pixel_indices_df.to_numpy()
    H, W = arranged_pixels_indices.shape[:2]

    indices_cp = cp.asarray(arranged_pixels_indices.flatten())
    SFLP_geographic_points = get_SFL_geographic_points(indices=indices_cp, num_points=num_pixels)

    # === Load CMP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    cmp_image = np.array(img)
    cmp_image_cp = cp.asarray(cmp_image)

    # === Convert CMP to SFLP ===
    input_metadata = ProjectionMetadata(EncodingProjection.CMP)
    R, G, B = geographic_points_to_image_info(
        SFLP_geographic_points,
        cmp_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    R_np = cp.asnumpy(R)
    G_np = cp.asnumpy(G)
    B_np = cp.asnumpy(B)

    arranged_image_info = np.stack(
        [R_np.reshape(H, W), G_np.reshape(H, W), B_np.reshape(H, W)], axis=2
    )

    arranged_image_info_uint8 = np.clip(arranged_image_info, 0, 255).astype(np.uint8)

    # === Save SFLP Image ===
    img_out = Image.fromarray(arranged_image_info_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved SFLP image to {output_path}")

def EAC_to_SFLP_image_conversion(
    num_pixels: int,
    layout_csv_path: Path,
    input_image_path: Path,
    output_path: Path,
    sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an EAC image to an SFLP image.

    Args:
        num_pixels (int): Number of pixels to sample on.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing EAC image.
        output_path (Path): Path to save SFLP image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """

    # === Load Layout ===
    arranged_pixel_indices_df = pd.read_csv(layout_csv_path, header=None)
    arranged_pixels_indices = arranged_pixel_indices_df.to_numpy()
    H, W = arranged_pixels_indices.shape[:2]

    indices_cp = cp.asarray(arranged_pixels_indices.flatten())
    SFLP_geographic_points = get_SFL_geographic_points(indices=indices_cp, num_points=num_pixels)

    # === Load EAC Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    eac_image = np.array(img)
    eac_image_cp = cp.asarray(eac_image)

    # === Convert EAC to SFLP ===
    input_metadata = ProjectionMetadata(EncodingProjection.EAC)
    R, G, B = geographic_points_to_image_info(
        SFLP_geographic_points,
        eac_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    R_np = cp.asnumpy(R)
    G_np = cp.asnumpy(G)
    B_np = cp.asnumpy(B)

    arranged_image_info = np.stack(
        [R_np.reshape(H, W), G_np.reshape(H, W), B_np.reshape(H, W)], axis=2
    )

    arranged_image_info_uint8 = np.clip(arranged_image_info, 0, 255).astype(np.uint8)

    # === Save SFLP Image ===
    img_out = Image.fromarray(arranged_image_info_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved SFLP image to {output_path}")

def EAP_to_SFLP_image_conversion(
    num_pixels: int,
    layout_csv_path: Path,
    input_image_path: Path,
    output_path: Path,
    sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an EAP image to an SFLP image.

    Args:
        num_pixels (int): Number of pixels to sample on.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing EAP image.
        output_path (Path): Path to save SFLP image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """

    # === Load Layout ===
    arranged_pixel_indices_df = pd.read_csv(layout_csv_path, header=None)
    arranged_pixels_indices = arranged_pixel_indices_df.to_numpy()
    H, W = arranged_pixels_indices.shape[:2]

    indices_cp = cp.asarray(arranged_pixels_indices.flatten())
    SFLP_geographic_points = get_SFL_geographic_points(indices=indices_cp, num_points=num_pixels)

    # === Load EAP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    eac_image = np.array(img)
    eac_image_cp = cp.asarray(eac_image)

    # === Convert EAP to SFLP ===
    input_metadata = ProjectionMetadata(EncodingProjection.EAP)
    R, G, B = geographic_points_to_image_info(
        SFLP_geographic_points,
        eac_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    R_np = cp.asnumpy(R)
    G_np = cp.asnumpy(G)
    B_np = cp.asnumpy(B)

    arranged_image_info = np.stack(
        [R_np.reshape(H, W), G_np.reshape(H, W), B_np.reshape(H, W)], axis=2
    )

    arranged_image_info_uint8 = np.clip(arranged_image_info, 0, 255).astype(np.uint8)

    # === Save SFLP Image ===
    img_out = Image.fromarray(arranged_image_info_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved SFLP image to {output_path}")

def SFLP_to_CMP_image_conversion(
    face_size: int,
    layout_csv_path: Path,
    input_image_path: Path,
    output_path: Path,
    input_metadata: SFLPMetadata,
    sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    """
    Converts an SFLP image to an CMP image.

    Args:
        face_size (int): The number of pixels in a row or column on a face of the cube.
        layout_csv_path (Path): Path to the CSV file containing pixel layout.
        input_image_path (Path): Path to image file containing SFLP image.
        output_path (Path): Path to save CMP image.
        SFLP_num_pixels (int): Number of pixels in the input SFLP image.
        sample_type (Image.Resampling): The type of sampling to do to obtain pixel values.
    """

    # === Load Layout ===
    arranged_pixel_indices_df = pd.read_csv(layout_csv_path, header=None)
    arranged_pixels_indices = arranged_pixel_indices_df.to_numpy()
    H, W = arranged_pixels_indices.shape[:2]

    indices_cp = np.asarray(arranged_pixels_indices.flatten())
    CMP_geographic_points = get_CubeMap_geographic_points(CubeMap_tokens=indices_cp, face_size=face_size, is_CMP=True)

    # === Load SFLP Image ===
    input_path = input_image_path
    img = Image.open(input_path)
    sflp_image = np.array(img)
    sflp_image_cp = cp.asarray(sflp_image)

    # === Convert SFLP to CMP ===
    R, G, B = geographic_points_to_image_info(
        geographic_points=CMP_geographic_points,
        input_image=sflp_image_cp,
        input_metadata=input_metadata,
        sample_type=sample_type
    )

    R_np = cp.asnumpy(R)
    G_np = cp.asnumpy(G)
    B_np = cp.asnumpy(B)

    arranged_image_info = np.stack(
        [R_np.reshape(H, W), G_np.reshape(H, W), B_np.reshape(H, W)], axis=2
    )

    arranged_image_info_uint8 = np.clip(arranged_image_info, 0, 255).astype(np.uint8)

    # === Save SFLP Image ===
    img_out = Image.fromarray(arranged_image_info_uint8, mode="RGB")
    img_out.save(output_path)
    print(f"Saved SFLP image to {output_path}")

def reencode_ERP_image(
    input_image_path: Path,
    output_path: Path,
    output_size: tuple,
    sample_type: Image.Resampling = Image.Resampling.NEAREST
):
    W, H = output_size
    
    # === Load ERP Image ===
    img = Image.open(input_image_path).convert("RGB")
    input_width, input_height = img.size

    # === Downsample ===
    if (W, H) != (input_width, input_height):
        img_out = img.resize((W, H), resample=sample_type)
    else:
        img_out = img

    # === Save ERP Image ===
    img_out.save(output_path)
    print(f"Saved ERP image to {output_path}")

# endregion Image Conversion

# region Video Conversion

def ERP_to_SFLP_video_conversion(
    num_pixels: int,
    csv_path: Path,
    input_video_path: Path,
    output_dir: Path,
    output_name: str,
    output_video_type: VideoEncoder = VideoEncoder.H264,
    ffmpeg_path: str='ffmpeg',
    sample_type: Image.Resampling = Image.Resampling.NEAREST,
    bitrates: list[int] = [50000]
):
    """
    Converts an ERP video to an SFLP video.

    Args:
        num_pixels: The number of pixels in a frame of the SFLP video.
        csv_path: The path to the SFLP pixel arrangement.
        input_video_path: The path to the input ERP video.
        dataset_dir: The output directory.
        output_name: The name of the output file.
        output_video_type: The file type for the output video.
        ffmpeg_path: The path to the ffmpeg to use.
        sample_type: The type of resampling to use.
        bitrates: The bitrates to encode for.
    """

    # === Load Pixel Arrangement ===
    arranged_pixel_indices_df = pd.read_csv(csv_path, header=None)
    arranged_pixels_indices = arranged_pixel_indices_df.to_numpy()
    H, W = arranged_pixels_indices.shape[:2]
    indices_cp = cp.asarray(arranged_pixels_indices.flatten())
    SFLP_geographic_points = get_SFL_geographic_points(indices=indices_cp, num_points=num_pixels)

    # === Load Input Video ===
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise IOError(f"Cannot open input video: {input_video_path}")

    # === Make Temporary Output Path ===
    avi_output_path = output_dir / f"{output_name}.avi"

    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore
    out_SFLP = cv2.VideoWriter(str(avi_output_path), fourcc, fps, (W, H))

    if not out_SFLP.isOpened():
        raise IOError(f"Cannot open output video for writing: {avi_output_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames into SFLP...")

    while True:

        time_before_read = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break
            
        time_after_read = time.perf_counter()
        
        print(f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{frame_count}")

        time_before_convert_to_array = time.perf_counter()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ERP_image_cp = cp.asarray(frame_rgb)

        time_after_convert_to_array = time.perf_counter()

        input_metadata = ProjectionMetadata(EncodingProjection.ERP)
        R, G, B = geographic_points_to_image_info(
            SFLP_geographic_points,
            ERP_image_cp,
            input_metadata=input_metadata,
            sample_type=sample_type
        )

        time_after_conversion = time.perf_counter()

        arranged_image_gpu = cp.empty((H, W, 3), dtype=R.dtype)
        arranged_image_gpu[..., 0] = R.reshape(H, W)
        arranged_image_gpu[..., 1] = G.reshape(H, W)
        arranged_image_gpu[..., 2] = B.reshape(H, W)
        arranged_uint8_gpu = cp.clip(arranged_image_gpu, 0, 255).astype(cp.uint8)

        # Transfer the whole image back to np array
        arranged_image_cpu = cp.asnumpy(arranged_uint8_gpu)

        time_between_rearrangement = time.perf_counter()

        # Then clip and convert
        arranged_bgr = cv2.cvtColor(arranged_image_cpu, cv2.COLOR_RGB2BGR)

        time_after_rearrangement = time.perf_counter()

        out_SFLP.write(arranged_bgr)

        time_after_writing = time.perf_counter()

        # print(f"Time to read: {time_after_read - time_before_read}")
        # print(f"Time to array: {time_after_convert_to_array - time_after_read}")
        # print(f"Time to convert: {time_after_conversion - time_after_convert_to_array}")
        # print(f"Time to first part of rearrangement: {time_between_rearrangement - time_after_conversion}")
        # print(f"Time to second part of rearrange: {time_after_rearrangement - time_between_rearrangement}")
        # print(f"Time to write: {time_after_writing - time_after_rearrangement}")

        # Free GPU memory
        del R, G, B, arranged_image_cpu, arranged_image_gpu, arranged_bgr, ERP_image_cp
        cp._default_memory_pool.free_all_blocks()

    cap.release()
    out_SFLP.release()

    final_output_path = ""

    for bitrate in bitrates:
        bitrate_filename_suffix = f"bitrate_{bitrate}"

        encoding_info = f"{output_video_type.name}-Target_{bitrate}k-Preset_Slow"
        output_dir_subfolder = output_dir / encoding_info
        output_dir_subfolder.mkdir(parents=True, exist_ok=True)

        if output_video_type == VideoEncoder.H264:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H264(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.H265:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H265(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.AV1:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.webm"
            convert_to_av1(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)

    if avi_output_path.exists():
        avi_output_path.unlink()

    print(f"SFLP video saved to {final_output_path}")

def ERP_to_CMP_video_conversion(
    face_size: int,
    csv_path: Path,
    input_video_path: Path,
    output_dir: Path,
    output_name: str,
    output_video_type: VideoEncoder = VideoEncoder.H264,
    ffmpeg_path: str='ffmpeg',
    sample_type: Image.Resampling = Image.Resampling.NEAREST,
    bitrates: list[int] = [50000]
):
    """
    Converts an ERP video to a CMP video.

    Args:
        face_size: The number of pixels in a row of a cube face in a frame of the EAC video.
        csv_path: The path to the CubeMap pixel arrangement.
        input_video_path: The path to the input ERP video.
        output_dir: The output directory.
        output_name: The name of the output file.
        output_video_type: The file type for the output video.
        ffmpeg_path: The path to the ffmpeg to use.
        sample_type: The type of resampling to use.
        bitrates: The bitrates to encode for.
    """

    # === Load Pixel Arrangement ===
    arranged_pixels_tokens_df = pd.read_csv(csv_path, header=None)
    arranged_pixels_tokens = arranged_pixels_tokens_df.to_numpy()
    H, W = arranged_pixels_tokens.shape[:2]
    tokens = arranged_pixels_tokens.flatten()

    CMP_geographic_points = get_CubeMap_geographic_points(tokens, face_size, is_CMP=True)

    # === Load Input Video ===
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise IOError(f"Cannot open input video: {input_video_path}")

    # === Make Temporary Output Path ===
    avi_output_path = output_dir / f"{output_name}.avi"

    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore
    out_CMP = cv2.VideoWriter(str(avi_output_path), fourcc, fps, (W, H))

    if not out_CMP.isOpened():
        raise IOError(f"Cannot open output video for writing: {avi_output_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames into CMP...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{frame_count}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ERP_image_cp = cp.asarray(frame_rgb)

        input_metadata = ProjectionMetadata(EncodingProjection.ERP)
        R, G, B = geographic_points_to_image_info(
            geographic_points=CMP_geographic_points,
            input_image=ERP_image_cp,
            input_metadata=input_metadata,
            sample_type=sample_type
        )

        arranged_image_gpu = cp.empty((H, W, 3), dtype=R.dtype)
        arranged_image_gpu[..., 0] = R.reshape(H, W)
        arranged_image_gpu[..., 1] = G.reshape(H, W)
        arranged_image_gpu[..., 2] = B.reshape(H, W)
        arranged_uint8_gpu = cp.clip(arranged_image_gpu, 0, 255).astype(cp.uint8)

        # Transfer the whole image back to np array
        arranged_image_cpu = cp.asnumpy(arranged_uint8_gpu)

        # Then clip and convert
        arranged_bgr = cv2.cvtColor(arranged_image_cpu, cv2.COLOR_RGB2BGR)

        out_CMP.write(arranged_bgr)

        # Free GPU memory
        del R, G, B, arranged_image_gpu, arranged_image_cpu, arranged_bgr, ERP_image_cp
        cp._default_memory_pool.free_all_blocks()

    cap.release()
    out_CMP.release()

    final_output_path = ""

    for bitrate in bitrates:
        bitrate_filename_suffix = f"bitrate_{bitrate}"
        
        encoding_info = f"{output_video_type.name}-Target_{bitrate}k-Preset_Slow"
        output_dir_subfolder = output_dir / encoding_info
        output_dir_subfolder.mkdir(parents=True, exist_ok=True)

        if output_video_type == VideoEncoder.H264:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H264(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.H265:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H265(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.AV1:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.webm"
            convert_to_av1(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)

    if avi_output_path.exists():
        avi_output_path.unlink()

    print(f"CMP video saved to {final_output_path}")

def ERP_to_EAC_video_conversion(
    face_size: int,
    csv_path: Path,
    input_video_path: Path,
    output_dir: Path,
    output_name: str,
    output_video_type: VideoEncoder = VideoEncoder.H264,
    ffmpeg_path: str='ffmpeg',
    sample_type: Image.Resampling = Image.Resampling.NEAREST,
    bitrates: list[int] = [50000]
):
    """
    Converts an ERP video to a EAC video.

    Args:
        face_size: The number of pixels in a row of a cube face in a frame of the EAC video.
        csv_path: The path to the CubeMap pixel arrangement.
        input_video_path: The path to the input ERP video.
        output_dir: The output directory.
        output_name: The name of the output file.
        output_video_type: The file type for the output video.
        ffmpeg_path: The path to the ffmpeg to use.
        sample_type: The type of resampling to use.
        The bitrates to encode for.
    """

    # === Load Pixel Arrangement ===
    arranged_pixels_tokens_df = pd.read_csv(csv_path, header=None)
    arranged_pixels_tokens = arranged_pixels_tokens_df.to_numpy()
    H, W = arranged_pixels_tokens.shape[:2]
    tokens = arranged_pixels_tokens.flatten()

    EAC_geographic_points = get_CubeMap_geographic_points(tokens, face_size, is_CMP=False)

    # === Load Input Video ===
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise IOError(f"Cannot open input video: {input_video_path}")

    # === Make Temporary Output Path ===
    avi_output_path = output_dir / f"{output_name}.avi"

    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore
    out_EAC = cv2.VideoWriter(str(avi_output_path), fourcc, fps, (W, H))

    if not out_EAC.isOpened():
        raise IOError(f"Cannot open output video for writing: {avi_output_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames into EAC...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{frame_count}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ERP_image_cp = cp.asarray(frame_rgb)

        input_metadata = ProjectionMetadata(EncodingProjection.ERP)
        R, G, B = geographic_points_to_image_info(
            geographic_points=EAC_geographic_points,
            input_image=ERP_image_cp,
            input_metadata=input_metadata,
            sample_type=sample_type
        )

        arranged_image_gpu = cp.empty((H, W, 3), dtype=R.dtype)
        arranged_image_gpu[..., 0] = R.reshape(H, W)
        arranged_image_gpu[..., 1] = G.reshape(H, W)
        arranged_image_gpu[..., 2] = B.reshape(H, W)
        arranged_uint8_gpu = cp.clip(arranged_image_gpu, 0, 255).astype(cp.uint8)

        # Transfer the whole image back to np array
        arranged_image_cpu = cp.asnumpy(arranged_uint8_gpu)

        # Then clip and convert
        arranged_bgr = cv2.cvtColor(arranged_image_cpu, cv2.COLOR_RGB2BGR)

        out_EAC.write(arranged_bgr)

        # Free GPU memory
        del R, G, B, arranged_image_gpu, arranged_image_cpu, arranged_bgr, ERP_image_cp
        cp._default_memory_pool.free_all_blocks()

    cap.release()
    out_EAC.release()

    final_output_path = ""

    for bitrate in bitrates:
        bitrate_filename_suffix = f"bitrate_{bitrate}"

        encoding_info = f"{output_video_type.name}-Target_{bitrate}k-Preset_Slow"
        output_dir_subfolder = output_dir / encoding_info
        output_dir_subfolder.mkdir(parents=True, exist_ok=True)

        if output_video_type == VideoEncoder.H264:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H264(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.H265:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H265(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.AV1:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.webm"
            convert_to_av1(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)

    if avi_output_path.exists():
        avi_output_path.unlink()

    print(f"CMP video saved to {final_output_path}")

def ERP_to_EAP_video_conversion(
    input_video_path: Path,
    output_dir: Path,
    output_name: str,
    output_size: Tuple[int, int],
    output_video_type: VideoEncoder = VideoEncoder.H264,
    ffmpeg_path: str='ffmpeg',
    sample_type: Image.Resampling = Image.Resampling.NEAREST,
    bitrates: list[int] = [50000]
):
    """
    Converts an ERP video to an EAP video.

    Args:
        input_video_path: The path to the input ERP video.
        output_dir: The output directory.
        output_name: The name of the output file.
        output_size: The dimensions of the output video.
        output_video_type: The file type for the output video.
        ffmpeg_path: The path to the ffmpeg to use.
        sample_type: The type of resampling to use.
        bitrates: The bitrates to encode for.
    """

    W, H = output_size

    EAP_geographic_points = get_EAP_geographic_points(output_size)

    # === Load Input Video ===
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise IOError(f"Cannot open input video: {input_video_path}")

    # === Make Temporary Output Path ===
    avi_output_path = output_dir / f"{output_name}.avi"

    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore
    out_SFLP = cv2.VideoWriter(str(avi_output_path), fourcc, fps, (W, H))

    if not out_SFLP.isOpened():
        raise IOError(f"Cannot open output video for writing: {avi_output_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames into EAP...")

    while True:

        time_before_read = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break
            
        time_after_read = time.perf_counter()
        
        print(f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{frame_count}")

        time_before_convert_to_array = time.perf_counter()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ERP_image_cp = cp.asarray(frame_rgb)

        time_after_convert_to_array = time.perf_counter()

        input_metadata = ProjectionMetadata(EncodingProjection.ERP)
        R, G, B = geographic_points_to_image_info(
            EAP_geographic_points,
            ERP_image_cp,
            input_metadata=input_metadata,
            sample_type=sample_type
        )

        time_after_conversion = time.perf_counter()

        arranged_image_gpu = cp.empty((H, W, 3), dtype=R.dtype)
        arranged_image_gpu[..., 0] = R.reshape(H, W)
        arranged_image_gpu[..., 1] = G.reshape(H, W)
        arranged_image_gpu[..., 2] = B.reshape(H, W)
        arranged_uint8_gpu = cp.clip(arranged_image_gpu, 0, 255).astype(cp.uint8)

        # Transfer the whole image back to np array
        arranged_image_cpu = cp.asnumpy(arranged_uint8_gpu)

        time_between_rearrangement = time.perf_counter()

        # Then clip and convert
        arranged_bgr = cv2.cvtColor(arranged_image_cpu, cv2.COLOR_RGB2BGR)

        time_after_rearrangement = time.perf_counter()

        out_SFLP.write(arranged_bgr)

        time_after_writing = time.perf_counter()

        # print(f"Time to read: {time_after_read - time_before_read}")
        # print(f"Time to array: {time_after_convert_to_array - time_after_read}")
        # print(f"Time to convert: {time_after_conversion - time_after_convert_to_array}")
        # print(f"Time to first part of rearrangement: {time_between_rearrangement - time_after_conversion}")
        # print(f"Time to second part of rearrange: {time_after_rearrangement - time_between_rearrangement}")
        # print(f"Time to write: {time_after_writing - time_after_rearrangement}")

        # Free GPU memory
        del R, G, B, arranged_image_cpu, arranged_image_gpu, arranged_bgr, ERP_image_cp
        cp._default_memory_pool.free_all_blocks()

    cap.release()
    out_SFLP.release()

    final_output_path = ""

    for bitrate in bitrates:
        bitrate_filename_suffix = f"bitrate_{bitrate}"

        encoding_info = f"{output_video_type.name}-Target_{bitrate}k-Preset_Slow"
        output_dir_subfolder = output_dir / encoding_info
        output_dir_subfolder.mkdir(parents=True, exist_ok=True)

        if output_video_type == VideoEncoder.H264:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H264(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.H265:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H265(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.AV1:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.webm"
            convert_to_av1(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
            
    if avi_output_path.exists():
        avi_output_path.unlink()

    print(f"SFLP video saved to {final_output_path}")

def reencode_ERP_video(
    input_video_path: Path,
    output_dir: Path,
    output_name: str,
    output_size: Optional[tuple[int, int]] = None,
    output_video_type: VideoEncoder = VideoEncoder.H264,
    ffmpeg_path: str='ffmpeg',
    sample_type: Image.Resampling = Image.Resampling.NEAREST,
    bitrates: list[int] = [50000]
):
    """
    Re-encodes an ERP video to a standardized format and resolution.

    Args:
        input_video_path (Path): Path to input ERP video.
        output_dir (Path): Output directory.
        output_name (str): Output name prefix (no extension).
        output_size (tuple[int, int], optional): Output resolution (width, height). 
            Defaults to the input video resolution.
        bitrates: The bitrates to encode for.
    """
    # === Load Input Video ===
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open input video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine output size
    if output_size is None:
        width, height = input_width, input_height
    else:
        width, height = output_size

    print(f"Processing {frame_count} frames for ERP re-encoding...")
    print(f"Output resolution: {width}x{height}")

    # === Temporary AVI Output ===
    avi_output_path = output_dir / f"{output_name}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore
    out_erp = cv2.VideoWriter(str(avi_output_path), fourcc, fps, (width, height))

    if not out_erp.isOpened():
        raise IOError(f"Cannot open output video for writing: {avi_output_path}")

    # === Process Frames ===
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{frame_count}")

        # Convert to RGB for resizing (PIL does better resampling)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        if (width, height) != (input_width, input_height):
            frame_resized = frame_pil.resize((width, height), resample=sample_type)
        else:
            frame_resized = frame_pil

        frame_resized_np = np.array(frame_resized)
        frame_bgr = cv2.cvtColor(frame_resized_np, cv2.COLOR_RGB2BGR)

        out_erp.write(frame_bgr)

        # Free GPU memory
        del frame_rgb, frame_resized_np, frame_pil
        cp._default_memory_pool.free_all_blocks()

    cap.release()
    out_erp.release()

    final_output_path = ""

    for bitrate in bitrates:
        bitrate_filename_suffix = f"bitrate_{bitrate}"

        encoding_info = f"{output_video_type.name}-Target_{bitrate}k-Preset_Slow"
        output_dir_subfolder = output_dir / encoding_info
        output_dir_subfolder.mkdir(parents=True, exist_ok=True)

        if output_video_type == VideoEncoder.H264:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H264(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.H265:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.mp4"
            convert_to_H265(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)
        elif output_video_type == VideoEncoder.AV1:
            final_output_path = output_dir_subfolder / f"{output_name}-{bitrate_filename_suffix}.webm"
            convert_to_av1(avi_output_path, final_output_path, ffmpeg_path=ffmpeg_path, bitrate=bitrate)

    if avi_output_path.exists():
        avi_output_path.unlink()

    print(f"Re-encoded ERP video saved to {final_output_path}")

# endregion Video Conversion

#
# endregion VR Projection Type Conversions
#