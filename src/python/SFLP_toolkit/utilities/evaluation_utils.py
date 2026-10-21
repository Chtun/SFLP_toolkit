
"""Evaluation module.

This module includes functions to evaluate the visual quality of different VR encoding methods (SFLP, ERP, CMP, EAC) for Image and Video.

Methods:

    calculate_spherical_PSNR: Calculate the Spherical Peak Signal-to-Noise Ratio (S-PSNR) for an input 360-degree image/video frame.
    compute_PSNR: Computes the PSNR value from a set of ground truth samples and compared samples.
    calculate_spherical_SSIM: Calculate the Spherical Structural Similarity Index (S-SSIM) for an input 360-degree image/video frame.
    evaluate_video_quality: Evaluates the quality of a 360-degree video by calculating Spherical PSNR or Spherical SSIM against a ground truth 360-degree video.
    evaluate_image_quality: Evaluates the quality of a 360-degree image by calculating Spherical PSNR or Spherical SSIM against a ground truth 360-degree image.
    
    extract_average_quality: Calculates the average quality over all frames for each video encoding method.
    generate_rd_curve: Generates a Rate-Distortion curve for a single video, comparing multiple encoding methods.
    save_average_quality: Saves the average quality values to a data table.
    compute_bd_rate: Compute Bjøntegaard Delta Rate (BD-Rate) between two RD curves.
    get_exact_bitrate: Uses ffprobe to extract the exact bitrate of a video in bits per second.

"""

from enum import Enum
from pathlib import Path
import cupy as cp
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tqdm import tqdm

from SFLP_toolkit.utilities.pixel_arrangement_utils import get_SFL_geographic_points
from SFLP_toolkit.utilities.file_conversion_utils import *

class WeightingType(Enum):
    EQUAL = "EQUAL"
    VIEWING_BEHAVIOR = "VIEWING_BEHAVIOR" # VIEWING_BEHAVIOR for weights based on where users typically look

class ComparisonMetric(Enum):
    SPSNR = "SPSNR"
    SSSIM = "SSSIM"


#
# region RGB to YUV
#

def to_luminance(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

def to_chroma_u(rgb):
    return -0.14713 * rgb[0] - 0.28886 * rgb[1] + 0.436 * rgb[2]

def to_chroma_v(rgb):
    return 0.615 * rgb[0] - 0.51499 * rgb[1] - 0.10001 * rgb[2]

def rgb_to_yuv_vectorized(samples):
    # samples is a tuple: (cp.ndarray, cp.ndarray, cp.ndarray)
    # Stack them to create a (3, N) matrix
    samples_stacked = cp.stack(samples) 
    
    weights = cp.array([
        [ 0.299,    0.587,    0.114   ],
        [-0.14713, -0.28886,  0.436   ],
        [ 0.615,   -0.51499, -0.10001 ]
    ])
    
    # Perform the transformation
    # result will be a (3, N) array where result[0]=Y, result[1]=U, result[2]=V
    return cp.dot(weights, samples_stacked.reshape(3, -1))

#
# endregion RGB to YUV
#

#
# region Spherical-PSNR
#

def calculate_spherical_PSNR(
    num_evaluation_points: int,
    ground_truth_image: cp.ndarray,
    ground_truth_metadata: ProjectionMetadata,
    compared_image: cp.ndarray,
    compared_metadata: ProjectionMetadata,
    sample_type: Image.Resampling,
    weighting_type: WeightingType,
    print_errors: bool = False,
    visualize_error: bool = False
):
    """
    Calculate the Spherical Peak Signal-to-Noise Ratio (S-PSNR) for an input image.
    
    Args:
        num_evaluation_points: The number of Spherical Fibonacci Lattice points to generate for S-PSNR calculations (choose number that is co-prime with num_pixels).
        ground_truth_image: The image that represents the ground truth of the image content.
        ground_truth_metadata: The metadata of the ground truth image (currently only supports ERP).
        compared_image: The image being compared to the ground truth.
        compared_metadata: The metadata of the compared image (SFLP, ERP, CMP, EAC, etc.)
        sample_type: The type of sampling to conduct.
        weighting_type: The weighting scheme to apply to the samples.

    Returns:
        The spherical PSNR calculated.
    """


    ground_truth_projection_type = ground_truth_metadata.projection_type
    compared_projection_type = compared_metadata.projection_type

    if ground_truth_projection_type != EncodingProjection.ERP:
        raise ValueError("Only ERP is supported for ground truth image!")
    
    all_indices = cp.arange(num_evaluation_points)
    fb_geographic_points = get_SFL_geographic_points(indices=all_indices, num_points=num_evaluation_points)

    ground_truth_samples = sample_ERP(
        geographic_points=fb_geographic_points,
        input_image=ground_truth_image,
        sample_type=sample_type
    )

    lat_deg = fb_geographic_points[0]
    lat = lat_deg * (cp.pi / 180.0)
    lon_deg = fb_geographic_points[1]

    if weighting_type == WeightingType.EQUAL:
        weights = None
    elif weighting_type == WeightingType.VIEWING_BEHAVIOR:
        sigma = cp.pi / 6
        weights = cp.exp(-0.5 * (lat / sigma) ** 2)
    else:
        raise ValueError(f"Weighting type {weighting_type} is not supported!")

    if compared_projection_type == EncodingProjection.ERP:
        compared_samples = sample_ERP(geographic_points=fb_geographic_points, input_image=compared_image, sample_type=sample_type)
    elif compared_projection_type == EncodingProjection.CMP:
        compared_samples = sample_CMP(geographic_points=fb_geographic_points, input_image=compared_image, sample_type=sample_type)
    elif compared_projection_type == EncodingProjection.EAC:
        compared_samples = sample_EAC(geographic_points=fb_geographic_points, input_image=compared_image, sample_type=sample_type)
    elif compared_projection_type == EncodingProjection.EAP:
        compared_samples = sample_EAP(geographic_points=fb_geographic_points, input_image=compared_image, sample_type=sample_type)
    elif compared_projection_type == EncodingProjection.SFLP:
        if not isinstance(compared_metadata, SFLPMetadata):
            raise TypeError("Expected SFLPMetadata for SFLP projection")
        SFLP_metadata: SFLPMetadata = compared_metadata
        compared_samples = sample_SFLP(
            geographic_points=fb_geographic_points,
            input_image=compared_image,
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
        raise ValueError(f"Compared Projection Type (EncodingProjection.{compared_projection_type}) is not supported for S-PSNR calculations!")

    # Convert to YUV using the vectorized function
    gt_yuv = rgb_to_yuv_vectorized(ground_truth_samples)
    compared_yuv = rgb_to_yuv_vectorized(compared_samples)

    if visualize_error or print_errors:
        # Squared differences in YUV space
        Y_diff = (gt_yuv[0] - compared_yuv[0]) ** 2
        U_diff = (gt_yuv[1] - compared_yuv[1]) ** 2
        V_diff = (gt_yuv[2] - compared_yuv[2]) ** 2

        # total_diff using the 6:1:1 weighting (divided by total weights: 8)
        total_diff = (6 * Y_diff + U_diff + V_diff) / 8
        
        # Flatten and process for top errors (using ravel to handle any input shape)
        total_diff_flat = total_diff.ravel()
        sorted_indices = cp.argsort(total_diff_flat)[::-1]
        top_indices = sorted_indices[:50]
        top_diff = total_diff_flat[top_indices]

        total_diff_np = cp.asnumpy(total_diff_flat)
        lat_np = cp.asnumpy(lat_deg.ravel())
        lon_np = cp.asnumpy(lon_deg.ravel())

        top_geographic = cp.stack([lat_deg.ravel(), lon_deg.ravel()], axis=1)[top_indices]

        if weighting_type == WeightingType.EQUAL:
            mean_error = cp.mean(total_diff)
        else:
            mean_error = cp.sum(weights * total_diff) / cp.sum(weights)

        std_error = cp.std(total_diff)

        if print_errors:
            print(f"Top Errors (6:1:1 YUV): {top_diff}")
            
            # Format the coordinates to avoid scientific notation and use 2 decimals
            geo_str = cp.array2string(top_geographic, 
                                      formatter={'float_kind': lambda x: f"{x:7.2f}"},
                                      separator=', ')
            
            print(f"Top error points (Lat, Lon):\n{geo_str}")
            print(f"\nAverage error (6:1:1 YUV): {mean_error:.6f}")
            print(f"Standard deviation of error (6:1:1 YUV): {std_error:.6f}")

            print()
            print()

        upper_limit = 50

        if visualize_error:
            plt.figure(figsize=(10, 6))
            sc = plt.scatter(lon_np, lat_np, c=total_diff_np, s=5, cmap='viridis', alpha=0.8, vmin=0, vmax=upper_limit)
            plt.colorbar(sc, label='Weighted YUV Error (6:1:1)', extend='max')
            plt.title("YUV Error Heatmap on Sphere (Lat vs Lon)")
            plt.xlabel("Longitude (degrees)")
            plt.ylabel("Latitude (degrees)")
            plt.grid(True, alpha=0.3)
            plt.show()

    # Compute PSNR using the new efficient channel-weighting logic
    SPSNR = compute_PSNR(
        gt_yuv.reshape(3, -1), 
        compared_yuv.reshape(3, -1), 
        weights=weights.ravel() if weights is not None else None,
        channel_weights=[6, 1, 1]
    )

    return SPSNR

def compute_PSNR(ground_truth_samples: cp.ndarray, compared_samples: cp.ndarray, weights=None, channel_weights=None):
    """
    Args:
        ground_truth_samples: (C, N) CuPy array of samples.
        compared_samples: (C, N) CuPy array of samples.
        weights: (N,) Spherical weights (latitude-based).
        channel_weights: (C,) Channel-wise weights (e.g., [6, 1, 1]).
    """
    max_pixel_value = 255.0
    
    # Calculate squared difference: shape (C, N)
    diff_sq = (ground_truth_samples - compared_samples) ** 2

    if weights is not None:
        # 1. Apply spherical weights across the spatial dimension (N)
        # result shape: (C,)
        mse_per_channel = cp.sum(diff_sq * weights, axis=1) / cp.sum(weights)
    else:
        mse_per_channel = cp.mean(diff_sq, axis=1)

    if channel_weights is not None:
        # 2. Apply channel weights (Y:6, U:1, V:1) across the channel dimension (C)
        channel_weights = cp.array(channel_weights)
        mse = cp.sum(mse_per_channel * channel_weights) / cp.sum(channel_weights)
    else:
        mse = cp.mean(mse_per_channel)

    eps = 1e-12
    mse_safe = cp.maximum(mse, eps)

    # GPU-side PSNR
    PSNR = 10.0 * cp.log10((max_pixel_value ** 2) / mse_safe)

    # Return CuPy scalar
    return PSNR

#
# endregion Spherical-PSNR
#

#
# region Spherical-SSIM
#

def calculate_spherical_SSIM(
    num_evaluation_points: int,
    ground_truth_image: cp.ndarray,
    ground_truth_metadata: ProjectionMetadata,
    compared_image: cp.ndarray,
    compared_metadata: ProjectionMetadata,
    sample_type: Image.Resampling,
    weighting_type: WeightingType,
    window_size: int = 11,
    print_errors: bool = False,
    visualize_error: bool = False
):
    """
    Calculate the Spherical Structural Similarity Index (S-SSIM) using local 11x11 patches.
    """

    if ground_truth_metadata.projection_type != EncodingProjection.ERP:
        raise ValueError("Only ERP is supported for ground truth image!")

    # Generate Fibonacci Lattice Centers
    all_indices = cp.arange(num_evaluation_points)
    lat_centers, lon_centers = get_SFL_geographic_points(indices=all_indices, num_points=num_evaluation_points)
    
    # Define the Local 11x11 Neighborhood Offsets using a small angular delta. 0.08 degrees is roughly 1 pixel at 4K ERP.
    delta = 0.08 
    offset = cp.linspace(-(window_size // 2) * delta, (window_size // 2) * delta, window_size)
    dx, dy = cp.meshgrid(offset, offset)
    dx = dx.ravel() # shape (121,)
    dy = dy.ravel() # shape (121,)

    # Create the Grid of all points to sample (Num_Evaluation_Points * 121)
    # patch_geographic_points shape: (2, N * 121)
    patch_lats = (lat_centers[:, None] + dy).ravel()
    patch_lons = (lon_centers[:, None] + dx).ravel()
    
    # Mirror/Clamp latitudes at the poles to avoid invalid geographic coordinates
    patch_lats = cp.clip(patch_lats, -90.0, 90.0)
    patch_geo = cp.stack([patch_lats, patch_lons])

    # Sample RGB from Ground Truth (ERP)
    gt_rgb_samples = sample_ERP(
        geographic_points=patch_geo,
        input_image=ground_truth_image,
        sample_type=sample_type
    )

    # Sample RGB from Compared Image
    compared_projection_type = compared_metadata.projection_type
    if compared_projection_type == EncodingProjection.ERP:
        comp_rgb_samples = sample_ERP(patch_geo, compared_image, sample_type)
    elif compared_projection_type == EncodingProjection.CMP:
        comp_rgb_samples = sample_CMP(patch_geo, compared_image, sample_type)
    elif compared_projection_type == EncodingProjection.EAC:
        comp_rgb_samples = sample_EAC(patch_geo, compared_image, sample_type)
    elif compared_projection_type == EncodingProjection.EAP:
        comp_rgb_samples = sample_EAP(patch_geo, compared_image, sample_type)
    elif compared_projection_type == EncodingProjection.SFLP:
        if not isinstance(compared_metadata, SFLPMetadata):
            raise TypeError("Expected SFLPMetadata for SFLP projection")
        SFLP_metadata: SFLPMetadata = compared_metadata
        comp_rgb_samples = sample_SFLP(
            geographic_points=patch_geo,
            input_image=compared_image,
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
        raise ValueError(f"Projection {compared_projection_type} not supported for S-SSIM.")

    gt_y = to_luminance(gt_rgb_samples)
    comp_y = to_luminance(comp_rgb_samples)

    # Group into Patches and Compute Local SSIM
    # Reshape to (num_evaluation_points, window_size^2)
    gt_patches = gt_y.reshape(num_evaluation_points, window_size**2)
    comp_patches = comp_y.reshape(num_evaluation_points, window_size**2)
    
    # Calculate SSIM for every point on the lattice
    ssim_map = compute_ssim_vectorized(gt_patches, comp_patches)

    if visualize_error or print_errors:
        # structural_error = 1.0 - SSIM (higher is worse)
        structural_error = 1.0 - ssim_map
        
        # Sort and find top 50 worst structural failures
        sorted_indices = cp.argsort(structural_error)[::-1]
        top_indices = sorted_indices[:50]
        top_error = structural_error[top_indices]
        top_geographic = cp.stack([lat_centers, lon_centers], axis=1)[top_indices]

        if print_errors:
            print(f"Top Structural Failures (1-SSIM): {top_error}")
            print(f"Top error points (Lat, Lon): {top_geographic}")
            print(f"\nAverage error: {cp.mean(structural_error):.6f}")
            print(f"Standard deviation of error: {cp.std(structural_error):.6f}")

        # Preparation for Plotting
        error_np = cp.asnumpy(structural_error)
        lat_np = cp.asnumpy(lat_centers)
        lon_np = cp.asnumpy(lon_centers)

        if visualize_error:
            plt.figure(figsize=(10, 6))
            # Use 'magma' or 'viridis'; vmin=0, vmax=0.1 captures subtle structural losses
            sc = plt.scatter(lon_np, lat_np, c=error_np, s=5, cmap='magma', alpha=0.8, vmin=0, vmax=0.1)
            plt.colorbar(sc, label='Structural Error (1 - SSIM)', extend='max')
            plt.title(f"Spherical SSIM Error Heatmap ({compared_projection_type.name})")
            plt.xlabel("Longitude (degrees)")
            plt.ylabel("Latitude (degrees)")
            plt.grid(True, alpha=0.3)
            plt.show()

    # Apply Spherical Weighting
    if weighting_type == WeightingType.EQUAL:
        final_ssim = cp.mean(ssim_map)
    elif weighting_type == WeightingType.VIEWING_BEHAVIOR:
        # Re-calculate center weights based on center latitudes
        lat_rad = lat_centers * (cp.pi / 180.0)
        sigma = cp.pi / 6
        weights = cp.exp(-0.5 * (lat_rad / sigma) ** 2)
        final_ssim = cp.sum(weights * ssim_map) / cp.sum(weights)
    else:
        raise ValueError(f"Weighting {weighting_type} not supported.")

    return final_ssim

def compute_ssim_vectorized(X, Y, bit_depth=8):
    """
    Computes SSIM for multiple patches at once.
    X, Y: (N, Window_Size^2) CuPy arrays
    """
    L = (2**bit_depth) - 1
    K1, K2 = 0.01, 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Means
    mu_x = cp.mean(X, axis=1)
    mu_y = cp.mean(Y, axis=1)
    
    # Variances and Covariance
    # We use ddof=1 for sample variance to match standard SSIM implementations
    sigma_x_sq = cp.var(X, axis=1)
    sigma_y_sq = cp.var(Y, axis=1)
    sigma_xy = cp.mean(X * Y, axis=1) - (mu_x * mu_y)

    # SSIM Formula: (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / (mu_x^2 + mu_y^2 + C1)(sigma_x_sq + sigma_y_sq + C2)
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    
    # Stabilize against zero denominator
    ssim_map = numerator / (denominator + 1e-10)
    
    return cp.clip(ssim_map, -1.0, 1.0)

#
# endregion Spherical-SSIM
#

#
# region Evaluation Functions
#

def evaluate_video_quality(
    num_evaluation_points: int,    
    ground_truth_video_path: Path,
    ground_truth_metadata: ProjectionMetadata,
    compared_video_path: Path,
    compared_metadata: ProjectionMetadata,
    sample_type: Image.Resampling = Image.Resampling.BILINEAR,
    weighting_type: WeightingType = WeightingType.VIEWING_BEHAVIOR,
    comparison_metric: ComparisonMetric = ComparisonMetric.SPSNR,
    print_errors: bool = False,
    visualize_error: bool = False,
):
    """
    Evaluates the quality of a 360-degree video by calculating Spherical PSNR against a ground truth 360-degree video.

    Args:
        num_evaluation_points: The number of points to sample at.
        ground_truth_video_path: The path to the ground truth video.
        ground_truth_metadata: The metadata of the ground truth video.
        compared_video_path: The path to the compared video.
        compared_metadata: The metadata of the compared video.
        sample_type: The type of sampling to conduct (nearest-neighbor or bilinear).
        weighting_type: The weighting scheme to apply to the samples.
    
    Returns:
        A dictionary containing frame # to S-PSNR value pairings.
    """
    
    # === Load Input Videos ===
    cap_truth = cv2.VideoCapture(str(ground_truth_video_path))
    if not cap_truth.isOpened():
        raise IOError(f"Cannot open ground truth video: {ground_truth_video_path}")
    
    cap_compared = cv2.VideoCapture(str(compared_video_path))
    if not cap_compared.isOpened():
        raise IOError(f"Cannot open compared video: {compared_video_path}")
    
    if cap_truth.get(cv2.CAP_PROP_FRAME_COUNT) != cap_compared.get(cv2.CAP_PROP_FRAME_COUNT):
        raise ValueError(f"Ground truth video ({ground_truth_video_path}) and compared video ({compared_video_path}) do not have the same number of frames!")

    frame_count = int(cap_truth.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Computing {comparison_metric} for {frame_count} frames ...")

    # === Create Dictionary to Save Quality Values ===\
    quality_gpu_per_frame = {}

    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        ret_truth, frame_truth = cap_truth.read()
        ret_compared, frame_compared = cap_compared.read()

        if not ret_truth:
            break

        frame_truth = cv2.cvtColor(frame_truth, cv2.COLOR_BGR2RGB)
        frame_truth_cp = cp.asarray(frame_truth)

        frame_compared = cv2.cvtColor(frame_compared, cv2.COLOR_BGR2RGB)
        frame_compared_cp = cp.asarray(frame_compared)

        if comparison_metric == ComparisonMetric.SPSNR:
            quality_gpu = calculate_spherical_PSNR(
                num_evaluation_points=num_evaluation_points,
                ground_truth_image=frame_truth_cp,
                ground_truth_metadata=ground_truth_metadata,
                compared_image=frame_compared_cp,
                compared_metadata=compared_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
                print_errors=print_errors,
                visualize_error=visualize_error,
            )
        elif comparison_metric == ComparisonMetric.SSSIM:
            quality_gpu = calculate_spherical_SSIM(
                num_evaluation_points=num_evaluation_points,
                ground_truth_image=frame_truth_cp,
                ground_truth_metadata=ground_truth_metadata,
                compared_image=frame_compared_cp,
                compared_metadata=compared_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
                print_errors=print_errors,
                visualize_error=visualize_error,
            )
        else:
            raise ValueError(f"Comparison Metric {comparison_metric} not supported!")

        current_frame_num = frame_idx + 1
        quality_gpu_per_frame[current_frame_num] = quality_gpu

    frame_nums = sorted(quality_gpu_per_frame.keys())
    gpu_vals = [quality_gpu_per_frame[f] for f in frame_nums]

    # stack to one CuPy array and sync once
    gpu_array = cp.stack(gpu_vals)          # shape: (num_frames,)
    qualities_np = cp.asnumpy(gpu_array)    # single sync

    quality_per_frame = {
        frame: float(val) for frame, val in zip(frame_nums, qualities_np)
    }

    return quality_per_frame

def evaluate_image_quality(
    num_evaluation_points: int,    
    ground_truth_image_path: Path,
    ground_truth_metadata: ProjectionMetadata,
    compared_image_path: Path,
    compared_metadata: ProjectionMetadata,
    sample_type: Image.Resampling = Image.Resampling.BILINEAR,
    weighting_type: WeightingType = WeightingType.VIEWING_BEHAVIOR,
    comparison_metric: ComparisonMetric = ComparisonMetric.SPSNR,
    print_errors: bool = False,
    visualize_error: bool = False
):
    """
    Evaluates the quality of a 360-degree image by calculating Spherical PSNR against a ground truth 360-degree image.
    
    Args:
        num_evaluation_points (int): The number of points to sample for S-PSNR calculation.
        ground_truth_image_path (Path): The file path to the ground truth image.
        ground_truth_metadata (any): Metadata for the ground truth image projection.
        compared_image_path (Path): The file path to the compared image.
        compared_metadata (any): Metadata for the compared image projection.
        sample_type (Image.Resampling): The sampling method to use.
        weighting_type (any): The weighting scheme to apply to the samples.

    Returns:
        float: The single S-PSNR value representing the quality of the compared image.
    """

    # === Load Input Images ===
    frame_truth = cv2.imread(str(ground_truth_image_path))
    if frame_truth is None:
        raise IOError(f"Cannot open ground truth image: {ground_truth_image_path}")
    
    frame_compared = cv2.imread(str(compared_image_path))
    if frame_compared is None:
        raise IOError(f"Cannot open compared image: {compared_image_path}")

    # === Convert Images and Move to GPU ===
    # Convert from BGR to RGB
    frame_truth = cv2.cvtColor(frame_truth, cv2.COLOR_BGR2RGB)
    # Convert to a Cupy array for GPU processing
    frame_truth_cp = cp.asarray(frame_truth)

    frame_compared = cv2.cvtColor(frame_compared, cv2.COLOR_BGR2RGB)
    frame_compared_cp = cp.asarray(frame_compared)

    # === Calculate Spherical PSNR ===
    # Call the function to get a single S-PSNR value
    if comparison_metric == ComparisonMetric.SPSNR:
        quality_gpu = calculate_spherical_PSNR(
            num_evaluation_points=num_evaluation_points,
            ground_truth_image=frame_truth_cp,
            ground_truth_metadata=ground_truth_metadata,
            compared_image=frame_compared_cp,
            compared_metadata=compared_metadata,
            sample_type=sample_type,
            weighting_type=weighting_type,
            print_errors=print_errors,
            visualize_error=visualize_error
        )
    elif comparison_metric == ComparisonMetric.SSSIM:
        quality_gpu = calculate_spherical_SSIM(
            num_evaluation_points=num_evaluation_points,
            ground_truth_image=frame_truth_cp,
            ground_truth_metadata=ground_truth_metadata,
            compared_image=frame_compared_cp,
            compared_metadata=compared_metadata,
            sample_type=sample_type,
            weighting_type=weighting_type,
            print_errors=print_errors,
            visualize_error=visualize_error
        )
    else:
        raise ValueError(f"Comparison Metric {comparison_metric} not supported!")

    quality = float(cp.asnumpy(quality_gpu))
    return quality

#
# endregion Evaluation Functions
#

#
# region Analysis
#

def extract_average_quality(file_path: Path) -> pd.Series:
    """
    Calculates the average quality over all frames for each video encoding method.

    Args:
        file_path: The path to the .csv file that holds the quality values.

    Returns:
        The average quality value for each video encoding method.
    """
    df = pd.read_csv(file_path)

    # Exclude the first column 'Frame' and calculate the mean of the remaining columns
    avg_values = df.iloc[:, 1:].mean()

    return avg_values

def generate_rd_curve(
        video_data: dict,
        bitrates: dict[str, list],
        video_name: str,
        output_path: Path,
        comparison_metric: ComparisonMetric,
        bitrate_title: str = "Bitrate (Kbps)",
    ):
    """
    Generates a Rate-Distortion curve for a single video, comparing multiple methods.

    Args:
        video_data (dict): A dictionary where keys are method names and values
                           are lists of S-PSNR values corresponding to the bitrates.
        bitrates (list): A list of bitrate values in Kbps.
        video_name (str): The name of the video for the graph title.
        output_path (str): The filename to save the graph.
    """

    plt.figure(figsize=(10, 6))

    # Plot each method's data
    for method, quality_values in video_data.items():
        if quality_values:
            plt.plot(bitrates[method], quality_values, marker='o', label=method)

    # Add labels, title, and legend
    plt.title(f"Rate-Distortion Curve for {video_name}", fontsize=16)
    plt.xlabel(bitrate_title, fontsize=12)
    plt.ylabel(f"Average {comparison_metric.name}", fontsize=12)
    plt.grid(True)
    plt.legend(title="Method")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"R-D curve for {video_name} saved to {output_path}")

    plt.close()

def save_average_quality(video_data: dict[str, list[float]], bitrates: list, output_path: Path):
    """
    Saves the average quality values to a data table.

    Args:
        video_data (dict): A dictionary where keys are method names and values
                           are lists of quality values corresponding to the bitrates.
        bitrates (list): A list of bitrate values in Kbps.
        output_path (str): The filename to save the data table.

    """
    # Create DataFrame where rows = methods, cols = bitrates
    df = pd.DataFrame(video_data, index=bitrates)

    # Reset index so 'bitrate' becomes its own column
    df.index.name = "Bitrate (Kbps)"
    df.reset_index(inplace=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Average quality saved to {output_path}")

def compute_bd_rate(R_ref, quality_ref, R_test, quality_test, target_max_quality: float = None) -> float:
    """
    Compute Bjøntegaard Delta Rate (BD-Rate) between two RD curves.
    
    Args:
        R_ref: list or np.array, reference bitrates (Kbps)
        quality_ref: list or np.array, reference quality values
        R_test: list or np.array, test method bitrates (Kbps)
        quality_test: list or np.array, test method quality values
        target_max_quality: Optional float to cap the integration range at a specific quality value (e.g., 40 dB for PSNR). If None, uses the maximum common quality.

    Returns:
        BD-rate percentage (negative means test method is better)
    """
    # 1. Sort by quality and convert to log space for bitrates
    R_ref, quality_ref = np.array(R_ref), np.array(quality_ref)
    R_test, quality_test = np.array(R_test), np.array(quality_test)

    # Sort data by quality
    idx_ref = np.argsort(quality_ref)
    quality_ref, R_ref = quality_ref[idx_ref], R_ref[idx_ref]
    idx_test = np.argsort(quality_test)
    quality_test, R_test = quality_test[idx_test], R_test[idx_test]

    # Convert bitrates to log space
    log_R_ref = np.log(R_ref)
    log_R_test = np.log(R_test)

    # 2. Define the common integration range (intersection of quality ranges)
    quality_min = max(quality_ref[0], quality_test[0])
    # End at the target quality cap, unless the data doesn't go that high
    quality_max = min(target_max_quality, min(quality_ref[-1], quality_test[-1])) if target_max_quality else min(quality_ref[-1], quality_test[-1])

    # 3. Handle non-overlapping curves
    if quality_min >= quality_max:
        # Use the bitrate ratio at the closest quality point as an approximation
        if quality_ref[0] > quality_test[-1]:
            # R_ref starts above R_test range
            return 100 * (R_test[-1] - R_ref[0]) / R_ref[0]
        else:
            # R_test starts above R_ref range
            return 100 * (R_test[0] - R_ref[-1]) / R_ref[-1]

    interp_ref = interpolate.interp1d(quality_ref, log_R_ref, kind='linear')
    interp_test = interpolate.interp1d(quality_test, log_R_test, kind='linear')

    # Integrate the difference between curves over the reference quality range
    integrand = lambda q: interp_test(q) - interp_ref(q)
    area_diff, _ = quad(
        integrand,
        quality_min,
        quality_max,
        limit=200,
        epsabs=1e-6,  # absolute tolerance
        epsrel=1e-4   # relative tolerance
    )

    # Compute BD-rate percentage
    bd_rate = (np.exp(area_diff / (quality_max - quality_min)) - 1) * 100

    return bd_rate

def compute_bd_rate_full_match(
        bitrate_ref,
        quality_ref,
        bitrate_test,
        quality_test,
        min_quality_range: float = None,
        target_max_quality: float = None,
        visualize: bool = False
    ) -> tuple[float, bool]:
    """
    Compute Bjøntegaard Delta Rate (BD-Rate) between two RD curves over the full match.

    Args:
        bitrate_ref: list or np.array, reference bitrates (Kbps)
        quality_ref: list or np.array, reference quality values
        bitrate_test: list or np.array, test method bitrates (Kbps)
        quality_test: list or np.array, test method quality values
        min_quality_range: Optional float specifying the minimum required quality range for a stable BD-rate measurement. If the actual range is smaller, the function will return an instability flag.
        target_max_quality: Optional float to cap the integration range at a specific quality value (e.g., 40 dB for PSNR). If None, uses the maximum common quality.
        visualize: If True, generates a plot of the RD curves and highlights the integration area.

    Returns:
        A tuple containing:
            - BD-rate percentage (negative means test method is better)
            - A boolean indicating whether the measurement is considered stable based on the quality range.
    """
    # Convert to numpy and sort by quality
    idx_ref = np.argsort(quality_ref)
    br_ref, q_ref = np.array(bitrate_ref)[idx_ref], np.array(quality_ref)[idx_ref]
    
    idx_test = np.argsort(quality_test)
    br_test, q_test = np.array(bitrate_test)[idx_test], np.array(quality_test)[idx_test]

    # Convert bitrates to log space
    log_br_ref = np.log(br_ref)
    log_br_test = np.log(br_test)

    # Determine Integration Range
    # We start where both curves begin
    q_min = max(q_ref[0], q_test[0])
    
    # End at the target quality cap, unless the data doesn't go that high
    q_max = min(target_max_quality, max(q_ref[-1], q_test[-1])) if target_max_quality else max(q_ref[-1], q_test[-1])

    interp_ref = interpolate.interp1d(q_ref, log_br_ref, kind='linear', fill_value="extrapolate")
    interp_test = interpolate.interp1d(q_test, log_br_test, kind='linear', fill_value="extrapolate")

    # Integrate the difference between curves over the reference PSNR range
    integrand = lambda psnr: interp_test(psnr) - interp_ref(psnr)
    area_diff, _ = quad(
        integrand,
        q_min,
        q_max,
        limit=200,
        epsabs=1e-6,  # absolute tolerance
        epsrel=1e-4   # relative tolerance
    )

    if visualize:
        plt.close()

        q_axis = np.linspace(min(q_ref[0], q_test[0]), max(q_ref[-1], q_test[-1]), 100)
        
        plt.figure(figsize=(8, 5))
        
        # Plot actual data points
        plt.scatter(q_ref, log_br_ref, color='blue', label='Ref Points')
        plt.scatter(q_test, log_br_test, color='red', label='Test Points')
        
        # Plot the interpolated curves
        plt.plot(q_axis, interp_ref(q_axis), 'blue', linestyle='--', alpha=0.5, label='Ref Curve')
        plt.plot(q_axis, interp_test(q_axis), 'red', linestyle='--', alpha=0.5, label='Test Curve')
        
        # Highlight the Integration Area (The "BD-Rate Zone")
        plt.axvspan(q_min, q_max, color='gray', alpha=0.2, label='Integration Range')
        
        plt.xlabel('Quality (PSNR)')
        plt.ylabel('Log(Bitrate)')
        plt.title(f'RD-Curve Interpolation (p_max: {q_max:.2f})')
        plt.legend()
        plt.grid(True)

        plt.show()
        plt.close()

    # Convert to %
    bd_rate = (np.exp(area_diff / (q_max - q_min)) - 1) * 100

    # Return whether the measurement may be unstable
    if min_quality_range is None or q_max - q_min >= min_quality_range:
        is_stable = True
    else:
        is_stable = False

    return bd_rate, is_stable

def get_exact_bitrate(video_path: Path) -> float:
    """
    Uses ffprobe to extract the exact bitrate of a video in bits per second.

    Args:
        video_path: The path to the video.

    Returns:
        bitrate in kbps (float).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", 
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        bitrate = result.stdout.strip()
        return float(bitrate) / 1000.0 if bitrate else -1  # convert to kbps
    except Exception as e:
        print(f"Could not read bitrate for {video_path}: {e}")
        return -1

def matches_categories(group_name: str, categories_to_match: list, match_all: bool=False):
    """
    Helper function to see if a group matches a set of categories.

    Args:
        group_name: The name of the group.
        categories_to_match: The list of categories to match.
        match_all: True if all categories must be in group name, false if only need one.

    Returns:
        True if the group name matches the appropriate category/categories, False otherwise. 
    """

    if match_all:
        all_categories_in = True
        for category in categories_to_match:
            if not category in group_name:
                all_categories_in = False

        return all_categories_in
    else:
        for category in categories_to_match:
            if category in group_name:
                return True
        
        return False

#
# endregion Analysis
#