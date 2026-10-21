"""Spherical Tiling and Heatmap Visualization Package.

This package provides tools for generating and analyzing area of spherical tilings, as well as generating heatmaps for visualization on 360-degree video viewer data.
It includes functionality for data processing, spherical tile center and boundary calculations, heatmap calculations, and visualization.
"""

from .data_types import Point, GeographicPoint, Vector, ValidationError, convert_vectors_to_coordinates
from SFLP_toolkit.utilities.pixel_arrangement_utils import *
from SFLP_toolkit.utilities.file_conversion_utils import *
from SFLP_toolkit.utilities.evaluation_utils import *

__version__ = "1.0.0"
__author__ = "Chitsein Htun"
__all__ = [
    'Point',
    'GeographicPoint',
    'Vector',
    'ValidationError',
    'convert_vectors_to_coordinates',

    # Pixel Arrangement utilities
    'golden_ratio',
    'get_SFL_vector',
    'get_projected_point',
    'get_projected_SFL_point',
    'get_projected_generator_vector',
    'generator_vec_geodesic_distance',
    
    'find_optimal_generator_vectors',
    'get_tile_index',
    'get_row_col_index',

    'geographic_to_vector_gpu',
    'generate_SFL_projected_gpu',
    'get_SFL_projected_gpu',
    'get_SFL_geographic_points',

    'generate_SFLP_pixel_arrangement',
    'generate_CubeMap_pixel_arrangement',

    # File Conversion utilities
    'get_CubeMap_geographic_points',
    'get_EAP_geographic_points',

    'geographic_to_ERP_index',
    'geographic_to_ERP_index_float',
    'map_vectors_to_SFLP_indices',

    'sample_ERP',
    'sample_CMP',
    'sample_EAC',
    'sample_EAP',
    'sample_SFLP',

    'ERP_to_SFLP_image_conversion',
    'ERP_to_CMP_image_conversion',
    'ERP_to_EAC_image_conversion',
    'ERP_to_EAP_image_conversion',
    'CMP_to_SFLP_image_conversion',
    'EAC_to_SFLP_image_conversion',
    'EAP_to_SFLP_image_conversion',
    'SFLP_to_CMP_image_conversion',
    'reencode_ERP_image',

    'ERP_to_SFLP_video_conversion',
    'ERP_to_CMP_video_conversion',
    'ERP_to_EAC_video_conversion',
    'ERP_to_EAP_video_conversion',
    'reencode_ERP_video',

    # Evaluation utilites
    'calculate_spherical_PSNR',
    'compute_PSNR',
    'evaluate_video_quality',
    'evaluate_image_quality',

    'extract_average_quality',
    'generate_rd_curve',
    'save_average_quality',
    'compute_bd_rate',
    'get_exact_bitrate',
]
