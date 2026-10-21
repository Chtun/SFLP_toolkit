"""Utilities module for spherical tiling and heatmap visualizations.

This module provides utility functions for data processing, spherical tile center and boundary calculations, heatmap calculations,
and visualization.
"""

from .pixel_arrangement_utils import (
    golden_ratio,
    get_SFL_vector,
    get_projected_point,
    get_projected_SFL_point,
    get_projected_generator_vector,
    generator_vec_geodesic_distance,

    find_optimal_generator_vectors,

    get_tile_index,
    get_row_col_index,

    geographic_to_vector_gpu,
    generate_SFL_projected_gpu,
    get_SFL_projected_gpu,
    get_SFL_geographic_points,

    generate_SFL_projected_gpu,
    generate_SFLP_pixel_arrangement,
    generate_CubeMap_pixel_arrangement,

)

from .file_conversion_utils import (
    get_CubeMap_geographic_points,
    get_EAP_geographic_points,

    geographic_to_ERP_index,
    geographic_to_ERP_index_float,
    map_vectors_to_SFLP_indices,

    sample_ERP,
    sample_CMP,
    sample_EAC,
    sample_EAP,
    sample_SFLP,

    ERP_to_SFLP_image_conversion,
    ERP_to_CMP_image_conversion,
    ERP_to_EAC_image_conversion,
    ERP_to_EAP_image_conversion,
    CMP_to_SFLP_image_conversion,
    EAC_to_SFLP_image_conversion,
    EAP_to_SFLP_image_conversion,
    SFLP_to_CMP_image_conversion,
    reencode_ERP_image,

    ERP_to_SFLP_video_conversion,
    ERP_to_CMP_video_conversion,
    ERP_to_EAC_video_conversion,
    ERP_to_EAP_video_conversion,
    reencode_ERP_video,
)

from .evaluation_utils import (
    calculate_spherical_PSNR,
    compute_PSNR,
    evaluate_video_quality,
    evaluate_image_quality,

    extract_average_quality,
    generate_rd_curve,
    save_average_quality,
    compute_bd_rate,
    get_exact_bitrate,
)

__all__ = [
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