from pathlib import Path
from shutil import which
from PIL import Image
import math
import csv

from SFLP_toolkit.utilities.evaluation_utils import (
  evaluate_image_quality,
  generator_vec_geodesic_distance,
  ProjectionMetadata,
  EncodingProjection,
  SFLPMetadata,
  WeightingType,
  SFLPArrangementType,
)


sample_type = Image.Resampling.BILINEAR
weighting_type = WeightingType.VIEWING_BEHAVIOR

# Input directories
ground_truth_dir = Path("..\\input")
compared_dir = Path("..\\output\\images")


# Number of Points on Sphere to Sample for S-PSNR calculations
num_evaluation_points = 100001

# Metadata
ground_truth_metadata = ProjectionMetadata(EncodingProjection.ERP)

SFLP_num_pixels = 18000001
SFLP_arrangement_type = SFLPArrangementType.STANDARD
# Hint: use generate_SFLP_pixel_arrangement() to generate the pixel arrangement to help derive these values.
optimal_index_hops = [2584, 4181] # The optimal index hops for the SFLP arrangement are derived from the Fibonacci sequence,
                                    # which is used to determine the pixel arrangement in the SFLP projection.
padding_size = 2
first_element_SFLP_index = -1 # Uses the first element of the first column in the generated pixel arrangement: arr[0][0].
last_element_SFLP_index = 0 # Uses the last element of the first column in the generated pixel arrangement: arr[0][col_length - 1].

optimal_index_hops_metadata = [(k, generator_vec_geodesic_distance(k, SFLP_num_pixels)) for k in optimal_index_hops]
optimal_index_hops_metadata = [(k, val[0], val[1], val[2]) for k, val in optimal_index_hops_metadata]

SFLP_metadata = SFLPMetadata(
    num_pixels=SFLP_num_pixels,
    arrangement_type=SFLP_arrangement_type,
    optimal_index_hops=optimal_index_hops_metadata,
    first_element_SFLP_index=first_element_SFLP_index,
    last_element_SFLP_index=last_element_SFLP_index,
    padding_size=padding_size,
)

EAP_num_pixels = 18000000
EAP_metadata = ProjectionMetadata(EncodingProjection.EAP)

ERP_num_pixels = 18000000
ERP_metadata = ProjectionMetadata(EncodingProjection.ERP)

CMP_num_pixels = 17998944
CMP_metadata = ProjectionMetadata(EncodingProjection.CMP)

EAC_num_pixels = 17998944
EAC_metadata = ProjectionMetadata(EncodingProjection.EAC)

# Input Images
input_image_names = [
    "ERP-CableCar-snapshot",
]

if __name__ == "__main__":
    
    for input_image_name in input_image_names:

        ground_truth_image_name = input_image_name + ".jpg"
        ground_truth_image_path = ground_truth_dir / ground_truth_image_name


        image_name = f"Reencoded-{input_image_name}-{ERP_num_pixels}.jpeg"
        compared_image_path = compared_dir / image_name

        ERP_SPSNR = evaluate_image_quality(
            num_evaluation_points=num_evaluation_points,
            ground_truth_image_path=ground_truth_image_path,
            ground_truth_metadata=ground_truth_metadata,
            compared_image_path=compared_image_path,
            compared_metadata=ERP_metadata,
            sample_type=sample_type,
            weighting_type=weighting_type,
        )
        
        print(f"The reference ERP image has S-PSNR of: {ERP_SPSNR}")


        image_name = f"EAP-{input_image_name}-{EAP_num_pixels}.jpeg"
        compared_image_path = compared_dir / image_name

        EAP_SPSNR = evaluate_image_quality(
            num_evaluation_points=num_evaluation_points,
            ground_truth_image_path=ground_truth_image_path,
            ground_truth_metadata=ground_truth_metadata,
            compared_image_path=compared_image_path,
            compared_metadata=EAP_metadata,
            sample_type=sample_type,
            weighting_type=weighting_type,
        )
        
        print(f"The reference EAP image has S-PSNR of: {EAP_SPSNR}")

        image_name = f"CMP-{input_image_name}-{CMP_num_pixels}.jpeg"
        compared_image_path = compared_dir / image_name

        CMP_SPSNR = evaluate_image_quality(
            num_evaluation_points=num_evaluation_points,
            ground_truth_image_path=ground_truth_image_path,
            ground_truth_metadata=ground_truth_metadata,
            compared_image_path=compared_image_path,
            compared_metadata=CMP_metadata,
            sample_type=sample_type,
            weighting_type=weighting_type,
        )
        
        print(f"The reference CMP image has S-PSNR of: {CMP_SPSNR}")

        image_name = f"EAC-{input_image_name}-{EAC_num_pixels}.jpeg"
        compared_image_path = compared_dir / image_name

        EAC_SPSNR = evaluate_image_quality(
            num_evaluation_points=num_evaluation_points,
            ground_truth_image_path=ground_truth_image_path,
            ground_truth_metadata=ground_truth_metadata,
            compared_image_path=compared_image_path,
            compared_metadata=EAC_metadata,
            sample_type=sample_type,
            weighting_type=weighting_type,
        )
        
        print(f"The reference EAC image has S-PSNR of: {EAC_SPSNR}")


        image_name = f"SFLP-{input_image_name}-{SFLP_num_pixels}-{SFLP_arrangement_type.name}.jpeg"
        compared_image_path = compared_dir / image_name

        SFLP_SPSNR = evaluate_image_quality(
            num_evaluation_points=num_evaluation_points,
            ground_truth_image_path=ground_truth_image_path,
            ground_truth_metadata=ground_truth_metadata,
            compared_image_path=compared_image_path,
            compared_metadata=SFLP_metadata,
            sample_type=sample_type,
            weighting_type=weighting_type,
        )
        
        print(f"The reference SFLP image has S-PSNR of: {SFLP_SPSNR}")