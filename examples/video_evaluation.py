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
  ComparisonMetric,
  SFLPArrangementType,
)


sample_type = Image.Resampling.BILINEAR
weighting_type = WeightingType.VIEWING_BEHAVIOR
output_video_container = VideoEncoder.H265
comparison_metric = ComparisonMetric.SPSNR
presets_and_pixels = "Preset_Slow-18M_Pixels"

debug_msg_and_viz = False

# Input directories
ground_truth_dir = Path("D:\\GitHub_Projects\\VR_Encoding_Dataset\\Input") # Change this directory to where your high-resolution videos reside
reencoded_dir = Path("D:\\GitHub_Projects\\VR_Encoding_Dataset\\Output") # Change this directory where your compressed videos reside

# Output directory
output_dir = Path(f"..\\output\\{comparison_metric.name}\\SJTU\\weighting-{weighting_type.name}")
output_dir.mkdir(parents=True, exist_ok=True)

# Input Videos
input_video_names = [
    # 8K Res

    # SJTU
    "AcademicBuilding",
    "AdministrationBuilding",
    "EastGate",
    "kaixuanmenye_pano",
    "lanqiuchang_pano",
    "Library",
    "runner_pano",
    "shuangziqiao_pano",
    "SiyuanGate",
    "SouthGate",
    "StudyRoom",
    "Sward",
]

# Bitrates
bitrates = [
    3000,
    5000,
    10000,
    25000,
    40000,
]

# Number of Points on Sphere to Sample for S-PSNR calculations
num_evaluation_points = 100001

# Metadata
ground_truth_metadata = ProjectionMetadata(EncodingProjection.ERP)

SFLP_num_pixels = 18000001
SFLP_arrangement_type = SFLPArrangementType.STANDARD
padding_size = 2
# Hint: use generate_SFLP_pixel_arrangement() to generate the pixel arrangement to help derive these values.
optimal_index_hops = [2584, 4181] # Uses the zig and zag column generator vector index hops:
                                    # zig and zag -> arr[0][0] - arr[0][1] and arr[0][1] - arr[0][2], zig < zag.
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


CubeMap_num_pixels = 17998944
face_size = int(math.sqrt(CubeMap_num_pixels // 6))
CMP_metadata = ProjectionMetadata(EncodingProjection.CMP)
EAC_metadata =  ProjectionMetadata(EncodingProjection.EAC)

ERP_reencoded_output_size = (6000, 3000)
ERP_num_pixels = ERP_reencoded_output_size[0] * ERP_reencoded_output_size[1]
ERP_metadata = ProjectionMetadata(EncodingProjection.ERP)
EAP_metadata = ProjectionMetadata(EncodingProjection.EAP)

if __name__ == "__main__":
    
    for input_video_name in input_video_names:
        for bitrate in bitrates:

            print(f"Processing input video: {input_video_name} with bitrate: {bitrate}")

            encoding_info = f"{SFLP_arrangement_type.name}-{output_video_container.name}-Target_{bitrate}k-{presets_and_pixels}-{weighting_type.name}"
            compared_subfolder = f"{output_video_container.name}-Target_{bitrate}k-Preset_Slow"
            compared_dir = Path(reencoded_dir, f"{compared_subfolder}")

            ground_truth_video_name = input_video_name + ".mp4"
            ground_truth_video_path = ground_truth_dir / ground_truth_video_name

            SFLP_video_name = f"SFLP-360_no_tiling-{input_video_name}-{SFLP_num_pixels}-SFLP_arrangement_{SFLP_arrangement_type.name}-bitrate_{bitrate}.mp4"
            SFLP_compared_video_path = compared_dir / SFLP_video_name

            SFLP_QUALITY = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=SFLP_compared_video_path,
                compared_metadata=SFLP_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
                comparison_metric=comparison_metric,
                print_errors=debug_msg_and_viz,
                visualize_error=debug_msg_and_viz,
            )

            ERP_video_name = f"Reencoded_ERP-{input_video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            ERP_video_path = compared_dir / ERP_video_name

            ERP_QUALITY = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=ERP_video_path,
                compared_metadata=ERP_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
                comparison_metric=comparison_metric,
                print_errors=debug_msg_and_viz,
                visualize_error=debug_msg_and_viz,
            )

            EAP_video_name = f"EAP-360_no_tiling-{input_video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            EAP_video_path = compared_dir / EAP_video_name

            EAP_QUALITY = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=EAP_video_path,
                compared_metadata=EAP_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
                comparison_metric=comparison_metric,
                print_errors=debug_msg_and_viz,
                visualize_error=debug_msg_and_viz,
            )

            CMP_video_name = f"CMP-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            CMP_video_path = compared_dir / CMP_video_name

            CMP_QUALITY = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=CMP_video_path,
                compared_metadata=CMP_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
                comparison_metric=comparison_metric,
                print_errors=debug_msg_and_viz,
                visualize_error=debug_msg_and_viz,
            )

            EAC_video_name = f"EAC-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            EAC_video_path = compared_dir / EAC_video_name

            EAC_QUALITY = evaluate_video_quality(
                num_evaluation_points=num_evaluation_points,
                ground_truth_video_path=ground_truth_video_path,
                ground_truth_metadata=ground_truth_metadata,
                compared_video_path=EAC_video_path,
                compared_metadata=EAC_metadata,
                sample_type=sample_type,
                weighting_type=weighting_type,
                comparison_metric=comparison_metric,
                print_errors=debug_msg_and_viz,
                visualize_error=debug_msg_and_viz,
            )
            
            # Write to CSV
            quality_csv_name = f"{comparison_metric.name}-{input_video_name}-{encoding_info}.csv"
            quality_analysis_path = output_dir / quality_csv_name
            with open(quality_analysis_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Frame", f"ERP ({ERP_num_pixels}) S-PSNR", f"EAP ({ERP_num_pixels}) S-PSNR", f"CMP ({CubeMap_num_pixels}) S-PSNR", f"EAC ({CubeMap_num_pixels}) S-PSNR", f"SFLP ({SFLP_num_pixels}) S-PSNR"])  # Header
                for frame in SFLP_QUALITY.keys():
                    ERP_quality_val = ERP_QUALITY[frame]
                    EAP_quality_val = EAP_QUALITY[frame]
                    CMP_quality_val = CMP_QUALITY[frame]
                    EAC_quality_val = EAC_QUALITY[frame]
                    SFLP_quality_val = SFLP_QUALITY[frame]

                    writer.writerow([frame, ERP_quality_val, EAP_quality_val, CMP_quality_val, EAC_quality_val, SFLP_quality_val])
