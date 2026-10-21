from pathlib import Path
from PIL import Image
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


# Sampling and Container Parameters
sample_type = Image.Resampling.BILINEAR
weighting_type = WeightingType.EQUAL
comparison_metric = ComparisonMetric.SPSNR
output_video_container = VideoEncoder.H265
SFLP_arrangement_type = SFLPArrangementType.STANDARD
preset = "Slow"

# Input directories
layout_dir = Path("../output")
ground_truth_dir = Path("D:\\GitHub_Projects\\VR_Encoding_Dataset\\input")
dataset_dir = Path("D:\\GitHub_Projects\\VR_Encoding_Dataset\\Output")

# Output directory
output_dir = Path(f"../output/{comparison_metric.name}")
output_dir.mkdir(parents=True, exist_ok=True)


# Number of Points on Sphere to Sample for S-PSNR calculations
num_evaluation_points = 100001

# Metadata
ground_truth_metadata = ProjectionMetadata(EncodingProjection.ERP)

SFLP_num_pixels_list = [
    8000001,
    14000001,
    18000001,
    21660001,
    24000001,
    26000001,
    32000001,
]

# Hint: use generate_SFLP_pixel_arrangement() to generate the pixel arrangement to help derive these values.
optimal_index_hops = [
    [2584, 4181],
    [4181, 6765],
    [4181, 6765],
    [4181, 6765],
    [4181, 6765],
    [4181, 6765],
    [6765, 10946],
] # Uses the zig and zag column generator vector index hops:
# zig and zag -> arr[0][0] - arr[0][1] and arr[0][1] - arr[0][2], zig < zag.
first_element_SFLP_index = [
    7996160,
    13999470,
    17997983,
    21656007,
    23999135,
    25995009,
    31991272,
] # Uses the first element of the first column in the generated pixel arrangement: arr[0][0].

last_element_SFLP_index = [
    2088,
    2685,
    3083,
    1242,
    3680,
    3879,
    7060,
] # Uses the last element of the first column in the generated pixel arrangement: arr[0][col_length - 1].

SFLP_metadata_list = []

for index in range(len(SFLP_num_pixels_list)):
    optimal_index_hops_metadata = [(k, generator_vec_geodesic_distance(k, SFLP_num_pixels_list[index])) for k in optimal_index_hops[index]]
    optimal_index_hops_metadata = [(k, val[0], val[1], val[2]) for k, val in optimal_index_hops_metadata]

    SFLP_metadata = SFLPMetadata(
        num_pixels=SFLP_num_pixels_list[index],
        arrangement_type=SFLP_arrangement_type,
        optimal_index_hops=optimal_index_hops_metadata,
        first_element_SFLP_index=first_element_SFLP_index[index],
        last_element_SFLP_index=last_element_SFLP_index[index],
    )

    SFLP_metadata_list.append(SFLP_metadata)


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

bitrates = [
    1000,
    3000,
    5000,
    10000,
    25000,
    40000,
    60000,
]

if __name__ == "__main__":
    
    for input_video_name in input_video_names:

        for bitrate in bitrates:
            encoding_info = f"{SFLP_arrangement_type.name}-{output_video_container.name}-Target_{bitrate}k-Preset_{preset}"
            encoding_info_dir = f"{output_video_container.name}-Target_{bitrate}k-Preset_{preset}"
            compared_dir = Path(dataset_dir, f"{encoding_info_dir}")
            
            SFLP_QUALITY_DICT = {}

            for SFLP_metadata in SFLP_metadata_list:
                SFLP_num_pixels = SFLP_metadata.num_pixels

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
                )

                SFLP_QUALITY_DICT[SFLP_num_pixels] = SFLP_QUALITY

                
                
            # Write to CSV
            quality_csv_name = f"{comparison_metric.name}-{input_video_name}-{encoding_info}-{weighting_type.name}.csv"
            quality_analysis_path = output_dir / quality_csv_name
            with open(quality_analysis_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                header = ["Frame"] + [f"SFLP {px} {comparison_metric.name}" for px in SFLP_num_pixels_list]
                writer.writerow(header)

                first_key = list(SFLP_QUALITY_DICT.keys())[0]
                first_quality_list = SFLP_QUALITY_DICT[first_key]


                for frame in range(1, len(first_quality_list) + 1):
                    row = [frame]
                    for px in SFLP_num_pixels_list:
                        row.append(SFLP_QUALITY_DICT[px].get(frame, ""))
                    writer.writerow(row)

