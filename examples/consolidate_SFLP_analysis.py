from PIL import Image
from pathlib import Path
from collections import defaultdict

from SFLP_toolkit.utilities.evaluation_utils import (
  ComparisonMetric,
  VideoEncoder,
  extract_average_quality,
  save_average_quality,
  generate_rd_curve,
  get_exact_bitrate,
  matches_categories,
  WeightingType,
  SFLPArrangementType,
)

from config import *


sample_type = Image.Resampling.BILINEAR
output_video_container = VideoEncoder.H265
weighting_type = WeightingType.VIEWING_BEHAVIOR
SFLP_arrangement_type = SFLPArrangementType.STANDARD
comparison_metric = ComparisonMetric.SPSNR
preset = "Preset_Slow"
pixels = "21.6M_Pixels"

temporal_levels = [low_temporal_name, medium_temporal_name, high_temporal_name]
spatial_levels = [low_spatial_name, medium_spatial_name, high_spatial_name]

average_names = [f"{t} Average" for t in temporal_levels] + [f"{s} Average" for s in spatial_levels]
average_names.append(macro_average_name)

# Whether a video must contain all categories, or just one, in order to be a part of the averaging group.
match_all = False


# Input directory
input_dir = Path(f"../output/{comparison_metric.name}/SFLP_Comparison/Pixel_Comparison")
video_root_dir = Path("D:/GitHub_Projects/VR_Encoding_Dataset/Output")

# Output directory
output_dir = Path(f"../output/{comparison_metric.name}")
output_dir.mkdir(parents=True, exist_ok=True)

SFLP_num_pixels_list = [
    8000001,
    14000001,
    18000001,
    21660001,
    24000001,
    26000001,
    32000001,
]

method_names = {}

for SFLP_num_pixels in SFLP_num_pixels_list:
    method_names[SFLP_num_pixels] = f"SFLP {SFLP_num_pixels} S-PSNR"

# Input Videos
input_video_names = [
    # 8K Res

    "original_ERP-CableCar",
    "original_ERP-Fireworks",
    "original_ERP-BridgeBikeRide",
    "original_ERP-DroneInDesert",

    "AcademicBuilding",
    "AdministrationBuilding",
    "kaixuanmenye_pano",
    "lanqiuchang_pano",
    "Library",
    "runner_pano",
    "shuangziqiao_pano",
    "SiyuanGate",
    "SouthGate",
    "StudyRoom",
]

video_categories = VIDEO_CATEGORIES

# Bitrates
bitrates = [
    5000,
    10000,
    25000,
    40000,
    60000,
]

if __name__ == "__main__":
    
    quality_per_video = {}

    for average_name in average_names:
        quality_per_video[average_name] = {}

    for input_video_name in input_video_names:
        quality_per_video[input_video_name] = {}

    for bitrate in bitrates:

        average_quality = {}
        for average_name in average_names:
            average_quality[average_name] = {}
            average_quality[average_name]["num_vids"] = 0

        for input_video_name in input_video_names:
            quality_info = f"{SFLP_arrangement_type.name}-{output_video_container.name}-Target_{bitrate}k-{preset}-{weighting_type.name}"
            quality_csv_name = f"{comparison_metric.name}-{input_video_name}-{quality_info}.csv"
            quality_path = input_dir / quality_csv_name

            averages = extract_average_quality(quality_path)
            quality_per_video[input_video_name][bitrate] = averages

            average_quality[macro_average_name]["num_vids"] += 1
            for average_name in average_names:
                    if matches_categories(average_name, video_categories[input_video_name], match_all=match_all):
                        average_quality[average_name]["num_vids"] += 1

            for method, avg_val in averages.items():
                if method not in average_quality[macro_average_name]:
                    for average_name in average_names:
                        average_quality[average_name][method] = 0

                average_quality[macro_average_name][method] += avg_val
                for average_name in average_names:
                    if matches_categories(average_name, video_categories[input_video_name], match_all=match_all):
                        average_quality[average_name][method] += avg_val
        
        for average_name in average_names:
            for method in average_quality[average_name]:
                if method != "num_vids" and average_quality[average_name]["num_vids"] != 0:
                    average_quality[average_name][method] = average_quality[average_name][method] / average_quality[average_name]["num_vids"]
            
            # Remove the num_vids value after using for normalization.
            average_quality[average_name].pop("num_vids", None)

            quality_per_video[average_name][bitrate] = average_quality[average_name]


    # Restructure the data so that each encoding method holds a list of bitrates.
    bitrates = sorted(list(next(iter(quality_per_video.values())).keys()))
    restructured_data = defaultdict(lambda: defaultdict(list))

    for video_name, bitrates_data in quality_per_video.items():
        if not ("Average" in video_name):
            for bitrate in sorted(bitrates_data.keys()):
                for method, value in bitrates_data[bitrate].items():
                    restructured_data[video_name][method].append(value)

    bitrate_savings = {}

    for average_name in average_names:
        bitrate_savings[average_name] = {}
        for SFLP_num_pixels in SFLP_num_pixels_list:
            bitrate_savings[average_name][method_names[SFLP_num_pixels]] = 0

    videos_calculated = {}
    for average_name in average_names:
        videos_calculated[average_name] =  {}
        for SFLP_num_pixels in SFLP_num_pixels_list:
            videos_calculated[average_name][method_names[SFLP_num_pixels]] = 0

    # Generate the RD curves.
    encoding_info_no_bitrate = f"{SFLP_arrangement_type.name}-{output_video_container.name}-{preset}"
    quality_info = f"{encoding_info_no_bitrate}-{pixels}-{weighting_type.name}"
    for video_name, video_data in restructured_data.items():
        
        # Extract the exact bitrates for each encoding type for this video.
        exact_bitrates = {}

        
        for SFLP_num_pixels in SFLP_num_pixels_list:
            exact_SFLP_bitrates = []
            for bitrate in bitrates:
                encoding_info = f"{output_video_container.name}-{preset}-SFLP_Comparison"
                video_dir = video_root_dir / f"{encoding_info}"
                
                SFLP_video_name = f"SFLP-360_no_tiling-{video_name}-{SFLP_num_pixels}-SFLP_arrangement_{SFLP_arrangement_type.name}-bitrate_{bitrate}.mp4"
                SFLP_video_path = video_dir / SFLP_video_name

                exact_SFLP_bitrates.append(get_exact_bitrate(SFLP_video_path))
            
            exact_bitrates[method_names[SFLP_num_pixels]] = exact_SFLP_bitrates


        graph_output_filename = f"rd_curve-{video_name}-{quality_info}.png"
        graph_output_file_path = output_dir / graph_output_filename
        modified_video_name = video_name.split("_pano")[0]
        if "original_ERP-" in modified_video_name:
            modified_video_name = video_name.split("original_ERP-")[1]
        generate_rd_curve(video_data, exact_bitrates, modified_video_name, graph_output_file_path, comparison_metric=comparison_metric)

        table_output_filename = f"average_{comparison_metric.name}-{video_name}-{quality_info}.csv"
        table_output_file_path = output_dir / table_output_filename
        save_average_quality(video_data=video_data, bitrates=bitrates, output_path=table_output_file_path)

        print()




                    