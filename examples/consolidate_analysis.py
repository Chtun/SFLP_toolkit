import pandas as pd
from PIL import Image
from pathlib import Path
from collections import defaultdict
import numpy as np

from SFLP_toolkit.utilities.evaluation_utils import (
  ComparisonMetric,
  VideoEncoder,
  extract_average_quality,
  save_average_quality,
  generate_rd_curve,
  compute_bd_rate,
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
pixels = "18M_Pixels"

temporal_levels = [low_temporal_name, medium_temporal_name, high_temporal_name]
spatial_levels = [low_spatial_name, medium_spatial_name, high_spatial_name]

average_names = [f"{t} Average" for t in temporal_levels] + [f"{s} Average" for s in spatial_levels]
average_names.append(macro_average_name)

# Whether a video must contain all categories, or just one, in order to be a part of the averaging group.
match_all = False


# Input directory
input_dir = Path(f"../output/{comparison_metric.name}/SJTU/weighting-{weighting_type.name}")
reencoded_dir = Path("D:/GitHub_Projects/VR_Encoding_Dataset/Output")

# Output directory
output_dir = Path(f"../output/{comparison_metric.name}/Averages")
output_dir.mkdir(parents=True, exist_ok=True)

SFLP_num_pixels = 18000001
CubeMap_num_pixels = 17998944
ERP_num_pixels = 18000000

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

video_categories = VIDEO_CATEGORIES

# Bitrates
bitrates = [
    3000,
    5000,
    10000,
    25000,
    40000,
]

method_names = {
    "ERP": f"ERP ({ERP_num_pixels}) S-PSNR",
    "EAP": f"EAP ({ERP_num_pixels}) S-PSNR",
    "CMP": f"CMP ({CubeMap_num_pixels}) S-PSNR",
    "EAC": f"EAC ({CubeMap_num_pixels}) S-PSNR",
    "SFLP": f"SFLP ({SFLP_num_pixels}) S-PSNR",
    
    "ERP vs. SFLP": f"SFLP ({SFLP_num_pixels}) bitrate saving against ERP ({ERP_num_pixels})",
    "EAP vs. SFLP": f"SFLP ({SFLP_num_pixels}) bitrate saving against EAP ({ERP_num_pixels})",
    "CMP vs. SFLP": f"SFLP ({SFLP_num_pixels}) bitrate saving against CMP ({CubeMap_num_pixels})",
    "EAC vs. SFLP": f"SFLP ({SFLP_num_pixels}) bitrate saving against EAC ({CubeMap_num_pixels})",
    
    "ERP vs. EAP": f"EAP ({ERP_num_pixels}) bitrate saving against ERP ({ERP_num_pixels})",
    "ERP vs. CMP": f"CMP ({CubeMap_num_pixels}) bitrate saving against ERP ({ERP_num_pixels})",
    "ERP vs. EAC": f"EAC ({CubeMap_num_pixels}) bitrate saving against ERP ({ERP_num_pixels})",
}

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
            quality_info = f"{SFLP_arrangement_type.name}-{output_video_container.name}-Target_{bitrate}k-{preset}-{pixels}-{weighting_type.name}"
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
        bitrate_savings[average_name]
        bitrate_savings[average_name][method_names["ERP vs. SFLP"]] = 0
        bitrate_savings[average_name][method_names["EAP vs. SFLP"]] = 0
        bitrate_savings[average_name][method_names["CMP vs. SFLP"]] = 0
        bitrate_savings[average_name][method_names["EAC vs. SFLP"]] = 0
        bitrate_savings[average_name][method_names["ERP vs. EAP"]] = 0
        bitrate_savings[average_name][method_names["ERP vs. CMP"]] = 0
        bitrate_savings[average_name][method_names["ERP vs. EAC"]] = 0

    videos_calculated = {}
    for average_name in average_names:
        videos_calculated[average_name] =  {}
        videos_calculated[average_name][method_names["ERP vs. SFLP"]] = 0
        videos_calculated[average_name][method_names["EAP vs. SFLP"]] = 0
        videos_calculated[average_name][method_names["CMP vs. SFLP"]] = 0
        videos_calculated[average_name][method_names["EAC vs. SFLP"]] = 0
        videos_calculated[average_name][method_names["ERP vs. EAP"]] = 0
        videos_calculated[average_name][method_names["ERP vs. CMP"]] = 0
        videos_calculated[average_name][method_names["ERP vs. EAC"]] = 0

    # Generate the RD curves.
    encoding_info_no_bitrate = f"{SFLP_arrangement_type.name}-{output_video_container.name}-{preset}"
    quality_info = f"{encoding_info_no_bitrate}-{pixels}-{weighting_type.name}"
    for video_name, video_data in restructured_data.items():
        
        # Extract the exact bitrates for each encoding type for this video.
        exact_bitrates = {}
        exact_ERP_bitrates = []
        exact_EAP_bitrates = []
        exact_CMP_bitrates = []
        exact_EAC_bitrates = []
        exact_SFLP_bitrates = []

        for bitrate in bitrates:
            encoding_info = f"{output_video_container.name}-Target_{bitrate}k-{preset}"
            video_dir = reencoded_dir / f"{encoding_info}"
            
            ERP_video_name = f"Reencoded_ERP-{video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            ERP_video_path = video_dir / ERP_video_name

            exact_ERP_bitrates.append(get_exact_bitrate(ERP_video_path))

            EAP_video_name = f"EAP-360_no_tiling-{video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            EAP_video_path = video_dir / EAP_video_name

            exact_EAP_bitrates.append(get_exact_bitrate(EAP_video_path))

            CMP_video_name = f"CMP-360_no_tiling-{video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            CMP_video_path = video_dir / CMP_video_name

            exact_CMP_bitrates.append(get_exact_bitrate(CMP_video_path))

            EAC_video_name = f"EAC-360_no_tiling-{video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            EAC_video_path = video_dir / EAC_video_name

            exact_EAC_bitrates.append(get_exact_bitrate(EAC_video_path))

            
            SFLP_video_name = f"SFLP-360_no_tiling-{video_name}-{SFLP_num_pixels}-SFLP_arrangement_{SFLP_arrangement_type.name}-bitrate_{bitrate}.mp4"
            SFLP_video_path = video_dir / SFLP_video_name

            exact_SFLP_bitrates.append(get_exact_bitrate(SFLP_video_path))


        for method in video_data:
            if "ERP" in method:
                exact_bitrates[method] = exact_ERP_bitrates
            elif "EAP" in method:
                exact_bitrates[method] = exact_EAP_bitrates
            elif "CMP" in method:
                exact_bitrates[method] = exact_CMP_bitrates
            elif "EAC" in method:
                exact_bitrates[method] = exact_EAC_bitrates
            elif "SFLP" in method:
                exact_bitrates[method] = exact_SFLP_bitrates
        
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

        ERP_quality = video_data[method_names["ERP"]]
        EAP_quality = video_data[method_names["EAP"]]
        CMP_quality = video_data[method_names["CMP"]]
        EAC_quality = video_data[method_names["EAC"]]
        SFLP_quality = video_data[method_names["SFLP"]]

        # Compute BD-Rate savings
        bd_ERP_vs_SFLP = compute_bd_rate(exact_ERP_bitrates, ERP_quality, exact_SFLP_bitrates, SFLP_quality)
        bd_EAP_vs_SFLP = compute_bd_rate(exact_EAP_bitrates, EAP_quality, exact_SFLP_bitrates, SFLP_quality)
        bd_CMP_vs_SFLP = compute_bd_rate(exact_CMP_bitrates, CMP_quality, exact_SFLP_bitrates, SFLP_quality)
        bd_EAC_vs_SFLP = compute_bd_rate(exact_EAC_bitrates, EAC_quality, exact_SFLP_bitrates, SFLP_quality)

        bd_ERP_vs_EAP = compute_bd_rate(exact_ERP_bitrates, ERP_quality, exact_EAP_bitrates, EAP_quality)
        bd_ERP_vs_CMP = compute_bd_rate(exact_ERP_bitrates, ERP_quality, exact_CMP_bitrates, CMP_quality)
        bd_ERP_vs_EAC = compute_bd_rate(exact_ERP_bitrates, ERP_quality, exact_EAC_bitrates, EAC_quality)


        bitrate_savings[video_name] = {}
        bitrate_savings[video_name][method_names["ERP vs. SFLP"]] = bd_ERP_vs_SFLP
        bitrate_savings[video_name][method_names["EAP vs. SFLP"]] = bd_EAP_vs_SFLP
        bitrate_savings[video_name][method_names["CMP vs. SFLP"]] = bd_CMP_vs_SFLP
        bitrate_savings[video_name][method_names["EAC vs. SFLP"]] = bd_EAC_vs_SFLP

        bitrate_savings[video_name][method_names["ERP vs. EAP"]] = bd_ERP_vs_EAP
        bitrate_savings[video_name][method_names["ERP vs. CMP"]] = bd_ERP_vs_CMP
        bitrate_savings[video_name][method_names["ERP vs. EAC"]] = bd_ERP_vs_EAC
        
        if bd_ERP_vs_SFLP != np.inf:
            bitrate_savings[macro_average_name][method_names["ERP vs. SFLP"]] += bd_ERP_vs_SFLP
            videos_calculated[macro_average_name][method_names["ERP vs. SFLP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["ERP vs. SFLP"]] += bd_ERP_vs_SFLP
                        videos_calculated[average_name][method_names["ERP vs. SFLP"]] += 1
        
        if bd_EAP_vs_SFLP != np.inf:
            bitrate_savings[macro_average_name][method_names["EAP vs. SFLP"]] += bd_EAP_vs_SFLP
            videos_calculated[macro_average_name][method_names["EAP vs. SFLP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["EAP vs. SFLP"]] += bd_EAP_vs_SFLP
                        videos_calculated[average_name][method_names["EAP vs. SFLP"]] += 1

        if bd_CMP_vs_SFLP != np.inf:
            bitrate_savings[macro_average_name][method_names["CMP vs. SFLP"]] += bd_CMP_vs_SFLP
            videos_calculated[macro_average_name][method_names["CMP vs. SFLP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["CMP vs. SFLP"]] += bd_CMP_vs_SFLP
                        videos_calculated[average_name][method_names["CMP vs. SFLP"]] += 1

        if bd_EAC_vs_SFLP != np.inf:
            bitrate_savings[macro_average_name][method_names["EAC vs. SFLP"]] += bd_EAC_vs_SFLP
            videos_calculated[macro_average_name][method_names["EAC vs. SFLP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["EAC vs. SFLP"]] += bd_EAC_vs_SFLP
                        videos_calculated[average_name][method_names["EAC vs. SFLP"]] += 1
        
        if bd_ERP_vs_EAP != np.inf:
            bitrate_savings[macro_average_name][method_names["ERP vs. EAP"]] += bd_ERP_vs_EAP
            videos_calculated[macro_average_name][method_names["ERP vs. EAP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["ERP vs. EAP"]] += bd_ERP_vs_EAP
                        videos_calculated[average_name][method_names["ERP vs. EAP"]] += 1

        if bd_ERP_vs_CMP != np.inf:
            bitrate_savings[macro_average_name][method_names["ERP vs. CMP"]] += bd_ERP_vs_CMP
            videos_calculated[macro_average_name][method_names["ERP vs. CMP"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["ERP vs. CMP"]] += bd_ERP_vs_CMP
                        videos_calculated[average_name][method_names["ERP vs. CMP"]] += 1
        
        if bd_ERP_vs_EAC != np.inf:
            bitrate_savings[macro_average_name][method_names["ERP vs. EAC"]] += bd_ERP_vs_EAC
            videos_calculated[macro_average_name][method_names["ERP vs. EAC"]] += 1

            for average_name in average_names:
                    if matches_categories(average_name, video_categories[video_name], match_all=match_all):
                        bitrate_savings[average_name][method_names["ERP vs. EAC"]] += bd_ERP_vs_EAC
                        videos_calculated[average_name][method_names["ERP vs. EAC"]] += 1
    
    for average_name in average_names:
        if videos_calculated[average_name][method_names["ERP vs. SFLP"]] != 0:
            bitrate_savings[average_name][method_names["ERP vs. SFLP"]] /= videos_calculated[average_name][method_names["ERP vs. SFLP"]]
        else:
            bitrate_savings[average_name][method_names["ERP vs. SFLP"]] = "N/A"

        if videos_calculated[average_name][method_names["EAP vs. SFLP"]] != 0:
            bitrate_savings[average_name][method_names["EAP vs. SFLP"]] /= videos_calculated[average_name][method_names["EAP vs. SFLP"]]
        else:
            bitrate_savings[average_name][method_names["EAP vs. SFLP"]] = "N/A"
        
        if videos_calculated[average_name][method_names["CMP vs. SFLP"]] != 0:
            bitrate_savings[average_name][method_names["CMP vs. SFLP"]] /= videos_calculated[average_name][method_names["CMP vs. SFLP"]]
        else:
            bitrate_savings[average_name][method_names["CMP vs. SFLP"]] = "N/A"

        if videos_calculated[average_name][method_names["EAC vs. SFLP"]] != 0:
            bitrate_savings[average_name][method_names["EAC vs. SFLP"]] /= videos_calculated[average_name][method_names["EAC vs. SFLP"]]
        else:
            bitrate_savings[average_name][method_names["EAC vs. SFLP"]] = "N/A"

        if videos_calculated[average_name][method_names["ERP vs. EAP"]] != 0:
            bitrate_savings[average_name][method_names["ERP vs. EAP"]] /= videos_calculated[average_name][method_names["ERP vs. EAP"]]
        else:
            bitrate_savings[average_name][method_names["ERP vs. EAP"]] = "N/A"
        
        if videos_calculated[average_name][method_names["ERP vs. CMP"]] != 0:
            bitrate_savings[average_name][method_names["ERP vs. CMP"]] /= videos_calculated[average_name][method_names["ERP vs. CMP"]]
        else:
            bitrate_savings[average_name][method_names["ERP vs. CMP"]] = "N/A"
        
        if videos_calculated[average_name][method_names["ERP vs. EAC"]] != 0:
            bitrate_savings[average_name][method_names["ERP vs. EAC"]] /= videos_calculated[average_name][method_names["ERP vs. EAC"]]
        else:
            bitrate_savings[average_name][method_names["ERP vs. EAC"]] = "N/A"

    df_savings = pd.DataFrame.from_dict(bitrate_savings, orient="index")

    df_videos = df_savings[~df_savings.index.str.contains("Average|Macro")]

    # Now compute P25, P50, P75
    percentiles = df_videos.quantile([0.25, 0.5, 0.75])
    percentiles.index = ["P25", "P50", "P75"]

    # Append percentiles back to the original df_savings
    df_savings_with_percentiles = pd.concat([df_savings, percentiles])


    bitrate_savings_path = output_dir / f"bitrate_savings-{quality_info}.csv"
    df_savings_with_percentiles.to_csv(bitrate_savings_path)

    print(f"Bitrate savings saved to: {bitrate_savings_path}")
    print()
