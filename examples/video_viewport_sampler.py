from pathlib import Path
from PIL import Image
import math
import csv

from SFLP_toolkit.utilities.evaluation_utils import (
  generator_vec_geodesic_distance,
  ProjectionMetadata,
  EncodingProjection,
  SFLPMetadata,
  VideoEncoder,
  SFLPArrangementType,
)

from SFLP_toolkit.utilities.viewport_utils import (
    save_video_frame_viewport
)


sample_type = Image.Resampling.BILINEAR
input_video_container = VideoEncoder.H265
presets_and_pixels = "Preset_Slow-21.6M_Pixels"

# Input directories
ground_truth_dir = Path("D:\\GitHub_Projects\\VR_Encoding_Dataset\\Input") # Change this directory to where your high-resolution videos reside
reencoded_dir = Path("D:\\GitHub_Projects\\VR_Encoding_Dataset\\Output") # Change this directory where your compressed videos reside

reencoded_dir = Path("../output/videos/") 

# Output directory
output_dir = Path(f"..\\output\\images\\viewports")
output_dir.mkdir(parents=True, exist_ok=True)

# Input Videos
input_video_names = [
    # 8K Res

    # SJTU
    "AcademicBuilding",
    # "AdministrationBuilding",
    # "kaixuanmenye_pano",
    # "lanqiuchang_pano",
    # "Library",
    # "runner_pano",
    # "shuangziqiao_pano",
    # "SiyuanGate",
    # "SouthGate",
    # "StudyRoom",
    # "EastGate",
    # "Sward",
]

# Bitrates
bitrates = [
    # 1000,
    # 3000,
    # 5000,
    # 10000,
    25000,
    # 40000,
    # 60000,
]

# Viewport Sampling Parameters
viewport_direction = (0, 90)
viewport_angular_resolution = (90, 90)
viewport_resolution = (2160, 2160)

frame_number = 2

# Metadata
ground_truth_metadata = ProjectionMetadata(EncodingProjection.ERP)

SFLP_num_pixels = 18000001
SFLP_arrangement_type = SFLPArrangementType.STANDARD
padding_size = 2
# Hint: use generate_SFLP_pixel_arrangement() to generate the pixel arrangement to help derive these values.
optimal_index_hops = [2584, 4181] # [4181, 6765]
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
    padding_size=padding_size
)


CubeMap_num_pixels = 21660000
face_size = int(math.sqrt(CubeMap_num_pixels // 6))
CMP_metadata = ProjectionMetadata(EncodingProjection.CMP)
EAC_metadata =  ProjectionMetadata(EncodingProjection.EAC)

ERP_reencoded_output_size = (6600, 3300)
ERP_num_pixels = ERP_reencoded_output_size[0] * ERP_reencoded_output_size[1]
ERP_metadata = ProjectionMetadata(EncodingProjection.ERP)
EAP_metadata = ProjectionMetadata(EncodingProjection.EAP)

if __name__ == "__main__":
    
    for input_video_name in input_video_names:

        video_specific_output_dir = output_dir / input_video_name
        video_specific_output_dir.mkdir(parents=True, exist_ok=True)

        ground_truth_video_name = input_video_name + ".mp4"
        ground_truth_video_path = ground_truth_dir / ground_truth_video_name

        ground_truth_output_image_path = video_specific_output_dir / f"viewport-GroundTruth.png"
            
        save_video_frame_viewport(
            video_path=ground_truth_video_path,
            video_metadata=ERP_metadata,
            frame_number=frame_number,
            viewport_direction=viewport_direction,
            viewport_angular_resolution=viewport_angular_resolution,
            viewport_resolution=viewport_resolution,
            output_image_path=ground_truth_output_image_path,
            sample_type=sample_type,
        )

        for bitrate in bitrates:

            print(f"Processing input video: {input_video_name} with bitrate: {bitrate}")

            encoding_info = f"{input_video_container.name}-Target_{bitrate}k-Preset_Slow"
            input_dir = Path(reencoded_dir, f"{encoding_info}")

            ERP_video_name = f"Reencoded_ERP-{input_video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            ERP_video_path = input_dir / ERP_video_name
            ERP_output_image_path = video_specific_output_dir / f"viewport-ERP-{encoding_info}.png"

            save_video_frame_viewport(
                video_path=ERP_video_path,
                video_metadata=ERP_metadata,
                frame_number=frame_number,
                viewport_direction=viewport_direction,
                viewport_angular_resolution=viewport_angular_resolution,
                viewport_resolution=viewport_resolution,
                output_image_path=ERP_output_image_path,
                sample_type=sample_type,
            )

            EAP_video_name = f"EAP-360_no_tiling-{input_video_name}-{ERP_num_pixels}-bitrate_{bitrate}.mp4"
            EAP_video_path = input_dir / EAP_video_name
            EAP_output_image_path = video_specific_output_dir / f"viewport-EAP-{encoding_info}.png"

            save_video_frame_viewport(
                video_path=EAP_video_path,
                video_metadata=EAP_metadata,
                frame_number=frame_number,
                viewport_direction=viewport_direction,
                viewport_angular_resolution=viewport_angular_resolution,
                viewport_resolution=viewport_resolution,
                output_image_path=EAP_output_image_path,
                sample_type=sample_type,
            )
            CMP_video_name = f"CMP-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            CMP_video_path = input_dir / CMP_video_name
            CMP_output_image_path = video_specific_output_dir / f"viewport-CMP-{encoding_info}.png"

            save_video_frame_viewport(
                video_path=CMP_video_path,
                video_metadata=CMP_metadata,
                frame_number=frame_number,
                viewport_direction=viewport_direction,
                viewport_angular_resolution=viewport_angular_resolution,
                viewport_resolution=viewport_resolution,
                output_image_path=CMP_output_image_path,
                sample_type=sample_type,
            )

            EAC_video_name = f"EAC-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}-bitrate_{bitrate}.mp4"
            EAC_video_path = input_dir / EAC_video_name
            EAC_output_image_path = video_specific_output_dir / f"viewport-EAC-{encoding_info}.png"

            save_video_frame_viewport(
                video_path=EAC_video_path,
                video_metadata=EAC_metadata,
                frame_number=frame_number,
                viewport_direction=viewport_direction,
                viewport_angular_resolution=viewport_angular_resolution,
                viewport_resolution=viewport_resolution,
                output_image_path=EAC_output_image_path,
                sample_type=sample_type,
            )

            SFLP_video_name = f"SFLP-360_no_tiling-{input_video_name}-{SFLP_num_pixels}-SFLP_arrangement_{SFLP_arrangement_type.name}-bitrate_{bitrate}.mp4"
            SFLP_video_path = input_dir / SFLP_video_name
            SFLP_output_image_path = video_specific_output_dir / f"viewport-SFLP-Arrangement_{SFLP_arrangement_type.name}-{encoding_info}.png"

            SFLP_video_path = reencoded_dir / "SFLP-AcademicBuilding-18000001-STANDARD.mp4"
            SFLP_output_image_path = video_specific_output_dir / f"viewport-SFLP-Arrangement_{SFLP_arrangement_type.name}.png"

            save_video_frame_viewport(
                video_path=SFLP_video_path,
                video_metadata=SFLP_metadata,
                frame_number=frame_number,
                viewport_direction=viewport_direction,
                viewport_angular_resolution=viewport_angular_resolution,
                viewport_resolution=viewport_resolution,
                output_image_path=SFLP_output_image_path,
                sample_type=sample_type,
            )
