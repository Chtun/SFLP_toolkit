from pathlib import Path
from PIL import Image
import math
from config import FFMPEG, FFMPEG_GPU_PATH
from SFLP_toolkit.utilities.file_conversion_utils import (
  ERP_to_SFLP_video_conversion,
  ERP_to_CMP_video_conversion,
  ERP_to_EAC_video_conversion,
  ERP_to_EAP_video_conversion,
  reencode_ERP_video,
  VideoEncoder,
  SFLPArrangementType,
)

# Input Directories
input_dir = Path("D:\\GitHub_Projects\\VR_Encoding_Dataset\\Input") # Change this directory to where your high-resolution videos reside
layout_input_dir = Path("../output/layouts") # This directory holds the layouts for SFLP and CubeMap.

# Output Directory
output_dir = Path("D:\\GitHub_Projects\\VR_Encoding_Dataset\\Output") # Change this directory to where you want your compressed videos to reside
output_dir.mkdir(parents=True, exist_ok=True)

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

SFLP_num_pixels = 18000001
SFLP_arrangement_type = SFLPArrangementType.STANDARD

CubeMap_num_pixels = 17998944
face_size = int(math.sqrt(CubeMap_num_pixels // 6))
CubeMap_layout_csv_path = layout_input_dir / f"CubeMap_pixel_arrangement-360_No_Tiling-{CubeMap_num_pixels}.csv"

ERP_reencoded_output_size = (6000, 3000)
ERP_num_pixels = ERP_reencoded_output_size[0] * ERP_reencoded_output_size[1]

sample_type = Image.Resampling.BILINEAR
video_encoder = VideoEncoder.H265


def process_video(input_video_name: str, bitrates: list[int], SFLP_num_pixels: int):
    input_video_path = input_dir / f"{input_video_name}.mp4"
    
    SFLP_layout_csv_path = layout_input_dir / f"SFLP_arrangement_{SFLP_arrangement_type.name}-360_No_Tiling-{SFLP_num_pixels}_padded.csv"
    SFLP_output_name = f"SFLP-360_no_tiling-{input_video_name}-{SFLP_num_pixels}-SFLP_arrangement_{SFLP_arrangement_type.name}"
    ERP_to_SFLP_video_conversion(
        num_pixels=SFLP_num_pixels,
        csv_path=SFLP_layout_csv_path,
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=SFLP_output_name,
        output_video_type=video_encoder,
        ffmpeg_path=FFMPEG,
        sample_type=sample_type,
        bitrates=bitrates
    )

    CMP_output_name = f"CMP-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}"
    ERP_to_CMP_video_conversion(
        face_size=face_size,
        csv_path=CubeMap_layout_csv_path,
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=CMP_output_name,
        output_video_type=video_encoder,
        ffmpeg_path=FFMPEG,
        sample_type=sample_type,
        bitrates=bitrates
    )

    EAC_output_name = f"EAC-360_no_tiling-{input_video_name}-{CubeMap_num_pixels}"
    ERP_to_EAC_video_conversion(
        face_size=face_size,
        csv_path=CubeMap_layout_csv_path,
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=EAC_output_name,
        output_video_type=video_encoder,
        ffmpeg_path=FFMPEG,
        sample_type=sample_type,
        bitrates=bitrates
    )

    EAP_output_name = f"EAP-360_no_tiling-{input_video_name}-{ERP_num_pixels}"
    ERP_to_EAP_video_conversion(
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=EAP_output_name,
        output_size=ERP_reencoded_output_size,
        output_video_type=video_encoder,
        ffmpeg_path=FFMPEG,
        sample_type=sample_type,
        bitrates=bitrates
    )

    ERP_output_name = f"Reencoded_ERP-{input_video_name}-{ERP_num_pixels}"
    reencode_ERP_video(
        input_video_path=input_video_path,
        output_dir=output_dir,
        output_name=ERP_output_name,
        output_size=ERP_reencoded_output_size,
        output_video_type=video_encoder,
        ffmpeg_path=FFMPEG,
        sample_type=sample_type,
        bitrates=bitrates
    )

if __name__ == "__main__":
    for input_video_name in input_video_names:
        process_video(
            input_video_name=input_video_name,
            bitrates=bitrates,
            SFLP_num_pixels=SFLP_num_pixels
        )
