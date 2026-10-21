from pathlib import Path
from PIL import Image

from SFLP_toolkit.utilities.evaluation_utils import (
  VideoEncoder,
)


sample_type = Image.Resampling.BILINEAR
output_video_container = VideoEncoder.H265
presets_and_pixels = "Preset_Slow-21.6M_Pixels"

# Input directories
input_dir = Path(f"..\\output\\images\\viewports")

# Output directory
output_dir = Path(f"..\\output\\images\\viewports\\cutouts")
output_dir.mkdir(parents=True, exist_ok=True)

# Input Videos
input_video_names = [
    # 8K Res

    # SJTU
    # "AcademicBuilding",
    # "AdministrationBuilding",
    # "kaixuanmenye_pano",
    # "lanqiuchang_pano",
    "Library",
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
    5000,
    10000,
    25000,
    # 40000,
    # 60000,
]


# Image Cutout Parameters
corners = [(1060, 1060), (1210, 1240)] # List of tuples of (x, y) coordinates for the top-left and bottom-right corners of the cutout region for each video. Length should match length of input_video_names list.

def create_cutout(image_path: Path, output_path: Path, top_left: tuple, bottom_right: tuple):
    """
    Cuts out a rectangular section of an image and saves it.
    """
    if not image_path.exists():
        print(f"Skipping: {image_path.name} does not exist.")
        return

    with Image.open(image_path) as img:
        # PIL crop box is (left, top, right, bottom)
        crop_box = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        
        # Perform the crop
        cutout = img.crop(crop_box)
        
        # Save the result
        cutout_output_path = output_path.parent / f"{output_path.stem}-cutout.png"
        cutout.save(cutout_output_path)
        print(f"Created cutout: {cutout_output_path}")

if __name__ == "__main__":
    
    for input_video_name in input_video_names:

        video_specific_input_dir = input_dir / input_video_name
        video_specific_output_dir = output_dir / input_video_name
        video_specific_output_dir.mkdir(parents=True, exist_ok=True)

        ground_truth_input_image_path = video_specific_input_dir / f"viewport-GroundTruth.png"
        ground_truth_output_image_path = video_specific_output_dir / f"viewport-GroundTruth.png"

        # Run cutout on this image!
        create_cutout(ground_truth_input_image_path, ground_truth_output_image_path, corners[0], corners[1])

        for bitrate in bitrates:

            print(f"Processing input image: {input_video_name} with bitrate: {bitrate}")

            encoding_info = f"{output_video_container.name}-Target_{bitrate}k-Preset_Slow"


            ERP_input_image_path = video_specific_input_dir / f"viewport-ERP-{encoding_info}.png"
            ERP_output_image_path = video_specific_output_dir / f"viewport-ERP-{encoding_info}.png"

            # Run cutout on this image!
            create_cutout(ERP_input_image_path, ERP_output_image_path, corners[0], corners[1])

            EAP_input_image_path = video_specific_input_dir / f"viewport-EAP-{encoding_info}.png"
            EAP_output_image_path = video_specific_output_dir / f"viewport-EAP-{encoding_info}.png"

            # Run cutout on this image!
            create_cutout(EAP_input_image_path, EAP_output_image_path, corners[0], corners[1])
            
            CMP_input_image_path = video_specific_input_dir / f"viewport-CMP-{encoding_info}.png"
            CMP_output_image_path = video_specific_output_dir / f"viewport-CMP-{encoding_info}.png"

            # Run cutout on this image!
            create_cutout(CMP_input_image_path, CMP_output_image_path, corners[0], corners[1])
           
            EAC_input_image_path = video_specific_input_dir / f"viewport-EAC-{encoding_info}.png"
            EAC_output_image_path = video_specific_output_dir / f"viewport-EAC-{encoding_info}.png"

            # Run cutout on this image!
            create_cutout(EAC_input_image_path, EAC_output_image_path, corners[0], corners[1])
            
            SFLP_input_image_path = video_specific_input_dir / f"viewport-SFLP-{encoding_info}.png"
            SFLP_output_image_path = video_specific_output_dir / f"viewport-SFLP-{encoding_info}.png"

            # Run cutout on this image!
            create_cutout(SFLP_input_image_path, SFLP_output_image_path, corners[0], corners[1])