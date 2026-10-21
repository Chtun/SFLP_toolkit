#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>

#include "../include/file_conversion_utils.hpp"
#include "../include/viewport_utils.hpp"
#include "../include/magic_enum.hpp"

namespace fs = std::filesystem;
int main(int argc, char* argv[]) {
    try {
        // Default values
        SamplerResampling sample_type = SamplerResampling::NEAREST;
        VideoEncoder input_video_container = VideoEncoder::H265;
        
        std::string ground_truth_dir_str = "../../../input/videos/";
        std::string reencoded_dir_str = "../../../output/videos/";
        std::string output_dir_str = "../../../output/images/viewports";

        std::string input_video_name = "AcademicBuilding";
        int bitrate = 20000;
        int frame_number = 234;

        std::string ffmpeg_path = "ffmpeg";
        std::string ffprobe_path = "ffprobe";
        bool runtime_debug = false;

        // Config path defaults (empty string falls back to default constructors if not specified)
        std::string ground_truth_config_path = "../../../input/config/ERP_metadata.json";
        std::string ERP_config_path = "../../../input/config/ERP_metadata.json";
        std::string EAP_config_path = "../../../input/config/EAP_metadata.json";
        std::string CMP_config_path = "../../../input/config/CMP_metadata.json";
        std::string EAC_config_path = "../../../input/config/EAC_metadata.json";
        std::string SFLP_config_path = "../../../input/config/SFLP_metadata.json";

        // Viewport Sampling Parameters
        std::pair<double, double> viewport_direction(0.0, 90.0);
        std::pair<double, double> viewport_angular_resolution(90.0, 90.0);
        std::pair<int, int> viewport_resolution(2160, 2160);

        // Parse command-line flags
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::cout << "SFLP VR Toolkit - Video Viewport Sampler\n\n"
                        << "Usage: " << argv[0] << " [options]\n\n"
                        << "Options:\n"
                        << "  --help, -h                  Show this help message and exit\n"
                        << "  --input_name <string>       Name of input video (default: AcademicBuilding)\n"
                        << "  --bitrate <kbps>            Target bitrate (default: 20000)\n"
                        << "  --frame <num>               Frame number to sample (default: 234)\n"
                        << "  --input_dir <path>          Path to ground truth input video directory\n"
                        << "  --output_dir <path>         Path to re-encoded dataset output directory\n"
                        << "  --save_dir <path>           Path to save sampled viewports\n"
                        << "  --gt_config <path>          Path to Ground Truth metadata JSON\n"
                        << "  --erp_config <path>         Path to ERP metadata JSON\n"
                        << "  --eap_config <path>         Path to EAP metadata JSON\n"
                        << "  --cmp_config <path>         Path to CMP metadata JSON\n"
                        << "  --eac_config <path>         Path to EAC metadata JSON\n"
                        << "  --sflp_config <path>        Path to SFLP metadata JSON\n"
                        << "  --viewport_dir <yaw,pitch>  Viewport center direction angles (default: 0.0,90.0)\n"
                        << "  --viewport_fov <w,h>        Viewport angular resolution / FOV (default: 90.0,90.0)\n"
                        << "  --viewport_res <w,h>        Viewport pixel resolution (default: 2160,2160)\n"
                        << "  --ffmpeg_path <path>        Path to FFmpeg executable (default: ffmpeg)\n"
                        << "  -- ffprobe_path <path>      Path to FFprobe executable (default: ffprobe)\n"
                        << "  --runtime_debug             Displays detailed runtime cost for sampling process\n";
                return 0;
            } else if (arg == "--input_name" && i + 1 < argc) {
                input_video_name = argv[++i];
            } else if (arg == "--bitrate" && i + 1 < argc) {
                bitrate = std::stoi(argv[++i]);
            } else if (arg == "--frame" && i + 1 < argc) {
                frame_number = std::stoi(argv[++i]);
            } else if (arg == "--input_dir" && i + 1 < argc) {
                ground_truth_dir_str = argv[++i];
            } else if (arg == "--output_dir" && i + 1 < argc) {
                reencoded_dir_str = argv[++i];
            } else if (arg == "--save_dir" && i + 1 < argc) {
                output_dir_str = argv[++i];
            } else if (arg == "--gt_config" && i + 1 < argc) {
                ground_truth_config_path = argv[++i];
            } else if (arg == "--erp_config" && i + 1 < argc) {
                ERP_config_path = argv[++i];
            } else if (arg == "--eap_config" && i + 1 < argc) {
                EAP_config_path = argv[++i];
            } else if (arg == "--cmp_config" && i + 1 < argc) {
                CMP_config_path = argv[++i];
            } else if (arg == "--eac_config" && i + 1 < argc) {
                EAC_config_path = argv[++i];
            } else if (arg == "--sflp_config" && i + 1 < argc) {
                SFLP_config_path = argv[++i];
            } else if (arg == "--viewport_dir" && i + 1 < argc) {
                viewport_direction = parse_double_pair(argv[++i]);
            } else if (arg == "--viewport_fov" && i + 1 < argc) {
                viewport_angular_resolution = parse_double_pair(argv[++i]);
            } else if (arg == "--viewport_res" && i + 1 < argc) {
                viewport_resolution = parse_int_pair(argv[++i]);
            } else if (arg == "--ffmpeg_path" && i + 1 < argc) {
                ffmpeg_path = argv[++i];
            } else if (arg == "--ffprobe_path" && i + 1 < argc) {
                ffprobe_path = argv[++i];
            } else if (arg == "--runtime_debug" && i < argc) {
                runtime_debug = true;
            }
        }

        // Input/Output directories
        fs::path ground_truth_dir(ground_truth_dir_str);
        fs::path reencoded_dir(reencoded_dir_str);
        fs::path output_dir(output_dir_str);
        fs::create_directories(output_dir);

        // Metadata Definitions using load_metadata_from_json where paths are provided
        int SFLP_num_pixels = 18000001;

        ProjectionMetadata ground_truth_metadata = ground_truth_config_path.empty() ? 
            ProjectionMetadata(EncodingProjection::ERP, 18000000) : load_metadata_from_json(ground_truth_config_path, 18000000);

        ProjectionMetadata ERP_metadata = ERP_config_path.empty() ? 
            ProjectionMetadata(EncodingProjection::ERP, 18000000) : load_metadata_from_json(ERP_config_path, 18000000);

        ProjectionMetadata EAP_metadata = EAP_config_path.empty() ? 
            ProjectionMetadata(EncodingProjection::EAP, 18000000) : load_metadata_from_json(EAP_config_path, 18000000);

        ProjectionMetadata CMP_metadata = CMP_config_path.empty() ? 
            ProjectionMetadata(EncodingProjection::CMP, 17998944) : load_metadata_from_json(CMP_config_path, 17998944);

        ProjectionMetadata EAC_metadata = EAC_config_path.empty() ? 
            ProjectionMetadata(EncodingProjection::EAC, 17998944) : load_metadata_from_json(EAC_config_path, 17998944);

        SFLPMetadata SFLP_metadata = SFLP_config_path.empty() ? 
            [&]() {
                int padding_size = 16;
                std::pair<int, int> optimal_index_hops(2584, 4181);
                return SFLPMetadata(SFLP_num_pixels, SFLPArrangementType::STANDARD, optimal_index_hops, -1, 0, padding_size);
            }() : [&]() {
                auto ptr = load_metadata_ptr_from_json(SFLP_config_path, SFLP_num_pixels);
                auto* sflp_ptr = dynamic_cast<SFLPMetadata*>(ptr.get());
                if (!sflp_ptr) {
                    throw std::runtime_error("Failed to cast ProjectionMetadata to SFLPMetadata!");
                }
                return *sflp_ptr;
            }();
        

        fs::path video_specific_output_dir = output_dir / input_video_name;
        fs::create_directories(video_specific_output_dir);

        // fs::path ground_truth_video_path = ground_truth_dir / (input_video_name + ".mp4");
        // fs::path ground_truth_output_image_path = video_specific_output_dir / "viewport-GroundTruth.png";

        // std::cout << "Sampling Ground Truth viewport for video: " << input_video_name << " at frame " << frame_number << "\n";
        // save_video_frame_viewport_gpu(
        //     ground_truth_video_path,
        //     ground_truth_metadata,
        //     frame_number,
        //     viewport_direction,
        //     viewport_angular_resolution,
        //     viewport_resolution,
        //     ground_truth_output_image_path,
        //     sample_type,
        //     ffmpeg_path,
        //     ffprobe_path,
        //     runtime_debug
        // );

        std::cout << "Processing re-encoded viewports with bitrate: " << bitrate << " kbps\n";

        std::string encoding_info = std::string(magic_enum::enum_name(input_video_container)) + 
                                    "-Target_" + std::to_string(bitrate) + 
                                    "k-Preset_Slow";
        fs::path input_dir = reencoded_dir; // / encoding_info;

        // // ERP
        // fs::path ERP_video_path = input_dir / ("Reencoded_ERP-" + input_video_name + "-" + std::to_string(ERP_metadata.num_pixels) + ".mp4");
        // fs::path ERP_output_image_path = video_specific_output_dir / ("viewport-ERP-" + encoding_info + ".png");
        // save_video_frame_viewport_gpu(
        //     ERP_video_path,
        //     ERP_metadata,
        //     frame_number,
        //     viewport_direction,
        //     viewport_angular_resolution,
        //     viewport_resolution,
        //     ERP_output_image_path,
        //     sample_type,
        //     ffmpeg_path,
        //     ffprobe_path,
        //     runtime_debug
        // );

        // // EAP
        // fs::path EAP_video_path = input_dir / ("EAP-" + input_video_name + "-" + std::to_string(EAP_metadata.num_pixels) + ".mp4");
        // fs::path EAP_output_image_path = video_specific_output_dir / ("viewport-EAP-" + encoding_info + ".png");
        // save_video_frame_viewport_gpu(
        //     EAP_video_path,
        //     EAP_metadata,
        //     frame_number,
        //     viewport_direction,
        //     viewport_angular_resolution,
        //     viewport_resolution,
        //     EAP_output_image_path,
        //     sample_type,
        //     ffmpeg_path,
        //     ffprobe_path,
        //     runtime_debug
        // );

        // // CMP
        // fs::path CMP_video_path = input_dir / ("CMP-" + input_video_name + "-" + std::to_string(CMP_metadata.num_pixels) + ".mp4");
        // fs::path CMP_output_image_path = video_specific_output_dir / ("viewport-CMP-" + encoding_info + ".png");
        // save_video_frame_viewport_gpu(
        //     CMP_video_path,
        //     CMP_metadata,
        //     frame_number,
        //     viewport_direction,
        //     viewport_angular_resolution,
        //     viewport_resolution,
        //     CMP_output_image_path,
        //     sample_type,
        //     ffmpeg_path,
        //     ffprobe_path,
        //     runtime_debug
        // );    

        // EAC
        fs::path EAC_video_path = input_dir / ("EAC-" + input_video_name + "-" + std::to_string(EAC_metadata.num_pixels) + ".mp4");
        fs::path EAC_output_image_path = video_specific_output_dir / ("viewport-EAC-" + encoding_info + ".png");
        save_video_frame_viewport_gpu(
            EAC_video_path,
            EAC_metadata,
            frame_number,
            viewport_direction,
            viewport_angular_resolution,
            viewport_resolution,
            EAC_output_image_path,
            sample_type,
            ffmpeg_path,
            ffprobe_path,
            runtime_debug
        );

        // SFLP
        fs::path SFLP_video_path = input_dir / (std::string(magic_enum::enum_name(SFLP_metadata.projection_type)) + "-"
            + input_video_name + "-"
            + std::to_string(SFLP_metadata.num_pixels) + "-"
            + std::string(magic_enum::enum_name(SFLP_metadata.arrangement_type)) + ".mp4"
        );
        
        fs::path SFLP_output_image_path = video_specific_output_dir / ("viewport-SFLP-"+ encoding_info + ".png");
        save_video_frame_viewport_gpu(
            SFLP_video_path,
            SFLP_metadata,
            frame_number,
            viewport_direction,
            viewport_angular_resolution,
            viewport_resolution,
            SFLP_output_image_path, 
            sample_type,
            ffmpeg_path,
            ffprobe_path,
            runtime_debug
        );

        std::cout << "All viewports extracted successfully for " << input_video_name << "!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}