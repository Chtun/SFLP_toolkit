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
        std::string output_dir_str = "../../../output/videos/viewport_sequences";

        std::string input_video_name = "AcademicBuilding";
        int bitrate = 20000;

        std::string ffmpeg_path = "ffmpeg";
        std::string ffprobe_path = "ffprobe";
        bool runtime_debug = false;
        bool verbose_debug = false;

        // Viewport config path default
        std::string viewport_config_path = "../../../input/config/video_viewport_config.json";

        // Metadata config path defaults
        std::string ground_truth_config_path = "../../../input/config/ERP_metadata.json";
        std::string ERP_config_path = "../../../input/config/ERP_metadata.json";
        std::string EAP_config_path = "../../../input/config/EAP_metadata.json";
        std::string CMP_config_path = "../../../input/config/CMP_metadata.json";
        std::string EAC_config_path = "../../../input/config/EAC_metadata.json";
        std::string SFLP_config_path = "../../../input/config/SFLP_metadata.json";

        // Parse command-line flags
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::cout << "SFLP VR Toolkit - Viewport Sequence Video Renderer\n\n"
                          << "Usage: " << argv[0] << " [options]\n\n"
                          << "Options:\n"
                          << "  --help, -h                  Show this help message and exit\n"
                          << "  --sample_type <type>        Sampling type: NEAREST, BILINEAR, IDW (default: NEAREST)\n"
                          << "  --input_name <string>       Name of input video (default: AcademicBuilding)\n"
                          << "  --bitrate <kbps>            Target bitrate (default: 20000)\n"
                          << "  --input_dir <path>          Path to ground truth input video directory\n"
                          << "  --output_dir <path>         Path to rendered output video directory\n"
                          << "  --viewport_config <path>    Path to Viewport Configuration JSON\n"
                          << "  --gt_config <path>          Path to Ground Truth metadata JSON\n"
                          << "  --erp_config <path>         Path to ERP metadata JSON\n"
                          << "  --eap_config <path>         Path to EAP metadata JSON\n"
                          << "  --cmp_config <path>         Path to CMP metadata JSON\n"
                          << "  --eac_config <path>         Path to EAC metadata JSON\n"
                          << "  --sflp_config <path>        Path to SFLP metadata JSON\n"
                          << "  --ffmpeg_path <path>        Path to FFmpeg executable (default: ffmpeg)\n"
                          << "  --ffprobe_path <path>       Path to FFprobe executable (default: ffprobe)\n"
                          << "  --runtime_debug             Displays detailed runtime cost for sampling process\n"
                          << "  --verbose_debug             Shows verbose debug (only if runtime_debug flag is used)\n";
                return 0;
            } else if (arg == "--sample_type" && i + 1 < argc) {
                std::string st_str = argv[++i];
                if (st_str == "NEAREST" || st_str == "nearest") {
                    sample_type = SamplerResampling::NEAREST;
                } else if (st_str == "BILINEAR" || st_str == "bilinear") {
                    sample_type = SamplerResampling::BILINEAR;
                } else if (st_str == "IDW" || st_str == "idw") {
                    sample_type = SamplerResampling::IDW;
                } else {
                    std::cerr << "Warning: Unknown sample type '" << st_str 
                              << "'. Defaulting to NEAREST.\n";
                    sample_type = SamplerResampling::NEAREST;
                }
            } else if (arg == "--input_name" && i + 1 < argc) {
                input_video_name = argv[++i];
            } else if (arg == "--bitrate" && i + 1 < argc) {
                bitrate = std::stoi(argv[++i]);
            } else if (arg == "--input_dir" && i + 1 < argc) {
                ground_truth_dir_str = argv[++i];
            } else if (arg == "--output_dir" && i + 1 < argc) {
                output_dir_str = argv[++i];
            } else if (arg == "--viewport_config" && i + 1 < argc) {
                viewport_config_path = argv[++i];
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
            } else if (arg == "--ffmpeg_path" && i + 1 < argc) {
                ffmpeg_path = argv[++i];
            } else if (arg == "--ffprobe_path" && i + 1 < argc) {
                ffprobe_path = argv[++i];
            } else if (arg == "--runtime_debug" && i < argc) {
                runtime_debug = true;
            } else if (arg == "--verbose_debug" && i < argc) {
                verbose_debug = true;
            }
        }

        // Input/Output directories
        fs::path ground_truth_dir(ground_truth_dir_str);
        fs::path reencoded_dir(reencoded_dir_str);
        fs::path output_dir(output_dir_str);
        fs::create_directories(output_dir);


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

        std::string encoding_info = std::string(magic_enum::enum_name(input_video_container)) + 
                                    "-Target_" + std::to_string(bitrate) + 
                                    "k-Preset_Slow";
        fs::path input_dir = reencoded_dir;

        // // Ground Truth Video Sequence
        // fs::path ground_truth_video_path = ground_truth_dir / (input_video_name + ".mp4");
        // fs::path ground_truth_output_video_path = video_specific_output_dir / "viewport_sequence-GroundTruth.mp4";
        // std::cout << "Rendering Ground Truth viewport video sequence...\n";
        // process_video_viewport_rendering(
        //     viewport_config_path,
        //     ground_truth_video_path,
        //     ground_truth_output_video_path,
        //     ground_truth_metadata,
        //     sample_type,
        //     ffmpeg_path,
        //     ffprobe_path,
        //     runtime_debug,
        //     verbose_debug
        // );

        // // Video Sequence from ERP
        // fs::path ERP_video_path = input_dir / ("Reencoded_ERP-" + input_video_name + "-" + std::to_string(ERP_metadata.num_pixels) + ".mp4");
        // fs::path ERP_output_video_path = video_specific_output_dir / ("viewport_sequence-ERP-" + encoding_info + ".mp4");
        // std::cout << "Rendering ERP viewport video sequence...\n";
        // process_video_viewport_rendering(
        //     viewport_config_path,
        //     ERP_video_path,
        //     ERP_output_video_path,
        //     ERP_metadata,
        //     sample_type,
        //     ffmpeg_path,
        //     ffprobe_path,
        //     runtime_debug
        // );

        // // Video Sequence from EAP
        // fs::path EAP_video_path = input_dir / ("EAP-360_no_tiling-" + input_video_name + "-" + std::to_string(EAP_metadata.num_pixels) + ".mp4");
        // fs::path EAP_output_video_path = video_specific_output_dir / ("viewport_sequence-EAP-" + encoding_info + ".mp4");
        // std::cout << "Rendering EAP viewport video sequence...\n";
        // process_video_viewport_rendering(
        //     viewport_config_path,
        //     EAP_video_path,
        //     EAP_output_video_path,
        //     EAP_metadata,
        //     sample_type,
        //     ffmpeg_path,
        //     ffprobe_path,
        //     runtime_debug,
        //     verbose_debug
        // );

        // // Video Sequence from CMP
        // fs::path CMP_video_path = input_dir / ("CMP-360_no_tiling-" + input_video_name + "-" + std::to_string(CMP_metadata.num_pixels) + ".mp4");
        // fs::path CMP_output_video_path = video_specific_output_dir / ("viewport_sequence-CMP-" + encoding_info + ".mp4");
        // std::cout << "Rendering CMP viewport video sequence...\n";
        // process_video_viewport_rendering(
        //     viewport_config_path,
        //     CMP_video_path,
        //     CMP_output_video_path,
        //     CMP_metadata,
        //     sample_type,
        //     ffmpeg_path,
        //     ffprobe_path,
        //     runtime_debug,
        //     verbose_debug
        // );

        // Video Sequence from EAC
        fs::path EAC_video_path = input_dir / ("EAC-" + input_video_name + "-" + std::to_string(EAC_metadata.num_pixels) + ".mp4");
        fs::path EAC_output_video_path = video_specific_output_dir / ("viewport_sequence-EAC-" + encoding_info + ".mp4");
        std::cout << "\nRendering EAC viewport video sequence...\n";
        process_video_viewport_rendering(
            viewport_config_path,
            EAC_video_path,
            EAC_output_video_path,
            EAC_metadata,
            sample_type,
            ffmpeg_path,
            ffprobe_path,
            runtime_debug,
            verbose_debug
        );

        // Video Sequence frin SFLP
        fs::path SFLP_video_path = input_dir / (std::string(magic_enum::enum_name(SFLP_metadata.projection_type)) + "-"
            + input_video_name + "-"
            + std::to_string(SFLP_metadata.num_pixels) + "-"
            + std::string(magic_enum::enum_name(SFLP_metadata.arrangement_type)) + ".mp4"
        );
        fs::path SFLP_output_video_path = video_specific_output_dir / ("viewport_sequence-SFLP-" + encoding_info + ".mp4");
        std::cout << "\nRendering SFLP viewport video sequence...\n";
        process_video_viewport_rendering(
            viewport_config_path,
            SFLP_video_path,
            SFLP_output_video_path,
            SFLP_metadata,
            sample_type,
            ffmpeg_path,
            ffprobe_path,
            runtime_debug,
            verbose_debug
        );

        std::cout << "All projection viewport video sequences rendered successfully for " << input_video_name << "!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}