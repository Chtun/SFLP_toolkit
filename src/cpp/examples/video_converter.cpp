#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>

#include "../include/data_types.hpp"
#include "../include/file_conversion_utils.hpp"
#include "../include/pixel_arrangement_utils.hpp"

// Macro for standard CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

int main(int argc, char* argv[]) {
    try {
        // Set default values matching video workflows
        std::string input_dir = "../../../input/videos";
        std::string input_file_name = "AcademicBuilding.mp4";
        std::string input_metadata_path = "../../../input/config/ERP_metadata.json";
        
        std::string output_dir = "../../../output/videos";
        std::string output_file_name_override = ""; // Empty string triggers auto-generation
        
        std::string ffmpeg_path = "ffmpeg";
        std::string ffprobe_path = "ffprobe";
        EncodingProjection output_projection_type = EncodingProjection::SFLP;
        int output_num_pixels = 18000001;
        int padding_size = 2;
        int bitrate = 20000; // kbps
        int fps = 25;
        double duration = 1.0;
        std::string preset = "slow"; // FFmpeg preset for H.265 encoding
        int width = 0;  // Will be autopopulated via ffprobe
        int height = 0; // Will be autopopulated via ffprobe

        // Parse command-line flags
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::cout << "SFLP VR Toolkit - Video Converter\n\n"
                          << "Usage: " << argv[0] << " [options]\n\n"
                          << "Options:\n"
                          << "  --help, -h                        Show this help message and exit\n"
                          << "  --input_dir <path>                Path to input video directory\n"
                          << "  --input_file_name <name>          Name of the input video file\n"
                          << "  --input_metadata_path <path>      Path to input video's metadata\n"
                          << "  --output_dir <path>               Path to output video directory\n"
                          << "  --output_file_name <name>         Override auto-generated output H.265 video filename\n"
                          << "  --output_projection_type <type>   Target projection type (default: SFLP)\n"
                          << "  --output_image_num_pixels <num>   Target number of pixels for SFLP (default: 18000001)\n"
                          << "  --padding_size <num>              Target padding size (ensures output height and width are divisible by padding_size, default: 2)\n"
                          << "  --ffmpeg_path <path>              Path to ffmpeg executable\n"
                          << "  --ffprobe_path <path>             Path to ffprobe executable\n"
                          << "  --bitrate <kbps>                  Target bitrate (default: 20000)\n"
                          << "  --preset <string>                 FFmpeg preset for H.265 encoding (default: slow)\n"
                          << "  --width <num>                     Override source frame width\n"
                          << "  --height <num>                    Override source frame height\n";
                return 0;
            } else if (arg == "--input_dir" && i + 1 < argc) {
                input_dir = argv[++i];
            } else if (arg == "--input_file_name" && i + 1 < argc) {
                input_file_name = argv[++i];
            } else if (arg == "--input_metadata_path" && i + 1 < argc) {
                input_metadata_path = argv[++i];
            } else if (arg == "--output_dir" && i + 1 < argc) {
                output_dir = argv[++i];
            } else if (arg == "--output_file_name" && i + 1 < argc) {
                output_file_name_override = argv[++i];
            } else if (arg == "--output_projection_type" && i + 1 < argc) {
                std::string pt_str = argv[++i];
                if (pt_str == "SFLP" || pt_str == "sflp") {
                    output_projection_type = EncodingProjection::SFLP;
                } else if (pt_str == "EAP" || pt_str == "eap") {
                    output_projection_type = EncodingProjection::EAP;
                } else if (pt_str == "CMP" || pt_str == "cmp") {
                    output_projection_type = EncodingProjection::CMP;
                } else if (pt_str == "EAC" || pt_str == "eac") {
                    output_projection_type = EncodingProjection::EAC;
                } else {
                    std::cerr << "Warning: Unknown projection type '" << pt_str 
                              << "'. Defaulting to SFLP.\n";
                    output_projection_type = EncodingProjection::SFLP;
                }
            } else if (arg == "--output_image_num_pixels" && i + 1 < argc) {
                output_num_pixels = std::stoi(argv[++i]);
            } else if (arg == "--padding_size" && i + 1 < argc) {
                padding_size = std::stoi(argv[++i]);
            } else if (arg == "--ffmpeg_path" && i + 1 < argc) {
                ffmpeg_path = argv[++i];
            } else if (arg == "--ffprobe_path" && i + 1 < argc) {
                ffprobe_path = argv[++i];
            } else if (arg == "--bitrate" && i + 1 < argc) {
                bitrate = std::stoi(argv[++i]);
            } else if (arg == "--preset" && i + 1 < argc) {
                preset = argv[++i];
            } else if (arg == "--width" && i + 1 < argc) {
                width = std::stoi(argv[++i]);
            } else if (arg == "--height" && i + 1 < argc) {
                height = std::stoi(argv[++i]);
            }
        }

        // Construct full input path
        std::string input_video_path = input_dir + "/" + input_file_name;

        // Auto-generate output filename if not explicitly provided
        std::string final_output_file_name = output_file_name_override;
        if (final_output_file_name.empty()) {
            // Extract the base video name (remove file extension if present)
            std::string video_name = input_file_name;
            size_t last_dot = video_name.find_last_of(".");
            if (last_dot != std::string::npos) {
                video_name = video_name.substr(0, last_dot);
            }

            // Determine projection string for filename
            std::string proj_str;
            switch(output_projection_type) {
                case EncodingProjection::ERP: proj_str = "ERP"; break;
                case EncodingProjection::EAP:  proj_str = "EAP";  break;
                case EncodingProjection::CMP:  proj_str = "CMP";  break;
                case EncodingProjection::EAC:  proj_str = "EAC";  break;
                case EncodingProjection::SFLP: proj_str = "SFLP"; break;
                default:                       proj_str = "UNKNOWN"; break;
            }

            // Build format: {EncodingProjection}-{VideoName}-{output_num_pixels}
            final_output_file_name = proj_str + "-" + video_name + "-" + std::to_string(output_num_pixels);
            
            // Append ArrangementType if SFLP is selected
            if (output_projection_type == EncodingProjection::SFLP) {
                final_output_file_name += "-STANDARD";
            }
            
            final_output_file_name += ".mp4";
        }

        // Construct full output path
        std::string output_video_path = output_dir + "/" + final_output_file_name;

        // Auto-detect resolution and framerate via ffprobe if not explicitly overridden
        if (width <= 0 || height <= 0) {
            std::cout << "Probing input video dimensions using ffprobe...\n";
            if (!probe_video_metadata(ffprobe_path, input_video_path, width, height, fps, duration)) {
                std::cerr << "Warning: ffprobe auto-detection failed. Falling back to default dimensions (8192x4096).\n";
                if (width <= 0) width = 8192;
                if (height <= 0) height = 4096;
            }
        }

        std::cout << "Processing input video: " << input_video_path << "\n";
        std::cout << "Detected Source Resolution: " << width << "x" << height << "\n";
        std::cout << "Saving output H.265 video: " << output_video_path << "\n";
        std::cout << "Target Framerate: " << fps << " FPS | Preset: " << preset << "\n";

        ProjectionMetadata input_metadata = input_metadata_path.empty() ? 
            ProjectionMetadata(EncodingProjection::ERP, 180000000) : 
            load_metadata_from_json(input_metadata_path);

        // TIME TRACKER: Initial Pixel Arrangement
        auto arrangement_start = std::chrono::high_resolution_clock::now();

        PixelMatrix<int> arranged_layout;
        int out_w = 0;
        int out_h = 0;
        int total_layout_pixels = 0;
        int face_size = 0;

        if (output_projection_type == EncodingProjection::SFLP)
        { 
            arranged_layout = generate_SFLP_arrangement_standard_gpu(
                output_num_pixels, 
                padding_size
            );

            out_w = arranged_layout.cols;
            out_h = arranged_layout.rows;
            total_layout_pixels = out_h * out_w;
        } 
        else if (output_projection_type == EncodingProjection::EAC || output_projection_type == EncodingProjection::CMP)
        {
            // Square face dimensions: width = face_size * 3, height = face_size * 2
            face_size = static_cast<int>(std::floor(std::sqrt(output_num_pixels / 6.0)));
            int rem = face_size % padding_size;
            if (rem != 0) {
                face_size -= rem;
            }

            out_w = face_size * 3;
            out_h = face_size * 2;
            total_layout_pixels = out_h * out_w;
        }
        else {
            throw std::runtime_error("Unsupported output projection type specified for video conversion.");
        }


        auto arrangement_end = std::chrono::high_resolution_clock::now();
        double arrangement_duration_ms = std::chrono::duration<double, std::milli>(arrangement_end - arrangement_start).count();
        std::cout << "[Performance Timer] Initial Pixel Arrangement Time: " << arrangement_duration_ms << " ms\n";

        std::cout << "Output video dimensions: " << out_w << "x" << out_h << "\n";

        // Construct FFmpeg pipes...
        std::string decode_cmd = ffmpeg_path + " -i \"" + input_video_path + "\" -f rawvideo -pix_fmt rgb24 pipe:1";
        std::string encode_cmd = ffmpeg_path + " -y -f rawvideo -pix_fmt rgb24 -s " + 
                                std::to_string(out_w) + "x" + std::to_string(out_h) + 
                                " -r " + std::to_string(fps) + 
                                " -i pipe:0 -c:v libx265 -preset " + preset + " -b:v " + std::to_string(bitrate) + 
                                "k -maxrate " + std::to_string(bitrate) + 
                                "k -bufsize " + std::to_string(10 * bitrate) + 
                                "k -pix_fmt yuv420p -movflags +faststart -threads 8 \"" + 
                                output_video_path + "\"";

        #if defined(_WIN32)
        FILE* pipe_in = _popen(decode_cmd.c_str(), "rb");
        FILE* pipe_out = _popen(encode_cmd.c_str(), "wb");
        #else
        FILE* pipe_in = popen(decode_cmd.c_str(), "r");
        FILE* pipe_out = popen(encode_cmd.c_str(), "w");
        #endif

        if (!pipe_in || !pipe_out) {
            throw std::runtime_error("Failed to open FFmpeg pipes for video processing.");
        }

        size_t frame_bytes_in = static_cast<size_t>(width) * height * 3;
        std::vector<unsigned char> frame_buffer(frame_bytes_in);

        // Persistent GPU Allocations
        int* d_indices = nullptr;
        double* d_lat = nullptr;
        double* d_lon = nullptr;
        unsigned char* d_erp_image_gpu = nullptr;
        unsigned char* d_output_rgb = nullptr;

        size_t index_bytes = total_layout_pixels * sizeof(int);
        size_t coord_bytes = total_layout_pixels * sizeof(double);
        size_t erp_image_bytes = frame_bytes_in * sizeof(unsigned char);
        size_t output_rgb_bytes = total_layout_pixels * 3 * sizeof(unsigned char);

        CUDA_CHECK(cudaMalloc(&d_indices, index_bytes));
        CUDA_CHECK(cudaMalloc(&d_lat, coord_bytes));
        CUDA_CHECK(cudaMalloc(&d_lon, coord_bytes));
        CUDA_CHECK(cudaMalloc(&d_erp_image_gpu, erp_image_bytes));
        CUDA_CHECK(cudaMalloc(&d_output_rgb, output_rgb_bytes));

        // Pre-compute geographic points
        if (output_projection_type == EncodingProjection::SFLP)
        { 
            // Upload static arrangement indices
            CUDA_CHECK(cudaMemcpy(d_indices, arranged_layout.data.data(), index_bytes, cudaMemcpyHostToDevice));
            compute_sfl_geographic_points_gpu(d_indices, total_layout_pixels, output_num_pixels, d_lat, d_lon);
        } 
        else if (output_projection_type == EncodingProjection::EAC || output_projection_type == EncodingProjection::CMP)
        {
            std::cout << "Total layout pixels: " << total_layout_pixels << ", face size: " << face_size << "\n";

            bool is_cmp = (output_projection_type == EncodingProjection::CMP);
            get_CubeMap_geographic_points_gpu(
                total_layout_pixels,
                face_size,
                is_cmp,
                d_lat,
                d_lon
            );

            // Allocate host memory for debugging/printing
            std::vector<double> h_lat(total_layout_pixels);
            std::vector<double> h_lon(total_layout_pixels);

            // Copy results from device to host
            CUDA_CHECK(cudaMemcpy(h_lat.data(), d_lat, total_layout_pixels * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_lon.data(), d_lon, total_layout_pixels * sizeof(double), cudaMemcpyDeviceToHost));
        }
        else {
            cudaFree(d_indices);
            cudaFree(d_lat);
            cudaFree(d_lon);
            cudaFree(d_erp_image_gpu);
            cudaFree(d_output_rgb);
            throw std::runtime_error("Unsupported output projection type specified for video conversion.");
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        PixelMatrix<unsigned char> output_matrix;
        output_matrix.rows = out_h;
        output_matrix.cols = out_w;
        output_matrix.data.resize(total_layout_pixels * 3);

        int frame_idx = 0;
        const int max_expected_frames = static_cast<int>(duration * fps);

        // Runtime execution tracking params
        float gpu_kernel_time = 0.0f;
        double total_sampling_gpu_kernel_time_ms = 0.0;
        double total_sampling_time_ms = 0.0;
        double total_encoding_time_ms = 0.0;

        while (true) {
            if (frame_idx >= max_expected_frames) {
                std::cout << "Reached max expected frame limit (" << max_expected_frames << "). Stopping input read.\n";
                break;
            }

            auto read_start = std::chrono::high_resolution_clock::now();
            size_t read_bytes = fread(frame_buffer.data(), 1, frame_bytes_in, pipe_in);
            auto read_end = std::chrono::high_resolution_clock::now();
            
            if (read_bytes != frame_bytes_in) {
                break; // End of video stream
            }

            std::cout << "Processing frame " << frame_idx++ << "...\n";

            // --- TIME TRACKER: Image Sampling / Conversion (Streamlined) ---
            auto sampling_start = std::chrono::high_resolution_clock::now();

            sample_erp_frame_gpu(
                frame_buffer,
                height,
                width,
                d_erp_image_gpu,
                d_lat,
                d_lon,
                d_output_rgb,
                total_layout_pixels,
                output_matrix.data,
                gpu_kernel_time,
                false
            );

            auto sampling_end = std::chrono::high_resolution_clock::now();
            double sampling_ms = std::chrono::duration<double, std::milli>(sampling_end - sampling_start).count();
            total_sampling_time_ms += sampling_ms;
            total_sampling_gpu_kernel_time_ms += gpu_kernel_time;

            // --- TIME TRACKER: Encoding/Writing ---
            auto write_start = std::chrono::high_resolution_clock::now();
            fwrite(output_matrix.data.data(), 1, output_matrix.data.size(), pipe_out);
            auto write_end = std::chrono::high_resolution_clock::now();
            double write_ms = std::chrono::duration<double, std::milli>((write_end - write_start) + (read_end - read_start)).count();
            total_encoding_time_ms += write_ms;
        }

        // Cleanup persistent GPU allocations
        CUDA_CHECK(cudaFree(d_indices));
        CUDA_CHECK(cudaFree(d_lat));
        CUDA_CHECK(cudaFree(d_lon));
        CUDA_CHECK(cudaFree(d_erp_image_gpu));
        CUDA_CHECK(cudaFree(d_output_rgb));

#if defined(_WIN32)
        _pclose(pipe_in);
        _pclose(pipe_out);
#else
        pclose(pipe_in);
        pclose(pipe_out);
#endif

        std::cout << "\n==============================\n";
        std::cout << "        PERFORMANCE SUMMARY     \n";
        std::cout << "==============================\n";
        std::cout << "Initial Pixel Arrangement: " << arrangement_duration_ms << " ms\n";
        std::cout << "Total Image Sampling Time (Including CPU - GPU transfers): " << total_sampling_time_ms << " ms\n";
        std::cout << "Total Image Sampling Time (GPU kernel only): " << gpu_kernel_time << " ms\n";
        std::cout << "Total Encoding (After Sampling) Time: " << total_encoding_time_ms << " ms\n";
        std::cout << "Successfully generated H.265 SFLP video at: " << output_video_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}