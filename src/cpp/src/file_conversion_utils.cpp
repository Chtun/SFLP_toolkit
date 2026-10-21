#include "../include/file_conversion_utils.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdio>
#include <chrono>

// Macro for standard CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

void compute_sfl_geographic_points_cpu(
    const std::vector<int>& indices, 
    int num_points, 
    std::vector<double>& lat_out, 
    std::vector<double>& lon_out
) {
    int N = (num_points - 1) / 2;
    lat_out.resize(indices.size());
    lon_out.resize(indices.size());
    
    double phi = golden_ratio();

    for (size_t idx = 0; idx < indices.size(); ++idx) {
        int i_val = indices[idx] - N;
        double denominator = static_cast<double>(2 * N + 1);
        double lat_rad = std::asin((2.0 * i_val) / denominator);


        double rem = std::fmod(static_cast<double>(i_val), phi);
        if (rem < 0.0) rem += phi;
        double lon_deg = (rem * 360.0) / phi;

        double lat_deg = lat_rad * (180.0 / 3.14159265358979323846);
        lon_deg = std::fmod(lon_deg + 180.0, 360.0);
        if (lon_deg < 0.0) lon_deg += 360.0;
        lon_deg -= 180.0;

        lat_out[idx] = lat_deg;
        lon_out[idx] = lon_deg;
    }
}

std::pair<double, double> parse_double_pair(const std::string& str) {
    size_t comma = str.find(',');
    if (comma == std::string::npos) {
        throw std::runtime_error("Invalid pair format. Expected comma-separated values.");
    }
    return std::make_pair(std::stod(str.substr(0, comma)), std::stod(str.substr(comma + 1)));
}

std::pair<int, int> parse_int_pair(const std::string& str) {
    size_t comma = str.find(',');
    if (comma == std::string::npos) {
        throw std::runtime_error("Invalid pair format. Expected comma-separated values.");
    }
    return std::make_pair(std::stoi(str.substr(0, comma)), std::stoi(str.substr(comma + 1)));
}

// Helper functions for JSON parsing and metadata loading
std::string parse_json_string(const std::string& content, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t key_pos = content.find(search_key);
    if (key_pos == std::string::npos) return "";

    size_t colon_pos = content.find(':', key_pos);
    if (colon_pos == std::string::npos) return "";

    size_t start_quote = content.find('"', colon_pos);
    if (start_quote == std::string::npos) return "";

    size_t end_quote = content.find('"', start_quote + 1);
    if (end_quote == std::string::npos) return "";

    return content.substr(start_quote + 1, end_quote - start_quote - 1);
}

int parse_json_int(const std::string& content, const std::string& key, int default_val) {
    std::string search_key = "\"" + key + "\"";
    size_t key_pos = content.find(search_key);
    if (key_pos == std::string::npos) return default_val;

    size_t colon_pos = content.find(':', key_pos);
    if (colon_pos == std::string::npos) return default_val;

    size_t start_idx = colon_pos + 1;
    while (start_idx < content.size() && (content[start_idx] == ' ' || content[start_idx] == '\t' || content[start_idx] == '\n' || content[start_idx] == '\r')) {
        start_idx++;
    }

    size_t end_idx = start_idx;
    while (end_idx < content.size() && ((content[end_idx] >= '0' && content[end_idx] <= '9') || content[end_idx] == '-')) {
        end_idx++;
    }

    if (start_idx == end_idx) return default_val;
    return std::stoi(content.substr(start_idx, end_idx - start_idx));
}

std::string read_file_to_string(const std::string& filepath) {
    std::ifstream t(filepath);
    if (!t.is_open()) {
        throw std::runtime_error("Failed to open configuration JSON file: " + filepath);
    }
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

std::pair<int, int> parse_json_hops(const std::string& content) {
    size_t key_pos = content.find("\"OptimalIndexHops\"");
    if (key_pos == std::string::npos) return {1, 0};

    size_t bracket_pos = content.find('[', key_pos);
    if (bracket_pos == std::string::npos) return {1, 0};

    size_t closing_bracket = content.find(']', bracket_pos);
    if (closing_bracket == std::string::npos) return {1, 0};

    std::string array_content = content.substr(bracket_pos + 1, closing_bracket - bracket_pos - 1);
    std::stringstream ss(array_content);
    std::string val1_str, val2_str;

    if (std::getline(ss, val1_str, ',') && std::getline(ss, val2_str, ',')) {
        auto clean = [](std::string s) {
            s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
            s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
            return std::stoi(s);
        };
        return {clean(val1_str), clean(val2_str)};
    }
    return {1, 0};
}

ProjectionMetadata load_metadata_from_json(const std::string& json_path, int fallback_num_pixels) {
    std::string content = read_file_to_string(json_path);
    std::string proj_str = parse_json_string(content, "EncodingProjection");
    int num_pixels = parse_json_int(content, "num_pixels", fallback_num_pixels);

    if (proj_str == "ERP") {
        return ProjectionMetadata(EncodingProjection::ERP, num_pixels);
    } else if (proj_str == "EAP") {
        return ProjectionMetadata(EncodingProjection::EAP, num_pixels);
    } else if (proj_str == "CMP") {
        return ProjectionMetadata(EncodingProjection::CMP, num_pixels);
    } else if (proj_str == "EAC") {
        return ProjectionMetadata(EncodingProjection::EAC, num_pixels);
    } else if (proj_str == "SFLP") {
        std::string arrangement_str = parse_json_string(content, "ArrangementType");
        
        SFLPArrangementType arr_type = SFLPArrangementType::STANDARD;
        if (arrangement_str == "SEAM_REMOVED") arr_type = SFLPArrangementType::SEAM_REMOVED;
        else if (arrangement_str == "ZIGZAG") arr_type = SFLPArrangementType::ZIGZAG;

        std::pair<int, int> hops_values = parse_json_hops(content);

        int first_idx = parse_json_int(content, "primary_reference_SFLP_index", 0);
        int last_idx = parse_json_int(content, "secondary_reference_SFLP_index", num_pixels - 1);
        int padding_size = parse_json_int(content, "padding_size", 16);

        return SFLPMetadata(num_pixels, arr_type, hops_values, first_idx, last_idx, padding_size);
    } else {
        throw std::runtime_error("Unsupported or missing EncodingProjection in JSON: " + proj_str);
    }
}

std::unique_ptr<ProjectionMetadata> load_metadata_ptr_from_json(const std::string& json_path, int fallback_num_pixels) {
    std::string content = read_file_to_string(json_path);
    std::string proj_str = parse_json_string(content, "EncodingProjection");

    if (proj_str == "ERP") {
        return std::make_unique<ProjectionMetadata>(EncodingProjection::ERP, fallback_num_pixels);
    } else if (proj_str == "EAP") {
        return std::make_unique<ProjectionMetadata>(EncodingProjection::EAP, fallback_num_pixels);
    } else if (proj_str == "CMP") {
        return std::make_unique<ProjectionMetadata>(EncodingProjection::CMP, fallback_num_pixels);
    } else if (proj_str == "EAC") {
        return std::make_unique<ProjectionMetadata>(EncodingProjection::EAC, fallback_num_pixels);
    } else if (proj_str == "SFLP") {
        int num_pixels = parse_json_int(content, "num_pixels", fallback_num_pixels);
        std::string arrangement_str = parse_json_string(content, "ArrangementType");
        
        SFLPArrangementType arr_type = SFLPArrangementType::STANDARD;
        if (arrangement_str == "SEAM_REMOVED") arr_type = SFLPArrangementType::SEAM_REMOVED;
        else if (arrangement_str == "ZIGZAG") arr_type = SFLPArrangementType::ZIGZAG;

        std::pair<int, int> hops_values = parse_json_hops(content);

        int first_idx = parse_json_int(content, "primary_reference_SFLP_index", 0);
        int last_idx = parse_json_int(content, "secondary_reference_SFLP_index", num_pixels - 1);
        int padding_size = parse_json_int(content, "padding_size", 16);

        return std::make_unique<SFLPMetadata>(num_pixels, arr_type, hops_values, first_idx, last_idx, padding_size);
    } else {
        throw std::runtime_error("Unsupported or missing EncodingProjection in JSON: " + proj_str);
    }
}

// Helper to query video dimensions and fps automatically using ffprobe
bool probe_video_metadata(const std::string& ffprobe_path, const std::string& video_path, int& width, int& height, int& fps, double& duration) {
    std::string cmd = ffprobe_path + " -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1 \"" + video_path + "\"";
    
#if defined(_WIN32)
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return false;

    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    std::stringstream ss(result);
    std::string line;
    while (std::getline(ss, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        // trim carriage returns if any
        if (!val.empty() && val.back() == '\r') val.pop_back();

        if (key == "width") {
            width = std::stoi(val);
        } else if (key == "height") {
            height = std::stoi(val);
        } else if (key == "r_frame_rate") {
            size_t slash = val.find('/');
            if (slash != std::string::npos) {
                double num = std::stod(val.substr(0, slash));
                double den = std::stod(val.substr(slash + 1));
                if (den > 0) fps = static_cast<int>(num / den + 0.5);
            }
        } else if (key == "duration") {
            try {
                duration = std::stod(val);
            } catch (...) {
                // Fallback if duration is missing or "N/A" in stream headers
            }
        }
    }

    // Fallback: if stream-level duration isn't available, check container format duration
    if (duration <= 0.0) {
        std::string format_cmd = ffprobe_path + " -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"" + video_path + "\"";
#if defined(_WIN32)
        FILE* fpipe = _popen(format_cmd.c_str(), "r");
#else
        FILE* fpipe = popen(format_cmd.c_str(), "r");
#endif
        if (fpipe) {
            char fbuffer[128];
            std::string fresult = "";
            while (fgets(fbuffer, sizeof(fbuffer), fpipe) != nullptr) {
                fresult += fbuffer;
            }
#if defined(_WIN32)
            _pclose(fpipe);
#else
            pclose(fpipe);
#endif
            if (!fresult.empty()) {
                try {
                    duration = std::stod(fresult);
                } catch (...) {}
            }
        }
    }

    return true;
}

// Image conversion function supporting ERP, EAP, CMP, EAC< and SFLP with preloaded GPU coordinates
// Unified GPU image projection conversion function preloading coordinates internally
PixelMatrix<unsigned char> convert_image_projection_gpu(
    const std::string& input_image_path,
    EncodingProjection target_projection,
    int output_num_pixels,
    int padding_size
) {
    // Load input ERP image via stb_image
    int erp_w, erp_h, channels;
    unsigned char* loaded_data = stbi_load(input_image_path.c_str(), &erp_w, &erp_h, &channels, 3);
    if (!loaded_data) {
        throw std::runtime_error("Failed to load input image from path: " + input_image_path);
    }

    std::vector<unsigned char> h_erp_image(loaded_data, loaded_data + (erp_w * erp_h * 3));
    stbi_image_free(loaded_data);

    std::cout << "Loaded Source Resolution: " << erp_w << "x" << erp_h << "\n";

    // Determine layout dimensions and precompute coordinates
    PixelMatrix<int> arranged_layout;
    int out_w = 0;
    int out_h = 0;
    int total_layout_pixels = 0;
    int face_size = 0;

    // Runtime execution tracking params
    float gpu_kernel_time = 0.0f;
    auto arrangement_start = std::chrono::high_resolution_clock::now();

    if (target_projection == EncodingProjection::SFLP) {
        arranged_layout = generate_SFLP_arrangement_standard_gpu(output_num_pixels, padding_size);
        out_w = arranged_layout.cols;
        out_h = arranged_layout.rows;
        total_layout_pixels = out_h * out_w;
    } 
    else if (target_projection == EncodingProjection::EAC || target_projection == EncodingProjection::CMP) {
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
        throw std::runtime_error("Unsupported target projection type.");
    }

    auto arrangement_end = std::chrono::high_resolution_clock::now();
    double arrangement_duration_ms = std::chrono::duration<double, std::milli>(arrangement_end - arrangement_start).count();
    std::cout << "[Performance Timer] Initial Pixel Arrangement Time: " << arrangement_duration_ms << " ms\n";
    std::cout << "Output image dimensions: " << out_w << "x" << out_h << "\n";

#ifdef USE_CUDA
    int* d_indices = nullptr;
    double* d_lat = nullptr;
    double* d_lon = nullptr;
    unsigned char* d_erp_image_gpu = nullptr;
    unsigned char* d_output_rgb = nullptr;

    size_t index_bytes = total_layout_pixels * sizeof(int);
    size_t coord_bytes = total_layout_pixels * sizeof(double);
    size_t erp_image_bytes = h_erp_image.size() * sizeof(unsigned char);
    size_t output_rgb_bytes = total_layout_pixels * 3 * sizeof(unsigned char);

    CUDA_CHECK(cudaMalloc(&d_indices, index_bytes));
    CUDA_CHECK(cudaMalloc(&d_lat, coord_bytes));
    CUDA_CHECK(cudaMalloc(&d_lon, coord_bytes));
    CUDA_CHECK(cudaMalloc(&d_erp_image_gpu, erp_image_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_rgb, output_rgb_bytes));

    // Upload input image once to GPU
    CUDA_CHECK(cudaMemcpy(d_erp_image_gpu, h_erp_image.data(), erp_image_bytes, cudaMemcpyHostToDevice));

    // Pre-compute geographic points on GPU
    auto coord_start = std::chrono::high_resolution_clock::now();
    if (target_projection == EncodingProjection::SFLP) {
        CUDA_CHECK(cudaMemcpy(d_indices, arranged_layout.data.data(), index_bytes, cudaMemcpyHostToDevice));
        compute_sfl_geographic_points_gpu(d_indices, total_layout_pixels, output_num_pixels, d_lat, d_lon);
    } 
    else if (target_projection == EncodingProjection::EAC || target_projection == EncodingProjection::CMP) {
        bool is_cmp = target_projection == EncodingProjection::CMP;

        get_CubeMap_geographic_points_gpu(
            total_layout_pixels,
            face_size,
            is_cmp,
            d_lat,
            d_lon
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto coord_end = std::chrono::high_resolution_clock::now();
    double coord_duration_ms = std::chrono::duration<double, std::milli>(coord_end - coord_start).count();
    std::cout << "[Performance Timer] Coordinate Precomputation Time: " << coord_duration_ms << " ms\n";

    // Run GPU Sampling / Conversion Kernel
    PixelMatrix<unsigned char> output_matrix;
    output_matrix.rows = out_h;
    output_matrix.cols = out_w;
    output_matrix.data.resize(total_layout_pixels * 3);

    auto sampling_start = std::chrono::high_resolution_clock::now();

    sample_erp_frame_gpu(
        h_erp_image,
        erp_h,
        erp_w,
        d_erp_image_gpu,
        d_lat,
        d_lon,
        d_output_rgb,
        total_layout_pixels,
        output_matrix.data,
        gpu_kernel_time,
        true
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    auto sampling_end = std::chrono::high_resolution_clock::now();
    double sampling_ms = std::chrono::duration<double, std::milli>(sampling_end - sampling_start).count();

    // Download result back to host
    CUDA_CHECK(cudaMemcpy(
        output_matrix.data.data(), 
        d_output_rgb, 
        output_rgb_bytes, 
        cudaMemcpyDeviceToHost
    ));

    // Cleanup persistent GPU allocations
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_lat));
    CUDA_CHECK(cudaFree(d_lon));
    CUDA_CHECK(cudaFree(d_erp_image_gpu));
    CUDA_CHECK(cudaFree(d_output_rgb));

    std::cout << "\n==============================\n";
    std::cout << "        PERFORMANCE SUMMARY     \n";
    std::cout << "==============================\n";
    std::cout << "Initial Pixel Arrangement: " << arrangement_duration_ms << " ms\n";
    std::cout << "Coordinate Precomputation: " << coord_duration_ms << " ms\n";
    std::cout << "GPU Sampling / Conversion: " << sampling_ms << " ms\n";

    return output_matrix;
#else
    throw std::runtime_error("CUDA is disabled. Enable USE_CUDA to run GPU pipeline execution.");
#endif
}


// Single-frame GPU processing pipeline
void sample_erp_frame_gpu(
    const std::vector<unsigned char>& h_erp_image,
    int erp_h,
    int erp_w,
    unsigned char* d_erp_image_gpu,
    double* d_lat,
    double* d_lon,
    unsigned char* d_output_rgb,
    int total_layout_pixels,
    std::vector<unsigned char>& h_output_rgb,
    float& gpu_kernel_time,
    bool enable_timing
) {
#ifdef USE_CUDA
    cudaEvent_t start, stop;
    if (enable_timing) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    float elapsed_ms = 0.0f;

    // Upload current video frame to pre-allocated GPU memory
    if (enable_timing) CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_erp_image_gpu, h_erp_image.data(), h_erp_image.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    if (enable_timing) {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "[Timing] Frame H2D Memcpy: " << elapsed_ms << " ms\n";
    }

    // ERP-to-SFLP Conversion Kernel (using pre-computed d_lat, d_lon)
    if (enable_timing) CUDA_CHECK(cudaEventRecord(start));
    convert_ERP_projection_gpu(d_lat, d_lon, total_layout_pixels, d_erp_image_gpu, erp_h, erp_w, d_output_rgb);
    CUDA_CHECK(cudaGetLastError());
    if (enable_timing) {
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_time, start, stop));
        std::cout << "[Timing] convert_ERP_projection_gpu kernel: " << gpu_kernel_time << " ms\n";
    }

    // Download resulting frame back to host
    if (enable_timing) CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_output_rgb.data(), d_output_rgb, total_layout_pixels * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    if (enable_timing) {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "[Timing] D2H Memcpy: " << elapsed_ms << " ms\n";
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
#else
    throw std::runtime_error("CUDA is disabled. Enable USE_CUDA to run GPU pipeline execution.");
#endif
}