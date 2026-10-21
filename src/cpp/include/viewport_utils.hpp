#ifndef VIEWPORT_UTILS_HPP
#define VIEWPORT_UTILS_HPP

#include <cuda_runtime.h>
#include <utility>
#include <cstdint>
#include <string>
#include <memory>
#include <stdexcept>
#include <filesystem>

#include "file_conversion_utils.hpp"
#include "pixel_arrangement_utils.hpp"

struct ViewportConfig {
    int start_frame;
    int end_frame;
    std::pair<int, int> viewport_resolution;
    std::pair<double, double> viewport_angular_resolution;
    
    std::vector<std::pair<double, double>> frame_direction_lookup;
};

// Samples a viewport from a specific frame of a video file
std::vector<uint8_t> sample_video_frame_viewport_gpu(
    const std::filesystem::path& video_path,
    const ProjectionMetadata& video_metadata,
    int frame_number,
    std::pair<double, double> viewport_direction,
    std::pair<double, double> viewport_angular_resolution,
    std::pair<int, int> viewport_resolution,
    SamplerResampling sample_type = SamplerResampling::BILINEAR,
    const std::string& ffmpeg_path = "ffmpeg",
    const std::string& ffprobe_path = "ffprobe"
);

// Samples a video frame viewport and writes the output image to disk
void save_video_frame_viewport_gpu(
    const std::filesystem::path& video_path,
    const ProjectionMetadata& video_metadata,
    int frame_number,
    std::pair<double, double> viewport_direction,
    std::pair<double, double> viewport_angular_resolution,
    std::pair<int, int> viewport_resolution,
    const std::filesystem::path& output_image_path,
    SamplerResampling sample_type = SamplerResampling::BILINEAR,
    const std::string& ffmpeg_path = "ffmpeg",
    const std::string& ffprobe_path = "ffprobe",
    bool runtime_debug = false
);

// Construct viewport video given specified configuration and input video and saves viewport video stream
void save_viewport_video_gpu(
    const ViewportConfig& config,
    const std::filesystem::path& video_path,
    const std::filesystem::path& output_video_path,
    const ProjectionMetadata& video_metadata,
    SamplerResampling sample_type,
    const std::string& ffmpeg_path = "ffmpeg",
    const std::string& ffprobe_path = "ffprobe",
    bool runtime_debug = false,
    bool verbose_debug = false
) ;

// Processing a video viewport rendering and saving request.
void process_video_viewport_rendering(
    const std::filesystem::path& config_path,
    const std::filesystem::path& video_path,
    const std::filesystem::path& output_video_path,
    const ProjectionMetadata& video_metadata,
    SamplerResampling sample_type,
    const std::string& ffmpeg_path = "ffmpeg",
    const std::string& ffprobe_path = "ffprobe",
    bool runtime_debug = false,
    bool verbose_debug = false
);

// CUDA wrapper declarations
#ifdef USE_CUDA

// Computes geographic points for a viewport on the GPU and stores them in device memory
void get_viewport_geographic_points_gpu(
    std::pair<double, double> viewport_direction,
    std::pair<double, double> viewport_angular_resolution,
    std::pair<int, int> viewport_resolution,
    double* d_lat_out,
    double* d_lon_out
);

// Samples a viewport from a device frame buffer
void sample_viewport_gpu(
    const uint8_t* d_input_image,
    int img_height,
    int img_width,
    int img_channels,
    const ProjectionMetadata& metadata,
    std::pair<double, double> viewport_direction,
    std::pair<double, double> viewport_angular_resolution,
    std::pair<int, int> viewport_resolution,
    SamplerResampling sample_type,
    uint8_t* d_output_rgb,
    float& gpu_kernel_time,
    bool enable_timing,
    bool verbose_debug = false
);


#endif

#endif // VIEWPORT_UTILS_HPP