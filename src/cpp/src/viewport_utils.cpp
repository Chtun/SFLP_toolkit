#include "../include/viewport_utils.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <nlohmann/json.hpp>
#include <cmath>

// Macro for standard CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)


// Function 1: Parses JSON and builds the O(1) dictionary lookup structure
ViewportConfig parse_viewport_config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open viewport config file: " + config_path.string());
    }

    nlohmann::json j;
    file >> j;

    ViewportConfig config;
    config.start_frame = j.value("start_frame", 0);
    config.end_frame = j.value("end_frame", 10);

    auto res = j.at("viewport_resolution");
    config.viewport_resolution = {res[0].get<int>(), res[1].get<int>()};

    auto ang_res = j.at("viewport_angular_resolution");
    config.viewport_angular_resolution = {ang_res[0].get<double>(), ang_res[1].get<double>()};

    // Read raw specified direction change rules
    std::vector<std::tuple<int, double, double>> raw_directions;
    for (const auto& dir : j.at("viewport_directions")) {
        raw_directions.emplace_back(dir[0].get<int>(), dir[1].get<double>(), dir[2].get<double>());
    }

    // Build the O(1) frame lookup dictionary covering every frame from start_frame to end_frame
    int total_frames = (config.end_frame - config.start_frame) + 1;
    config.frame_direction_lookup.resize(total_frames);

    std::pair<double, double> current_dir = {0.0, 0.0};
    if (!raw_directions.empty()) {
        current_dir = {std::get<1>(raw_directions[0]), std::get<2>(raw_directions[0])};
    }

    size_t rule_idx = 0;
    for (int i = 0; i < total_frames; ++i) {
        int absolute_frame = config.start_frame + i;

        // Advance rule pointer if we hit a new specified starting frame block
        while (rule_idx < raw_directions.size() && std::get<0>(raw_directions[rule_idx]) <= absolute_frame) {
            current_dir = {std::get<1>(raw_directions[rule_idx]), std::get<2>(raw_directions[rule_idx])};
            ++rule_idx;
        }

        config.frame_direction_lookup[i] = current_dir;
    }

    return config;
}

// Sample a viewport from a specific frame of a video using FFmpeg and return the frame as a vector of RGB values
std::vector<uint8_t> sample_video_frame_viewport_gpu(
    const std::filesystem::path& video_path,
    const ProjectionMetadata& video_metadata,
    int frame_number,
    std::pair<double, double> viewport_direction,
    std::pair<double, double> viewport_angular_resolution,
    std::pair<int, int> viewport_resolution,
    SamplerResampling sample_type,
    const std::string& ffmpeg_path,
    const std::string& ffprobe_path,
    bool runtime_debug
) {
    if (frame_number < 0) {
        throw std::out_of_range("Frame number must be non-negative.");
    }

    // Auto-detect source video resolution and framerate via ffprobe
    int width = 0;
    int height = 0;
    int fps = 25;
    double duration = 1.0;
    if (!probe_video_metadata(ffprobe_path, video_path.string(), width, height, fps, duration)) {
        if (width <= 0) width = 8192;   // Fallback width
        if (height <= 0) height = 4096; // Fallback height
        std::cout << "Failed to probe video metadata! Falling back to default values.\n";
    }

    // Calculate exact timestamp for the requested frame
    double timestamp = static_cast<double>(frame_number) / static_cast<double>(fps);

    // Construct FFmpeg command to fast-seek (-ss) to the frame timestamp, decode exactly 1 frame, and pipe raw RGB24
    std::string decode_cmd = ffmpeg_path + " -ss " + std::to_string(timestamp) + 
                           " -i \"" + video_path.string() + 
                           "\" -frames:v 1 -f rawvideo -pix_fmt rgb24 pipe:1 2>nul";

    auto decode_start = std::chrono::high_resolution_clock::now();

#if defined(_WIN32)
    FILE* pipe = _popen(decode_cmd.c_str(), "rb");
#else
    FILE* pipe = popen(decode_cmd.c_str(), "r");
#endif

    if (!pipe) {
        throw std::runtime_error("Failed to open FFmpeg pipe for video: " + video_path.string());
    }

    size_t frame_bytes = static_cast<size_t>(width) * height * 3;
    std::vector<uint8_t> frame_rgb(frame_bytes);
    size_t bytes_read = fread(frame_rgb.data(), 1, frame_bytes, pipe);

#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    auto decode_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> decode_ms = decode_end - decode_start;

    if (bytes_read != frame_bytes) {
        throw std::runtime_error("Failed to read exact frame bytes from FFmpeg pipe for frame " + std::to_string(frame_number));
    }

    int res_w = viewport_resolution.first;
    int res_h = viewport_resolution.second;
    size_t output_frame_bytes = static_cast<size_t>(res_w) * res_h * 3;

    // Transfer frame data to GPU memory and allocate device output buffer
    uint8_t* d_input_image = nullptr;
    uint8_t* d_output_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_image, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_image, output_frame_bytes));
    CUDA_CHECK(cudaMemcpy(d_input_image, frame_rgb.data(), frame_bytes, cudaMemcpyHostToDevice));

    std::vector<uint8_t> h_output_rgb(res_w * res_h * 3);

    float gpu_kernel_time = 0.0f;

    auto sample_start = std::chrono::high_resolution_clock::now();

    sample_viewport_gpu(
        d_input_image,
        height,
        width,
        3,
        video_metadata,
        viewport_direction,
        viewport_angular_resolution,
        viewport_resolution,
        sample_type,
        d_output_image,
        gpu_kernel_time,
        runtime_debug
    );

    // Synchronize to capture exact GPU kernel/sampling execution time properly
    CUDA_CHECK(cudaDeviceSynchronize());

    auto sample_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> sample_ms = sample_end - sample_start;

    // Copy final output from device back to host vector
    CUDA_CHECK(cudaMemcpy(h_output_rgb.data(), d_output_image, output_frame_bytes, cudaMemcpyDeviceToHost));

    if (runtime_debug) {
        std::cout << "\n[Runtime Debug] Frame " << frame_number << " sample time (including Memcpy): " << sample_ms.count() << " ms\n";
        std::cout << "[Runtime Debug] Frame " << frame_number << " sample time (GPU kernels only): " << gpu_kernel_time << " ms\n";
        std::cout << "[Runtime Debug] Frame " << frame_number << " decode time: " << decode_ms.count() << " ms\n\n";
    }

    cudaFree(d_input_image);
    cudaFree(d_output_image);
    return h_output_rgb;
}

// Sample a viewport from a specific frame of a video and save it to disk
void save_video_frame_viewport_gpu(
    const std::filesystem::path& video_path,
    const ProjectionMetadata& video_metadata,
    int frame_number,
    std::pair<double, double> viewport_direction,
    std::pair<double, double> viewport_angular_resolution,
    std::pair<int, int> viewport_resolution,
    const std::filesystem::path& output_image_path,
    SamplerResampling sample_type,
    const std::string& ffmpeg_path,
    const std::string& ffprobe_path,
    bool runtime_debug
) {
    int res_w = viewport_resolution.first;
    int res_h = viewport_resolution.second;

    std::vector<uint8_t> viewport_frame_rgb = sample_video_frame_viewport_gpu(
        video_path,
        video_metadata,
        frame_number,
        viewport_direction,
        viewport_angular_resolution,
        viewport_resolution,
        sample_type,
        ffmpeg_path,
        ffprobe_path,
        runtime_debug
    );

    std::filesystem::create_directories(output_image_path.parent_path());

    // Save image using standard PNG/JPG writers (e.g., stb_image_write)
    int success = stbi_write_png(
        output_image_path.string().c_str(),
        res_w,
        res_h,
        3,
        viewport_frame_rgb.data(),
        res_w * 3
    );

    if (!success) {
        throw std::runtime_error("Failed to save viewport frame to " + output_image_path.string());
    }
    else {
        std::cout << "Viewport for frame " << std::to_string(frame_number) << " successfully saved to: " << output_image_path.string() << "\n";
    }

    // Free underlying CUDA memory caches
    cudaDeviceReset();
}

// Helper function to resolve the active viewport direction for a given frame
inline std::pair<double, double> get_active_direction(int current_frame, const ViewportConfig& config) {
    int index = current_frame - config.start_frame;
    if (index >= 0 && index < static_cast<int>(config.frame_direction_lookup.size())) {
        return config.frame_direction_lookup[index];
    }
    // Fallback if out of bounds
    return config.frame_direction_lookup.empty() ? std::pair<double, double>{0.0, 0.0} : config.frame_direction_lookup.back();
}

// Construct viewport video given specified configuration and input video and saves viewport video stream
void save_viewport_video_gpu(
    const ViewportConfig& config,
    const std::filesystem::path& video_path,
    const std::filesystem::path& output_video_path,
    const ProjectionMetadata& video_metadata,
    SamplerResampling sample_type,
    const std::string& ffmpeg_path,
    const std::string& ffprobe_path,
    bool runtime_debug,
    bool verbose_debug
) {
    int width = 0;
    int height = 0;
    int fps = 25;
    double duration = 1.0;
    if (!probe_video_metadata(ffprobe_path, video_path.string(), width, height, fps, duration)) {
        if (width <= 0) width = 8192;
        if (height <= 0) height = 4096;
    }

    int res_w = config.viewport_resolution.first;
    int res_h = config.viewport_resolution.second;
    size_t input_frame_bytes = static_cast<size_t>(width) * height * 3;
    size_t output_frame_bytes = static_cast<size_t>(res_w) * res_h * 3;

    double start_timestamp = static_cast<double>(config.start_frame) / static_cast<double>(fps);
    int total_frames_to_process = (config.end_frame - config.start_frame) + 1;

    // Use a fast local scratch file instead of live bi-directional pipes to eliminate Windows pipe backpressure stalls
    std::filesystem::path scratch_raw_path = output_video_path.parent_path() / "scratch_decoded_input.raw";

    // Step 1: Decode H.265 via hardware acceleration directly to a temporary raw file sink
    std::string decode_cmd = ffmpeg_path + " -c:v hevc_cuvid -hwaccel cuda -hwaccel_output_format cuda -ss " + std::to_string(start_timestamp) +
                           " -i \"" + video_path.string() +
                           "\" -frames:v " + std::to_string(total_frames_to_process) +
                           " -vf \"hwdownload,format=nv12,format=rgb24\" -f rawvideo -y \"" + scratch_raw_path.string() + "\" 2>nul";

    // Step 2: Set up output encoder pipe for H.265 or H.264 compression from file/stream
    std::string encode_cmd = ffmpeg_path + " -y -f rawvideo -vcodec rawvideo -s " +
                           std::to_string(res_w) + "x" + std::to_string(res_h) +
                           " -pix_fmt rgb24 -r " + std::to_string(fps) +
                           " -i pipe:0 -c:v libx265 -pix_fmt yuv420p \"" +
                           output_video_path.string() + "\" 2>nul";

    // Measure pure hardware decoding time for the entire batch
    auto decode_start = std::chrono::high_resolution_clock::now();
    int decode_status = std::system(decode_cmd.c_str());
    auto decode_end = std::chrono::high_resolution_clock::now();
    double total_hardware_decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

    if (decode_status != 0) {
        throw std::runtime_error("Failed to execute hardware-accelerated H.265 decoding step to scratch file.");
    }

    // Open raw decoded file for reading and output encoder pipe for writing
    FILE* file_in = fopen(scratch_raw_path.string().c_str(), "rb");
#if defined(_WIN32)
    FILE* pipe_out = _popen(encode_cmd.c_str(), "wb");
#else
    FILE* pipe_out = popen(encode_cmd.c_str(), "w");
#endif

    if (!file_in || !pipe_out) {
        if (file_in) fclose(file_in);
        if (pipe_out) {
#if defined(_WIN32)
            _pclose(pipe_out);
#else
            pclose(pipe_out);
#endif
        }
        throw std::runtime_error("Failed to open decoded raw input file or FFmpeg encoder pipe.");
    }

    uint8_t* h_pinned_frame_rgb = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pinned_frame_rgb, input_frame_bytes, cudaHostAllocDefault));
    
    std::vector<uint8_t> h_output_rgb(output_frame_bytes);

    uint8_t* d_input_image = nullptr;
    uint8_t* d_output_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_image, input_frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_image, output_frame_bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    double total_gpu_kernel_ms = 0.0;
    double total_sample_wrapper_ms = 0.0;
    int frames_processed = 0;
    
    float gpu_kernel_time = 0.0f;

    for (int i = 0; i < total_frames_to_process; ++i) {
        int current_frame_number = config.start_frame + i;

        size_t bytes_read = fread(h_pinned_frame_rgb, 1, input_frame_bytes, file_in);

        if (bytes_read != input_frame_bytes) {
            std::cout << "Warning: Reached end of input stream early at frame " << current_frame_number << " as only " << bytes_read << " bytes were read (expected: " << input_frame_bytes << " bytes).\n";
            break;
        }

        std::pair<double, double> active_direction = get_active_direction(current_frame_number, config);

        CUDA_CHECK(cudaMemcpyAsync(d_input_image, h_pinned_frame_rgb, input_frame_bytes, cudaMemcpyHostToDevice, stream));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);

        sample_viewport_gpu(
            d_input_image,
            height,
            width,
            3,
            video_metadata,
            active_direction,
            config.viewport_angular_resolution,
            config.viewport_resolution,
            sample_type,
            d_output_image,
            gpu_kernel_time,
            runtime_debug,
            verbose_debug
        );

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float sample_ms = 0;
        cudaEventElapsedTime(&sample_ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        CUDA_CHECK(cudaMemcpyAsync(h_output_rgb.data(), d_output_image, output_frame_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        fwrite(h_output_rgb.data(), 1, output_frame_bytes, pipe_out);

        total_gpu_kernel_ms += gpu_kernel_time;
        total_sample_wrapper_ms += sample_ms;
        frames_processed++;

        if (runtime_debug && verbose_debug) {
            std::cout << "[Runtime Debug] Frame " << current_frame_number 
                      << " | Direction (" << active_direction.first << ", " << active_direction.second << ")"
                      << " | Sample Time (GPU kernel only): " << gpu_kernel_time << " ms"
                      << " | Sample Time (Including Memcpy): " << sample_ms << " ms\n";
        }
    }

    if (file_in) {
        fclose(file_in);
        file_in = nullptr;
    }

#if defined(_WIN32)
    _pclose(pipe_out);
#else
    pclose(pipe_out);
#endif

    pipe_out = nullptr;

    std::error_code ec;
    std::filesystem::remove(scratch_raw_path, ec);
    if (ec && runtime_debug) {
        std::cout << "[Warning] Could not remove scratch file: " << ec.message() << "\n";
    }

    cudaFree(d_input_image);
    cudaFree(d_output_image);
    CUDA_CHECK(cudaFreeHost(h_pinned_frame_rgb));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "Successfully rendered H.265 viewport video sequence to: " << output_video_path.string() << "\n";

    if (runtime_debug) {
        double avg_decode_ms = frames_processed > 0 ? (total_hardware_decode_ms / frames_processed) : 0.0;
        std::cout << "\n========================================\n";
        std::cout << "        FINAL PERFORMANCE SUMMARY       \n";
        std::cout << "========================================\n";
        std::cout << " Total Frames Processed            : " << frames_processed << "\n";
        std::cout << " Total Hardware Decode Time        : " << total_hardware_decode_ms << " ms\n";
        std::cout << " Total GPU Kernel Time             : " << total_gpu_kernel_ms << " ms\n";
        std::cout << " Total Sample Wrapper Time         : " << total_sample_wrapper_ms << " ms\n";
        if (frames_processed > 0) {
            std::cout << "----------------------------------------\n";
            std::cout << " Average Decode Time / Frame       : " << avg_decode_ms << " ms\n";
            std::cout << " Average GPU Kernel Time / Frame   : " << (total_gpu_kernel_ms / frames_processed) << " ms\n";
            std::cout << " Average Wrapper Time / Frame      : " << (total_sample_wrapper_ms / frames_processed) << " ms\n";
        }
        std::cout << "========================================\n";
    }

    cudaDeviceReset();
}

// Processing a video viewport rendering and saving request.
void process_video_viewport_rendering(
    const std::filesystem::path& config_path,
    const std::filesystem::path& video_path,
    const std::filesystem::path& output_video_path,
    const ProjectionMetadata& video_metadata,
    SamplerResampling sample_type,
    const std::string& ffmpeg_path,
    const std::string& ffprobe_path,
    bool runtime_debug,
    bool verbose_debug
) {
    // Get the configuration struct
    ViewportConfig config = parse_viewport_config(config_path);

    // Render the viewport and save based on the 
    save_viewport_video_gpu(
        config,
        video_path,
        output_video_path,
        video_metadata,
        sample_type,
        ffmpeg_path,
        ffprobe_path,
        runtime_debug,
        verbose_debug
    );
}