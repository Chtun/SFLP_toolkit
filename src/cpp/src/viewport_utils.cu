#include <cmath>

#include "../include/viewport_utils.hpp"
#include "../include/file_conversion_utils.hpp"

#define M_PI 3.14159265358979323846

// CUDA Macro for error checking
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)
#endif

__global__ void viewport_geographic_kernel(
    double ang_res_h,
    double ang_res_v,
    int res_w,
    int res_h,
    double lat_center_rad,
    double lon_center_rad,
    double* __restrict__ lat_deg_out,
    double* __restrict__ lon_deg_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = res_w * res_h;
    if (idx >= total_pixels) return;

    int x_idx = idx % res_w;
    int y_idx = idx / res_w;

    // 1. Convert total field of view to radians
    double fov_h_rad = ang_res_h * (M_PI / 180.0);
    double fov_v_rad = ang_res_v * (M_PI / 180.0);

    // 2. Compute focal lengths based on image plane dimensions and FOV
    // f = (resolution / 2) / tan(FOV / 2)
    double f_x = (res_w > 1 ? (double)(res_w - 1) : 1.0) / (2.0 * tan(fov_h_rad * 0.5));
    double f_y = (res_h > 1 ? (double)(res_h - 1) : 1.0) / (2.0 * tan(fov_v_rad * 0.5));

    // Center coordinates of the viewport image
    double c_x = (double)(res_w - 1) * 0.5;
    double c_y = (double)(res_h - 1) * 0.5;

    // 3. Map pixel index to normalized camera local ray directions (pinhole camera model)
    // Looking down +X in local camera space: X is forward, Y is right, Z is up (or standard graphics convention)
    double u = ((double)x_idx - c_x) / f_x;
    double v = (c_y - (double)y_idx) / f_y; // Invert Y so screen-space top maps to +Z

    // Local 3D ray vector on the image plane (Forward = 1.0, Right = u, Up = v)
    double x_3d = 1.0;
    double y_3d = u;
    double z_3d = v;

    // Normalize local ray vector
    double inv_len = 1.0 / sqrt(x_3d * x_3d + y_3d * y_3d + z_3d * z_3d);
    x_3d *= inv_len;
    y_3d *= inv_len;
    z_3d *= inv_len;

    // 4. Rotate ray into world space based on center orientation (Yaw/Pitch / Lat/Lon)
    // Assuming standard rotation convention: Pitch (lat_center_rad) around Y-axis, Yaw (lon_center_rad) around Z-axis
    
    // Rotate by latitude (pitch around Y axis)
    double x_3d_rot1 = x_3d * cos(lat_center_rad) - z_3d * sin(lat_center_rad);
    double y_3d_rot1 = y_3d;
    double z_3d_rot1 = x_3d * sin(lat_center_rad) + z_3d * cos(lat_center_rad);

    // Rotate by longitude (yaw around Z axis)
    double final_x = x_3d_rot1 * cos(lon_center_rad) - y_3d_rot1 * sin(lon_center_rad);
    double final_y = x_3d_rot1 * sin(lon_center_rad) + y_3d_rot1 * cos(lon_center_rad);
    double final_z = z_3d_rot1;

    // 5. Convert world ray back to lat/lon coordinates for SFLP mapping
    double new_lat = asin(fmax(-1.0, fmin(1.0, final_z))) * (180.0 / M_PI);
    double new_lon = atan2(final_y, final_x) * (180.0 / M_PI);

    lat_deg_out[idx] = new_lat;
    lon_deg_out[idx] = new_lon;
}

void get_viewport_geographic_points_gpu(
    std::pair<double, double> viewport_direction,
    std::pair<double, double> viewport_angular_resolution,
    std::pair<int, int> viewport_resolution,
    double* d_lat_out,
    double* d_lon_out
) {
    double lat_center = viewport_direction.first;
    double lon_center = viewport_direction.second;
    double ang_res_h = viewport_angular_resolution.first;
    double ang_res_v = viewport_angular_resolution.second;
    int res_w = viewport_resolution.first;
    int res_h = viewport_resolution.second;

    int total_pixels = res_w * res_h;
    int blockSize = 256;
    int numBlocks = (total_pixels + blockSize - 1) / blockSize;

    double lat_center_rad = lat_center * (M_PI / 180.0);
    double lon_center_rad = lon_center * (M_PI / 180.0);

    viewport_geographic_kernel<<<numBlocks, blockSize>>>(
        ang_res_h,
        ang_res_v,
        res_w,
        res_h,
        lat_center_rad,
        lon_center_rad,
        d_lat_out,
        d_lon_out
    );
    CUDA_CHECK(cudaGetLastError());
}


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
    bool verbose_debug
) {
    int res_w = viewport_resolution.first;
    int res_h = viewport_resolution.second;
    int total_pixels = res_w * res_h;

    size_t latlon_bytes = total_pixels * sizeof(double);

    double *d_lat = nullptr;
    double *d_lon = nullptr;

    CUDA_CHECK(cudaMalloc(&d_lat, latlon_bytes));
    CUDA_CHECK(cudaMalloc(&d_lon, latlon_bytes));

    // Generate viewport geographic coordinates on the GPU
    get_viewport_geographic_points_gpu(
        viewport_direction,
        viewport_angular_resolution,
        viewport_resolution,
        d_lat,
        d_lon
    );

    // Delegate based on ProjectionType from ProjectionMetadata
    if (metadata.projection_type == EncodingProjection::SFLP) {
        const auto& sflp_metadata = dynamic_cast<const SFLPMetadata&>(metadata);

        // Call sample_SFLP_gpu directly targeting the output device buffer,
        // eliminating intermediate host transfers.
        sample_SFLP_gpu(
            d_lat,
            d_lon,
            total_pixels,
            d_input_image,
            img_height,
            img_width,
            img_channels,
            sflp_metadata.num_pixels,
            sflp_metadata.arrangement_type,
            sflp_metadata.optimal_index_hops,
            sflp_metadata.primary_reference_SFLP_index,
            sflp_metadata.secondary_reference_SFLP_index,
            sflp_metadata.non_padded_col_length,
            sflp_metadata.non_padded_row_length,
            sample_type,
            d_output_rgb,
            gpu_kernel_time,
            enable_timing,
            verbose_debug
        );
    } else if (metadata.projection_type == EncodingProjection::CMP || metadata.projection_type == EncodingProjection::EAC) {
        bool is_CMP = (metadata.projection_type == EncodingProjection::CMP);
        sample_CubeMap_gpu(
            d_lat,
            d_lon,
            total_pixels,
            d_input_image,
            img_height,
            img_width,
            img_channels,
            is_CMP,
            sample_type,
            d_output_rgb,
            gpu_kernel_time,
            enable_timing
        );
    } else {
        cudaFree(d_lat);
        cudaFree(d_lon);
        throw std::runtime_error("Unsupported projection type specified for viewport sampling.");
    }

    // Clean up device memory
    cudaFree(d_lat);
    cudaFree(d_lon);
}