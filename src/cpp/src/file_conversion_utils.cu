#include "../include/file_conversion_utils.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>


#define M_PI 3.14159265358979323846

// Macro for standard CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

__device__ constexpr double PI = 3.14159265358979323846;
__device__ constexpr double GOLDEN_RATIO = 1.61803398874989484820;
constexpr double SQRT_5 = 2.2360679774997896964091736687313;
constexpr double INV_SQRT_5 = 0.44721359549995879228;
constexpr double LOG_PHI = 0.48121182505960344700;
constexpr double LOG_PHI2 = 0.96242365011920689400;
constexpr double INV_LOG_PHI2 = 1.0390434606175137698754;

__device__ void get_SFL_vector_gpu(int index, int num_points, double& vx, double& vy, double& vz) {
    if (num_points % 2 == 0) {
        num_points -= 1;
    }

    int N = (num_points - 1) / 2;
    int i = index - N;

    double ratio = (2.0 * (double)i) / (2.0 * (double)N + 1.0);
    double lat_rad = asin(fmax(-1.0, fmin(ratio, 1.0)));
    double lat_deg = lat_rad * (180.0 / PI);

    double phi_golden = GOLDEN_RATIO;
    double rem = fmod((double)i, phi_golden);
    if (rem < 0.0) rem += phi_golden;
    double lon_deg = rem * (360.0 / phi_golden);

    lon_deg = fmod(lon_deg + 180.0, 360.0);
    if (lon_deg < 0.0) lon_deg += 360.0;
    lon_deg -= 180.0;

    double theta = lon_deg * (PI / 180.0);
    double phi = (90.0 - lat_deg) * (PI / 180.0); // Co-latitude conversion

    vx = sin(phi) * cos(theta);
    vy = sin(phi) * sin(theta);
    vz = cos(phi);
}

__global__ void sfl_geographic_kernel(
    const int* indices, 
    int num_indices, 
    int N_val, 
    int total_points, 
    double* lat_deg_out, 
    double* lon_deg_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices) {
        int index_val = indices[idx];
        int i_val = index_val - N_val;

        // lat_rad = asin(2 * i / (2 * N + 1))
        double denominator = static_cast<double>(2 * N_val + 1);
        double ratio = (2.0 * static_cast<double>(i_val)) / denominator;
        double lat_rad = asin(ratio);

        // lon_deg = ((i % phi) * 360.0 / phi)
        // Note: fmod handles floating point modulo for 'i % phi'
        double phi = GOLDEN_RATIO;
        double rem = fmod(static_cast<double>(i_val), phi);
        if (rem < 0.0) rem += phi; // match behavior if negative indices occur
        double lon_deg = (rem * 360.0) / phi;

        // Convert radians to degrees for latitude
        double lat_deg = lat_rad * (180.0 / PI);

        // Longitude normalization to range [-180, 180]
        lon_deg = fmod(lon_deg + 180.0, 360.0);
        if (lon_deg < 0.0) lon_deg += 360.0;
        lon_deg -= 180.0;

        lat_deg_out[idx] = lat_deg;
        lon_deg_out[idx] = lon_deg;
    }
}

extern "C" void compute_sfl_geographic_points_gpu(
    const int* d_indices, 
    int num_indices, 
    int num_points, 
    double* d_lat_out, 
    double* d_lon_out
) {
    int N = (num_points - 1) / 2;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_indices + threadsPerBlock - 1) / threadsPerBlock;

    sfl_geographic_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_indices, num_indices, N, num_points, d_lat_out, d_lon_out
    );
    
    cudaDeviceSynchronize();
}

// CUDA Kernel for Sampling ERP Image based on lat/lon coordinates
__global__ void erp_sampling_kernel(
    const double* d_lat,
    const double* d_lon,
    int total_pixels,
    const unsigned char* d_erp_image,
    int erp_h,
    int erp_w,
    unsigned char* d_output_rgb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        double lat = d_lat[idx]; // range: [-90, 90]
        double lon = d_lon[idx]; // range: [-180, 180]

        // Map lat/lon to ERP image UV coordinates
        // u = (lon + 180) / 360 * W
        // v = (90 - lat) / 180 * H
        double u = (lon + 180.0) / 360.0 * static_cast<double>(erp_w);
        double v = (90.0 - lat) / 180.0 * static_cast<double>(erp_h);

        // Nearest neighbor sampling coordinates
        int x = static_cast<int>(fmin(fmax(round(u), 0.0), static_cast<double>(erp_w - 1)));
        int y = static_cast<int>(fmin(fmax(round(v), 0.0), static_cast<double>(erp_h - 1)));

        int erp_pixel_idx = (y * erp_w + x) * 3; // RGB channels
        int out_pixel_idx = idx * 3;

        d_output_rgb[out_pixel_idx + 0] = d_erp_image[erp_pixel_idx + 0]; // R
        d_output_rgb[out_pixel_idx + 1] = d_erp_image[erp_pixel_idx + 1]; // G
        d_output_rgb[out_pixel_idx + 2] = d_erp_image[erp_pixel_idx + 2]; // B
    }
}

extern "C" void convert_ERP_projection_gpu(
    const double* d_lat,
    const double* d_lon,
    int total_pixels,
    const unsigned char* d_erp_image,
    int erp_h,
    int erp_w,
    unsigned char* d_output_rgb_out
) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    erp_sampling_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_lat, d_lon, total_pixels, d_erp_image, erp_h, erp_w, d_output_rgb_out
    );
    cudaDeviceSynchronize();
}

__device__ inline double madfrac(double a, double b) {
    return a * b - floor(a * b);
}

__device__ inline double normalize_theta(double theta) {
    return fmod(theta + PI, 2.0 * PI) - PI;
}

// Conversion kernel from lat/lon degrees to 3D unit vectors
__global__ void latlon_to_vectors_kernel(
    const double* __restrict__ lat_deg,
    const double* __restrict__ lon_deg,
    int num_points,
    double* __restrict__ vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    double lat_rad = lat_deg[idx] * (PI / 180.0);
    double lon_rad = lon_deg[idx] * (PI / 180.0);

    double cos_lat = cos(lat_rad);
    vectors[idx * 3 + 0] = cos_lat * cos(lon_rad);
    vectors[idx * 3 + 1] = cos_lat * sin(lon_rad);
    vectors[idx * 3 + 2] = sin(lat_rad);
}

__global__ void map_latlon_to_SFLP_indices_kernel(
    const double* __restrict__ lat_deg,
    const double* __restrict__ lon_deg,
    int num_pixels,
    int L,
    int* __restrict__ out_sampled_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;

    double lat_rad = lat_deg[idx] * (PI / 180.0);
    double lon_rad = lon_deg[idx] * (PI / 180.0);

    double z = sin(lat_rad);
    double theta = lon_rad;

    int k = max(2, (int)floor(
        log(num_pixels * PI * SQRT_5 * (1.0 - z * z)) * INV_LOG_PHI2
    ));

    double Fk = exp((double)k * LOG_PHI) * INV_SQRT_5;
    int F0 = (int)round(Fk);
    int F1 = (int)round(Fk * GOLDEN_RATIO);

    double b00 = 2.0 * PI * madfrac((double)F0 + 1.0, GOLDEN_RATIO - 1.0) - 2.0 * PI * (GOLDEN_RATIO - 1.0);
    double b01 = 2.0 * PI * madfrac((double)F1 + 1.0, GOLDEN_RATIO - 1.0) - 2.0 * PI * (GOLDEN_RATIO - 1.0);
    double b10 = 2.0 * (double)F0 / (double)num_pixels;
    double b11 = 2.0 * (double)F1 / (double)num_pixels;

    double det = b00 * b11 - b01 * b10;
    double inv_det = (fabs(det) > 1e-12) ? (1.0 / det) : 0.0;

    double invB_00 =  b11 * inv_det;
    double invB_01 = -b01 * inv_det;
    double invB_10 = -b10 * inv_det;
    double invB_11 =  b00 * inv_det;

    double c_u_init = normalize_theta(2.0 * PI * -1.0 * floor((double)num_pixels / 2.0) * (GOLDEN_RATIO - 1.0));
    double c_v_init = (1.0 / (double)num_pixels) - 1.0;
    double c_u = normalize_theta(theta - c_u_init);
    double c_v = z - c_v_init;

    double c_x = invB_00 * c_u + invB_01 * c_v;
    double c_y = invB_10 * c_u + invB_11 * c_v;
    int base_cx = (int)floor(c_x);
    int base_cy = (int)floor(c_y);

    double num_pixels_d = (double)num_pixels;
    double half_num_pixels = num_pixels_d * 0.5;
    double z_offset = (1.0 / num_pixels_d) - 1.0;

    int base_out_idx = idx * 4;

    // Fully unrolled calculations replacing local arrays entirely
    for (int i = 0; i < 4; ++i) {
        double cx_off = (double)(base_cx + (i & 1));
        double cy_off = (double)(base_cy + (i >> 1));

        double z_approx = (b10 * cx_off + b11 * cy_off) + z_offset;
        double z_clipped = fmax(-1.0, fmin(z_approx, 1.0));
        double z_reflected = z_clipped * 2.0 - z_approx;

        int s_idx = (int)floor(half_num_pixels + z_reflected * half_num_pixels);
        s_idx = max(0, min(s_idx, num_pixels - 1));
        
        out_sampled_indices[base_out_idx + i] = s_idx;
    }
}

// SFLP sampling helper kernel supporting NEAREST, BILINEAR, and IDW resampling types
__global__ void sample_SFLP_helper_kernel(
    const double* __restrict__ d_lat_deg,
    const double* __restrict__ d_lon_deg,
    const int* __restrict__ d_sampled_indices, 
    const int* __restrict__ d_rows_in,
    const int* __restrict__ d_cols_in,
    const uint8_t* __restrict__ input_image,
    int img_height,
    int img_width,
    int img_channels,
    int num_pixels,
    int L,
    SamplerResampling sample_type,
    uint8_t* __restrict__ out_r,
    uint8_t* __restrict__ out_g,
    uint8_t* __restrict__ out_b
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;

    // Fetch the 4 neighbor pixel values and coordinates
    int debug_rows[4];
    int debug_cols[4];
    double pixel_r[4], pixel_g[4], pixel_b[4];
    double angular_distances[4];

    // Compute target unit vector on the sphere from input lat/lon if needed for distance-based metrics
    double v_x = 0.0, v_y = 0.0, v_z = 0.0;
    if (sample_type == SamplerResampling::IDW || sample_type == SamplerResampling::BILINEAR) {
        double target_lat_rad = d_lat_deg[idx] * (PI / 180.0);
        double target_lon_rad = d_lon_deg[idx] * (PI / 180.0);
        double cos_target_lat = cos(target_lat_rad);
        v_x = cos_target_lat * cos(target_lon_rad);
        v_y = cos_target_lat * sin(target_lon_rad);
        v_z = sin(target_lat_rad);
    }

    for (int i = 0; i < 4; ++i) {
        int neighbor_flat_idx = idx * 4 + i;
        int s_idx = d_sampled_indices[neighbor_flat_idx];

        int row_idx = d_rows_in[neighbor_flat_idx];
        int col_idx = d_cols_in[neighbor_flat_idx];

        row_idx = max(0, min(row_idx, img_height - 1));
        col_idx = max(0, min(col_idx, img_width - 1));

        debug_rows[i] = row_idx;
        debug_cols[i] = col_idx;

        int img_pixel_offset = (row_idx * img_width + col_idx) * img_channels;
        pixel_r[i] = (double)input_image[img_pixel_offset + 0];
        pixel_g[i] = (double)input_image[img_pixel_offset + 1];
        pixel_b[i] = (double)input_image[img_pixel_offset + 2];

        if (sample_type == SamplerResampling::IDW || sample_type == SamplerResampling::BILINEAR) {
            double nx, ny, nz;
            get_SFL_vector_gpu(s_idx, num_pixels, nx, ny, nz);
            double dot_prod = fmax(-1.0, fmin(v_x * nx + v_y * ny + v_z * nz, 1.0));
            angular_distances[i] = acos(dot_prod);
        }
    }

    double accum_r = 0.0, accum_g = 0.0, accum_b = 0.0;

    if (sample_type == SamplerResampling::NEAREST) {
        // NEAREST: Pick the closest neighbor (index 0 by default convention or minimal angular distance if available)
        accum_r = pixel_r[0];
        accum_g = pixel_g[0];
        accum_b = pixel_b[0];
    } 
    else if (sample_type == SamplerResampling::BILINEAR) {
        // BILINEAR: Standard inverse angular weighting or linear weights based on angular proximity
        const double eps = 1e-6;
        double weights[4];
        double weight_sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            weights[i] = 1.0 / (angular_distances[i] + eps);
            weight_sum += weights[i];
        }
        for (int i = 0; i < 4; ++i) {
            double norm_weight = weights[i] / weight_sum;
            accum_r += norm_weight * pixel_r[i];
            accum_g += norm_weight * pixel_g[i];
            accum_b += norm_weight * pixel_b[i];
        }
    } 
    else if (sample_type == SamplerResampling::IDW) {
        // IDW: Inverse-distance weighting with epsilon matching Python (eps = 1e-6)
        const double eps = 1e-6;
        double inv_weights[4];
        double weight_sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            inv_weights[i] = 1.0 / (angular_distances[i] + eps);
            weight_sum += inv_weights[i];
        }
        for (int i = 0; i < 4; ++i) {
            double norm_weight = inv_weights[i] / weight_sum;
            accum_r += norm_weight * pixel_r[i];
            accum_g += norm_weight * pixel_g[i];
            accum_b += norm_weight * pixel_b[i];
        }
    }

    // Clip output to [0, 255]
    out_r[idx] = (uint8_t)fmax(0.0, fmin(255.0, accum_r));
    out_g[idx] = (uint8_t)fmax(0.0, fmin(255.0, accum_g));
    out_b[idx] = (uint8_t)fmax(0.0, fmin(255.0, accum_b));
}

__global__ void interleave_rgb_kernel(
    const uint8_t* d_r,
    const uint8_t* d_g,
    const uint8_t* d_b,
    int num_points,
    uint8_t* d_output_rgb
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_points) {
        d_output_rgb[idx * 3 + 0] = d_r[idx];
        d_output_rgb[idx * 3 + 1] = d_g[idx];
        d_output_rgb[idx * 3 + 2] = d_b[idx];
    }
}

// CUDA Wrapper for sampling on an SFLP frame using GPU acceleration
void sample_SFLP_gpu(
    const double* d_lat, const double* d_lon, int num_points_l,
    const uint8_t* d_input_image, int img_height, int img_width, int img_channels,
    int num_pixels, SFLPArrangementType arrangement_type,
    const std::pair<IndexHop, IndexHop>& optimal_index_hops,
    int primary_ref_SFLP_index, int secondary_ref_SFLP_index,
    int non_padded_col_length, int non_padded_row_length,
    SamplerResampling sample_type, uint8_t* d_output_rgb_interleaved,
    float& gpu_kernel_time, bool enable_timing, bool verbose_debug
) {
    int *d_sampled_indices = nullptr;
    uint8_t *d_out_r = nullptr, *d_out_g = nullptr, *d_out_b = nullptr;

    size_t out_bytes = num_points_l * sizeof(uint8_t);

    CUDA_CHECK(cudaMalloc(&d_sampled_indices, num_points_l * 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_r, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_g, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_b, out_bytes));

    int blockSize = 256;
    int numBlocks = (num_points_l + blockSize - 1) / blockSize;

    // Create CUDA event timers
    cudaEvent_t start, stop;
    float ms = 0.0f;

    if (enable_timing) {
        gpu_kernel_time = 0.0f;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    
    // Get the 4 nearest SFLP indices for each point.
    map_latlon_to_SFLP_indices_kernel<<<numBlocks, blockSize>>>(
        d_lat,
        d_lon,
        num_pixels,
        num_points_l,
        d_sampled_indices
    );
    CUDA_CHECK(cudaGetLastError());

    if (enable_timing) {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_kernel_time += ms;

        if (verbose_debug) {
            std::cout << "[Runtime Debug] map_latlon_to_SFLP_indices_kernel time: " << ms << " ms\n";
        }
    }


    int row_index_hop = optimal_index_hops.first.hop_index;
    int col_index_hop = optimal_index_hops.second.hop_index;
    double col_theta_comp = optimal_index_hops.second.theta_component;
    float map_kernel_time = 0.0f;

    // Get the row/col index pair for each sampled SFLP index.
    MappedIndices mapped_coords = get_row_col_index_gpu(
        d_sampled_indices,
        num_points_l * 4,
        primary_ref_SFLP_index,
        row_index_hop,
        col_index_hop,
        col_theta_comp,
        non_padded_col_length,
        map_kernel_time,
        enable_timing,
        verbose_debug
    );

    if (enable_timing) {
        gpu_kernel_time += map_kernel_time;

        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    // Sample based on the chosen sampling type and return the rgb for each point.
    sample_SFLP_helper_kernel<<<numBlocks, blockSize>>>(
        d_lat,
        d_lon,
        d_sampled_indices,
        mapped_coords.d_rows,
        mapped_coords.d_cols,
        d_input_image,
        img_height,
        img_width,
        img_channels,
        num_pixels,
        num_points_l,
        sample_type,
        d_out_r,
        d_out_g,
        d_out_b
    );
    CUDA_CHECK(cudaGetLastError());

    if (enable_timing) {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_kernel_time += ms;
        
        if (verbose_debug) {
            std::cout << "[Runtime Debug] sample_SFLP_helper_kernel time: " << ms << " ms\n";
        }

        CUDA_CHECK(cudaEventRecord(start, 0));        
    }

    // Interleave the rgb output.
    interleave_rgb_kernel<<<numBlocks, blockSize>>>(
        d_out_r,
        d_out_g,
        d_out_b,
        num_points_l,
        d_output_rgb_interleaved
    );
    CUDA_CHECK(cudaGetLastError());

    if (enable_timing) {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_kernel_time += ms;

        if (verbose_debug) {
            std::cout << "[Runtime Debug] interleave_rgb_kernel time: " << ms << " ms\n";
        }

        // Clean up events
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // Clean up device memory temporary allocations
    mapped_coords.free();
    cudaFree(d_sampled_indices);
    cudaFree(d_out_r);
    cudaFree(d_out_g);
    cudaFree(d_out_b);
}


// Device helper: Get cube map face ID and local u, v from 3D unit vector
__device__ inline void get_CubeMap_face_and_uv(double x, double y, double z, int& face_index, double& u, double& v) {
    double abs_x = fabs(x);
    double abs_y = fabs(y);
    double abs_z = fabs(z);

    bool is_x_major = (abs_x >= abs_y) && (abs_x >= abs_z);
    bool is_y_major = (abs_y > abs_x) && (abs_y >= abs_z);
    bool is_z_major = !is_x_major && !is_y_major;

    if (is_x_major) {
        if (x > 0.0) {
            face_index = 0; // X = +1
            u = -y / abs_x;
            v = z / abs_x;
        } else {
            face_index = 2; // X = -1
            u = y / abs_x;
            v = z / abs_x;
        }
    } else if (is_y_major) {
        if (y > 0.0) {
            face_index = 1; // Y = +1
            u = x / abs_y;
            v = z / abs_y;
        } else {
            face_index = 4; // Y = -1
            u = -z / abs_y;
            v = x / abs_y;
        }
    } else {
        if (z > 0.0) {
            face_index = 5; // Z = +1
            u = -y / abs_z;
            v = x / abs_z;
        } else {
            face_index = 3; // Z = -1
            u = y / abs_z;
            v = x / abs_z;
        }
    }
}

// Device helper: Map face index and local u,v to global CMP row/col fractional coordinates
__device__ inline void map_face_uv_to_CMP_row_col(int face_index, double u, double v, int face_size, double& row_f, double& col_f) {
    // CubeMap layout:
    // [ (0,0) X=+1 , (0,1) Y=+1 , (0,2) X=-1 ]
    // [ (1,0) Z=-1 , (1,1) Y=-1 , (1,2) Z=+1 ]
    const int face_rows[6] = {0, 0, 0, 1, 1, 1};
    const int face_cols[6] = {0, 1, 2, 0, 1, 2};

    double row_offset = static_cast<double>(face_rows[face_index] * face_size);
    double col_offset = static_cast<double>(face_cols[face_index] * face_size);

    double col_local = (u - 1.0) * -0.5 * static_cast<double>(face_size) - 0.5;
    double row_local = (v - 1.0) * -0.5 * static_cast<double>(face_size) - 0.5;

    col_f = col_offset + col_local;
    row_f = row_offset + row_local;
}

// Device helper: Map face index and local u,v to global EAC row/col fractional coordinates
__device__ inline void map_face_uv_to_EAC_row_col(int face_index, double u, double v, int face_size, double& row_f, double& col_f) {
    const int face_rows[6] = {0, 0, 0, 1, 1, 1};
    const int face_cols[6] = {0, 1, 2, 0, 1, 2};

    double row_offset = static_cast<double>(face_rows[face_index] * face_size);
    double col_offset = static_cast<double>(face_cols[face_index] * face_size);

    double pi_quarter = M_PI / 4.0;
    double col_local = ((atan(u) / pi_quarter) - 1.0) * -0.5 * static_cast<double>(face_size) - 0.5;
    double row_local = ((atan(v) / pi_quarter) - 1.0) * -0.5 * static_cast<double>(face_size) - 0.5;

    col_f = col_offset + col_local;
    row_f = row_offset + row_local;
}

// Device helper: Convert CMP row/col pixel indices back to Cartesian coordinates
__device__ inline void CMP_pixel_to_cartesian_coords(int row, int col, int face_size, double& x, double& y, double& z) {
    int face_row = row / face_size;
    int face_col = col / face_size;
    int face_index = face_row * 3 + face_col;

    int row_local = row % face_size;
    int col_local = col % face_size;

    double u = -1.0 * (2.0 * (static_cast<double>(col_local) + 0.5) / static_cast<double>(face_size)) + 1.0;
    double v = -1.0 * (2.0 * (static_cast<double>(row_local) + 0.5) / static_cast<double>(face_size)) + 1.0;

    // Inverse of CubeMap_ids_to_vectors (matching Python behavior)
    switch (face_index) {
        case 0: x = 1.0;  y = -u;   z = v;    break; // X = +1
        case 1: x = u;    y = 1.0;  z = v;    break; // Y = +1
        case 2: x = -1.0; y = u;    z = v;    break; // X = -1
        case 3: x = v;    y = u;    z = -1.0; break; // Z = -1
        case 4: x = v;    y = -1.0; z = -u;   break; // Y = -1
        case 5: x = v;    y = -u;   z = 1.0;  break; // Z = +1
    }
}

// Device helper: Convert EAC row/col pixel indices back to Cartesian coordinates
__device__ inline void EAC_pixel_to_cartesian_coords(int row, int col, int face_size, double& x, double& y, double& z) {
    int face_row = row / face_size;
    int face_col = col / face_size;
    int face_index = face_row * 3 + face_col;

    int row_local = row % face_size;
    int col_local = col % face_size;

    double pi_quarter = M_PI / 4.0;
    double norm_x = (static_cast<double>(col_local) + 0.5) / static_cast<double>(face_size);
    double norm_y = (static_cast<double>(row_local) + 0.5) / static_cast<double>(face_size);
    
    double u = tan((-(norm_x) * 2.0 + 1.0) * pi_quarter);
    double v = tan((-(norm_y) * 2.0 + 1.0) * pi_quarter);

    switch (face_index) {
        case 0: x = 1.0;  y = -u;   z = v;    break; // X = +1
        case 1: x = u;    y = 1.0;  z = v;    break; // Y = +1
        case 2: x = -1.0; y = u;    z = v;    break; // X = -1
        case 3: x = v;    y = u;    z = -1.0; break; // Z = -1
        case 4: x = v;    y = -1.0; z = -u;   break; // Y = -1
        case 5: x = v;    y = -u;   z = 1.0;  break; // Z = +1
    }
}

// CubeMap sampling helper kernel supporting NEAREST and BILINEAR (angular distance weighting)
__global__ void sample_CubeMap_helper_kernel(
    const double* __restrict__ d_lat_deg,
    const double* __restrict__ d_lon_deg,
    const uint8_t* __restrict__ input_image,
    int img_height,
    int img_width,
    int img_channels,
    int num_pixels,
    int face_size,
    bool is_CMP,
    SamplerResampling sample_type,
    uint8_t* __restrict__ out_r,
    uint8_t* __restrict__ out_g,
    uint8_t* __restrict__ out_b
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    // Compute target unit vector on the sphere
    double target_lat_rad = d_lat_deg[idx] * (M_PI / 180.0);
    double target_lon_rad = d_lon_deg[idx] * (M_PI / 180.0);
    double cos_target_lat = cos(target_lat_rad);
    double v_x = cos_target_lat * cos(target_lon_rad);
    double v_y = cos_target_lat * sin(target_lon_rad);
    double v_z = sin(target_lat_rad);

    // Get face and (u, v)
    int face_index;
    double u, v;
    get_CubeMap_face_and_uv(v_x, v_y, v_z, face_index, u, v);

    // Map to global fractional coordinates
    double row_f, col_f;
    if (is_CMP) {
        map_face_uv_to_CMP_row_col(face_index, u, v, face_size, row_f, col_f);
    } else {
        map_face_uv_to_EAC_row_col(face_index, u, v, face_size, row_f, col_f);
    }

    if (sample_type == SamplerResampling::NEAREST) {
        int row_nn = max(0, min(static_cast<int>(round(row_f)), img_height - 1));
        int col_nn = max(0, min(static_cast<int>(round(col_f)), img_width - 1));
        int pixel_offset = (row_nn * img_width + col_nn) * img_channels;

        out_r[idx] = input_image[pixel_offset + 0];
        out_g[idx] = input_image[pixel_offset + 1];
        out_b[idx] = input_image[pixel_offset + 2];
    } 
    else {
        // BILINEAR with angular distance weighting matching Python
        int row0 = static_cast<int>(floor(row_f));
        int col0 = static_cast<int>(floor(col_f));
        int row1 = max(0, min(row0 + 1, img_height - 1));
        row0 = max(0, min(row0, img_height - 1));
        int col1 = max(0, min(col0 + 1, img_width - 1));
        col0 = max(0, min(col0, img_width - 1));

        int offsets[4] = {
            (row0 * img_width + col0) * img_channels,
            (row0 * img_width + col1) * img_channels,
            (row1 * img_width + col0) * img_channels,
            (row1 * img_width + col1) * img_channels
        };

        double pixel_r[4], pixel_g[4], pixel_b[4];
        double angular_distances[4];

        for (int i = 0; i < 4; ++i) {
            pixel_r[i] = static_cast<double>(input_image[offsets[i] + 0]);
            pixel_g[i] = static_cast<double>(input_image[offsets[i] + 1]);
            pixel_b[i] = static_cast<double>(input_image[offsets[i] + 2]);

            double px, py, pz;
            int r_idx = (i == 2 || i == 3) ? row1 : row0;
            int c_idx = (i == 1 || i == 3) ? col1 : col0;

            if (is_CMP) {
                CMP_pixel_to_cartesian_coords(r_idx, c_idx, face_size, px, py, pz);
            } else {
                EAC_pixel_to_cartesian_coords(r_idx, c_idx, face_size, px, py, pz);
            }

            double dot_prod = fmax(-1.0, fmin(v_x * px + v_y * py + v_z * pz, 1.0));
            angular_distances[i] = acos(dot_prod);
        }

        const double eps = 1e-6;
        double weights[4];
        double weight_sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            weights[i] = 1.0 / (angular_distances[i] + eps);
            weight_sum += weights[i];
        }

        double accum_r = 0.0, accum_g = 0.0, accum_b = 0.0;
        for (int i = 0; i < 4; ++i) {
            double norm_w = weights[i] / weight_sum;
            accum_r += norm_w * pixel_r[i];
            accum_g += norm_w * pixel_g[i];
            accum_b += norm_w * pixel_b[i];
        }

        out_r[idx] = static_cast<uint8_t>(fmax(0.0, fmin(255.0, accum_r)));
        out_g[idx] = static_cast<uint8_t>(fmax(0.0, fmin(255.0, accum_g)));
        out_b[idx] = static_cast<uint8_t>(fmax(0.0, fmin(255.0, accum_b)));
    }
}

// CUDA Wrapper for sampling on CMP / EAC frames using GPU acceleration
void sample_CubeMap_gpu(
    const double* d_lat, const double* d_lon, int num_points_l,
    const uint8_t* d_input_image, int img_height, int img_width, int img_channels,
    bool is_CMP, SamplerResampling sample_type, uint8_t* d_output_rgb_interleaved,
    float& gpu_kernel_time, bool enable_timing
) {
    int face_size = img_width / 3;
    uint8_t *d_out_r = nullptr, *d_out_g = nullptr, *d_out_b = nullptr;
    size_t out_bytes = num_points_l * sizeof(uint8_t);

    CUDA_CHECK(cudaMalloc(&d_out_r, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_g, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_b, out_bytes));

    int blockSize = 256;
    int numBlocks = (num_points_l + blockSize - 1) / blockSize;

    cudaEvent_t start_event = nullptr, stop_event = nullptr;
    if (enable_timing) {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        CUDA_CHECK(cudaEventRecord(start_event, 0));
    }

    sample_CubeMap_helper_kernel<<<numBlocks, blockSize>>>(
        d_lat,
        d_lon,
        d_input_image,
        img_height,
        img_width,
        img_channels,
        num_points_l,
        face_size,
        is_CMP,
        sample_type,
        d_out_r,
        d_out_g,
        d_out_b
    );
    CUDA_CHECK(cudaGetLastError());

    interleave_rgb_kernel<<<numBlocks, blockSize>>>(
        d_out_r,
        d_out_g,
        d_out_b,
        num_points_l,
        d_output_rgb_interleaved
    );
    CUDA_CHECK(cudaGetLastError());

    if (enable_timing) {
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_time, start_event, stop_event));
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }

    cudaFree(d_out_r);
    cudaFree(d_out_g);
    cudaFree(d_out_b);
}