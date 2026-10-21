#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <iostream>

#include "../include/pixel_arrangement_utils.hpp"

#define M_PI 3.14159265358979323846

// Macro for standard CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

/* Forwards Projection Functions */

// Compute pixel indices and record boundary per column
__global__ void compute_indices_kernel(
    int* d_pixel_indices,
    int* d_boundary,
    int rows,
    int cols,
    int num_pixels,
    int row_index_hop,
    int col_index_hop
) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= rows || c >= cols) return;

    int idx = r * cols + c;

    int pixel_index = (rows - 1 - r) * col_index_hop + c * row_index_hop;
    bool mask_too_large = (pixel_index >= num_pixels);

    if (mask_too_large) {
        int fallback = c * row_index_hop - ((r + 1) * col_index_hop);
        pixel_index = fallback;

        if (pixel_index <= 0) {
            pixel_index = (c * row_index_hop) % col_index_hop;
        }

        atomicMin(&d_boundary[c], r);
    }

    d_pixel_indices[idx] = pixel_index;
}

// Apply padding mask and replace padded rows with boundary edge values
__global__ void apply_padding_kernel(
    int* d_pixel_indices,
    const int* d_boundary,
    int rows,
    int cols,
    int padded_amount
) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= rows || c >= cols) return;

    int boundary_row = d_boundary[c];
    
    int safe_boundary = boundary_row;
    if (safe_boundary < 0) safe_boundary = 0;
    if (safe_boundary >= rows) safe_boundary = rows - 1;

    int half_pad = padded_amount / 2;
    int start_pad_row = boundary_row - half_pad;

    if (r >= start_pad_row && r < boundary_row) {
        int edge_value = d_pixel_indices[safe_boundary * cols + c];
        d_pixel_indices[r * cols + c] = edge_value;
    }
}

// CUDA kernel to replicate the last valid column across padded columns
__global__ void replicate_padding_columns_kernel(
    int* __restrict__ d_pixel_indices,
    int longest_column_length,
    int base_num_columns,
    int padded_num_columns
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = longest_column_length * padded_num_columns;
    if (idx >= total_elements) return;

    int r = idx / padded_num_columns;
    int c = idx % padded_num_columns;

    // Only affect the newly padded columns
    if (c >= base_num_columns) {
        // Grab the value from the last valid column of the same row
        int last_valid_idx = r * padded_num_columns + (base_num_columns - 1);
        d_pixel_indices[idx] = d_pixel_indices[last_valid_idx];
    }
}

// Host Wrapper Function keeping SFLP arrangement entirely on GPU
PixelMatrix<int> generate_SFLP_arrangement_standard_gpu(int num_pixels, int padding_size) {
    // Calculate generator parameters (Host-side math is fine as it's quick metadata)
    int search_bound = 3 * static_cast<int>(std::floor(std::sqrt(num_pixels)));
    auto optimal_index_hops = find_optimal_generator_vectors(num_pixels, SFLPArrangementType::STANDARD, search_bound);
    
    int row_index_hop = std::get<0>(optimal_index_hops.first);
    int col_index_hop = std::get<0>(optimal_index_hops.second);

    int base_num_columns = col_index_hop;
    
    // --- Column Padding Calculation ---
    int col_padding_remainder = base_num_columns % padding_size;
    int col_padded_amount = (col_padding_remainder == 0) ? 0 : (padding_size - col_padding_remainder);
    int padded_num_columns = base_num_columns + col_padded_amount;

    int longest_column_length = std::floor(static_cast<double>(num_pixels) / col_index_hop);
    int row_padded_amount = padding_size - (longest_column_length % padding_size);
    longest_column_length += row_padded_amount;

    int total_elements = longest_column_length * padded_num_columns;
    size_t full_matrix_bytes = total_elements * sizeof(int);
    size_t boundary_bytes = base_num_columns * sizeof(int);

    // Allocate Device Memory
    int* d_pixel_indices = nullptr;
    int* d_boundary = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pixel_indices, full_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_boundary, boundary_bytes));

    // Initialize boundary values
    std::vector<int> h_initial_boundary(base_num_columns, longest_column_length);
    CUDA_CHECK(cudaMemcpy(d_boundary, h_initial_boundary.data(), boundary_bytes, cudaMemcpyHostToDevice));

    // Configure Grid and Block Dimensions for index computation
    dim3 block(16, 16);
    dim3 grid(
        (base_num_columns + block.x - 1) / block.x,
        (longest_column_length + block.y - 1) / block.y
    );

    // Launch Kernels for base columns
    compute_indices_kernel<<<grid, block>>>(
        d_pixel_indices,
        d_boundary,
        longest_column_length,
        padded_num_columns,
        num_pixels,
        row_index_hop,
        col_index_hop
    );
    CUDA_CHECK(cudaGetLastError());

    apply_padding_kernel<<<grid, block>>>(
        d_pixel_indices,
        d_boundary,
        longest_column_length,
        padded_num_columns,
        row_padded_amount
    );
    CUDA_CHECK(cudaGetLastError());

    // Launch GPU Kernel to replicate padding columns in parallel (replaces CPU loop)
    int pad_block = 256;
    int pad_grid = (total_elements + pad_block - 1) / pad_block;
    replicate_padding_columns_kernel<<<pad_grid, pad_block>>>(
        d_pixel_indices,
        longest_column_length,
        base_num_columns,
        padded_num_columns
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary device boundary memory
    CUDA_CHECK(cudaFree(d_boundary));

    // Final Host Transfer (done only once at the very end)
    std::vector<int> h_full_data(total_elements);
    CUDA_CHECK(cudaMemcpy(h_full_data.data(), d_pixel_indices, full_matrix_bytes, cudaMemcpyDeviceToHost));

    // Free Device Arrangement Memory
    CUDA_CHECK(cudaFree(d_pixel_indices));

    // Prepare Result Matrix
    PixelMatrix<int> result;
    result.rows = longest_column_length;
    result.cols = padded_num_columns;
    result.data = std::move(h_full_data);

    return result;
}

// Device helper: Convert EAC local pixel index to normalized planar coordinates [-1, 1]
__device__ inline void eac_indices_to_uv(int local_x, int local_y, int face_size, double& u, double& v) {
    // u = tan((-(col_local + 0.5) / face_size * 2 + 1) * pi_quarter)
    // v = tan((-(row_local + 0.5) / face_size * 2 + 1) * pi_quarter)
    double pi_quarter = M_PI / 4.0;
    
    double norm_x = (static_cast<double>(local_x) + 0.5) / static_cast<double>(face_size);
    double norm_y = (static_cast<double>(local_y) + 0.5) / static_cast<double>(face_size);
    
    u = tan((-(norm_x) * 2.0 + 1.0) * pi_quarter);
    v = tan((-(norm_y) * 2.0 + 1.0) * pi_quarter);
}

// Device helper: Convert standard CMP local pixel index to normalized planar coordinates [-1, 1]
__device__ inline void cmp_indices_to_uv(int local_x, int local_y, int face_size, double& u, double& v) {
    u = -1.0 * (2.0 * (static_cast<double>(local_x) + 0.5) / static_cast<double>(face_size)) + 1.0;
    v = -1.0 * (2.0 * (static_cast<double>(local_y) + 0.5) / static_cast<double>(face_size)) + 1.0;
}

// Device helper: Map face index and (u, v) to 3D Cartesian vectors (x, y, z)
__device__ inline void cubemap_id_to_vector(int face_idx, double u, double v, double& x, double& y, double& z) {
    // Standard face mapping convention matching standard layouts (+X, -X, +Y, -Y, +Z, -Z)
    switch (face_idx) {
        case 0: // +X
            x = 1.0; y = -u; z = v; break;
        case 1: // +Y
            x = u; y = 1.0; z = v; break;
        case 2: // -X
            x = -1.0; y = u; z = v; break;
        case 3: // -Z
            x = v; y = u; z = -1.0; break;
        case 4: // -Y
            x = v; y = -1.0; z = -u; break;
        case 5: // +Z
            x = v; y = -u; z = 1.0; break;

        default:
            x = 0.0; y = 0.0; z = 1.0; break;
    }
}

// CUDA Kernel to compute geographic points directly from thread indices
__global__ void compute_cubemap_geographic_points_kernel(
    int total_pixels,
    int face_size,
    bool is_cmp,
    double* __restrict__ d_lat,
    double* __restrict__ d_lon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    // Determine which cubemap face and local pixel coordinates correspond to this index
    int img_width = face_size * 3;
    int px = idx % img_width;
    int py = idx / img_width;

    int local_x = idx % face_size;
    int local_y = py % face_size;

    int face_idx = (py / face_size) * 3 + (px / face_size);

    double u = 0.0, v = 0.0;
    if (is_cmp) {
        cmp_indices_to_uv(local_x, local_y, face_size, u, v);
    } else {
        eac_indices_to_uv(local_x, local_y, face_size, u, v);
    }

    double x = 0.0, y = 0.0, z = 0.0;
    cubemap_id_to_vector(face_idx, u, v, x, y, z);

    // Normalize 3D vector
    double norm = sqrt(x * x + y * y + z * z);
    if (norm > 1e-6) {
        x /= norm;
        y /= norm;
        z /= norm;
    }

    // Convert Cartesian coordinates to Lat/Lon (in degrees)
    double lat = asin(z);
    double lon = atan2(y, x);

    d_lat[idx] = lat * (180.0 / M_PI);
    d_lon[idx] = lon * (180.0 / M_PI);
}

void get_CubeMap_geographic_points_gpu(
    int total_pixels,
    int face_size,
    bool is_cmp,
    double* d_lat,
    double* d_lon
) {
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;

    compute_cubemap_geographic_points_kernel<<<grid_size, block_size>>>(
        total_pixels,
        face_size,
        is_cmp,
        d_lat,
        d_lon
    );
    
    CUDA_CHECK(cudaGetLastError());
}

/* Backwards Projection (Sampling) Functions */

// Device implementation to obtain col index from SFLP index for STANDARD arrangement
__device__ inline int get_col_index_device(
    int sflp_index,
    int last_element_SFLP_index,
    int first_index_hop,
    int second_index_hop,
    double col_generator_theta_component
) {
    int multiplier = last_element_SFLP_index - sflp_index;
    if (col_generator_theta_component > 0.0) {
        multiplier = -multiplier;
    }

    int prev_fb_number, larger_fb_number;
    if (first_index_hop > second_index_hop) {
        prev_fb_number = first_index_hop - second_index_hop;
        larger_fb_number = first_index_hop;
    } else {
        prev_fb_number = second_index_hop - first_index_hop;
        larger_fb_number = second_index_hop;
    }

    int mod_larger = multiplier % larger_fb_number;
    if (mod_larger < 0) mod_larger += larger_fb_number;

    int mod_prev = (prev_fb_number % larger_fb_number);
    if (mod_prev < 0) mod_prev += larger_fb_number;

    int col_index = (mod_prev * mod_larger) % larger_fb_number;
    return col_index;
}

// Device implementation to obtain row index from SFLP index for STANDARD arrangement
__device__ inline int get_row_index_device(
    int sflp_index,
    int col_index,
    int last_element_SFLP_index,
    int row_index_hop,
    int col_index_hop,
    int col_length
) {
    long long bottom_row_index = (long long)col_index * row_index_hop + last_element_SFLP_index;
    long long row_index = (sflp_index - bottom_row_index) / col_index_hop;

    long long index_from_bottom = (long long)(col_length - 1) - row_index;
    long long index_from_top = -row_index - 1;

    long long final_row = (row_index >= 0) ? index_from_bottom : index_from_top;
    return (int)final_row;
}

// CUDA kernel to obtain row and col index from SFLP index for STANDARD arrangement
__global__ void get_row_col_index_kernel(
    const int* __restrict__ sflp_indices,
    int num_points,
    int last_element_SFLP_index,
    int row_index_hop,
    int col_index_hop,
    double col_generator_theta_component,
    int col_length,
    int L,
    int* __restrict__ out_rows,
    int* __restrict__ out_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;

    int s_idx = sflp_indices[idx];

    int col_idx = get_col_index_device(
        s_idx,
        last_element_SFLP_index,
        row_index_hop,
        col_index_hop,
        col_generator_theta_component
    );

    int row_idx = get_row_index_device(
        s_idx,
        col_idx,
        last_element_SFLP_index,
        row_index_hop,
        col_index_hop,
        col_length
    );

    out_cols[idx] = col_idx;
    out_rows[idx] = row_idx;
}

// Host Wrapper Function to compute row and column indices on the GPU
MappedIndices get_row_col_index_gpu(
    const int* d_sflp_indices,
    int num_points,
    int last_element_SFLP_index,
    int row_index_hop,
    int col_index_hop,
    double col_generator_theta_component,
    int col_length,
    float gpu_kernel_time,
    bool enable_timing,
    bool verbose_debug
) {
    size_t bytes = num_points * sizeof(int);

    int* d_out_rows = nullptr;
    int* d_out_cols = nullptr;

    CUDA_CHECK(cudaMalloc(&d_out_rows, bytes));
    CUDA_CHECK(cudaMalloc(&d_out_cols, bytes));

    int threads_per_block = 256;
    int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;

    if (enable_timing) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    
        // Record the start event
        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    get_row_col_index_kernel<<<num_blocks, threads_per_block>>>(
        d_sflp_indices,
        num_points,
        last_element_SFLP_index,
        row_index_hop,
        col_index_hop,
        col_generator_theta_component,
        col_length,
        num_points,
        d_out_rows,
        d_out_cols
    );

    CUDA_CHECK(cudaGetLastError());

    if (enable_timing) {
        // Record the stop event and synchronize
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_time, start, stop));

        if (verbose_debug) {
            std::cout << "[Runtime Debug] get_row_col_index_gpu execution time: " << gpu_kernel_time << " ms\n";
        }

        // Clean up events
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    MappedIndices result;
    result.d_rows = d_out_rows;
    result.d_cols = d_out_cols;
    result.size = num_points;

    return result;
}