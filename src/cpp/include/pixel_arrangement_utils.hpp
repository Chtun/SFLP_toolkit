#ifndef PIXEL_ARRANGEMENT_UTILS_HPP
#define PIXEL_ARRANGEMENT_UTILS_HPP

#include "data_types.hpp"

#include <cuda_runtime.h>


using GeneratorVectorTuple = std::tuple<int, double, double, double>;

// Enum representing the pixel arrangement strategies
enum class SFLPArrangementType {
    STANDARD,
    SEAM_REMOVED,
    ZIGZAG
};

enum class SamplerResampling {
    NEAREST,
    BILINEAR,
    IDW
};

struct IndexHop {
    int hop_index;
    double theta_component;
    double z_component;
};

// Represents a 2D Grid/Matrix of pixel values (replacing DataFrame)
template <typename T>
struct PixelMatrix {
    int rows{0};
    int cols{0};
    std::vector<T> data; // Flattened 2D array in row-major order

    T operator()(int r, int c) const {
        return data[r * cols + c];
    }

    T& operator()(int r, int c) {
        return data[r * cols + c];
    }

    T at(int r, int c) const {
        return data.at(r * cols + c);
    }

    T& at(int r, int c) {
        return data.at(r * cols + c);
    }
};

/**
 * @brief Returns the golden ratio constant phi.
 * 
 * @return The golden ratio ((1 + sqrt(5)) / 2).
 */
double golden_ratio();

/**
 * @brief Get the Spherical Fibonacci Lattice vector for a particular index value and number of SFL points.
 * 
 * @param index The index of the SFL point.
 * @param num_points The number of SFL points.
 * @return Vector The Spherical Fibonacci Lattice vector.
 * @throws ValidationError If num_points is less than or equal to 0.
 */
Vector get_SFL_vector(int index, int num_points);

/**
 * @brief Projects a point in geographic coordinates to a point in the unnormalized z-theta plane.
 * 
 * @param geographic_point The geographic point.
 * @return Point The projected point in the unnormalized z-theta plane.
 */
Point get_projected_point(const GeographicPoint& geographic_point);

/**
 * @brief Gets the projected Spherical Fibonacci Lattice point in the unnormalized projected z-theta plane.
 * 
 * @param index The index of the Spherical Fibonacci Lattice point.
 * @param num_points The number of points in the Spherical Fibonacci Lattice.
 * @return Point The projected Spherical Fibonacci Lattice point in the unnormalized z-theta plane.
 */
Point get_projected_SFL_point(int index, int num_points);

/**
 * @brief Get the generator vector in the unnormalized z-theta plane for a particular index hop value and number of SFL points.
 * 
 * @param k_index The difference in index between any two SFL points.
 * @param num_points The number of SFL points.
 * @return Point The generator vector in the unnormalized z-theta plane.
 * @throws ValidationError If num_points is less than or equal to 0.
 */
Point get_projected_generator_vector(int k_index, int num_points);

/**
 * @brief Get the geodesic distance of two points on the sphere separated by
 * the specified generator vector, where one point is at the equator.
 * 
 * @param k_index The k-th index jump for each pixel.
 * @param num_points The number of pixels.
 * @return std::tuple<double, double, double> A tuple containing:
 *         - Geodesic distance (angle in radians)
 *         - Unnormalized theta component
 *         - Unnormalized z component
 */
std::tuple<double, double, double> generator_vec_geodesic_distance(int k_index, int num_points);


std::pair<GeneratorVectorTuple, GeneratorVectorTuple> find_optimal_generator_vectors(
    int n, 
    SFLPArrangementType arrangement_type = SFLPArrangementType::STANDARD,
    int k_max = -1
);

// CPU Implementation Functions
PixelMatrix<int> generate_SFLP_arrangement_standard_cpu(int num_pixels, int padding_size);

// Main Driver Function
PixelMatrix<int> generate_SFLP_arrangement(
    int num_pixels, 
    SFLPArrangementType type, 
    int padding_size, 
    bool gpu_enabled = false
);

// CUDA wrapper declarations
#ifdef USE_CUDA
// CUDA Wrapper for SFLP Standard Pixel Arrangement Generation
PixelMatrix<int> generate_SFLP_arrangement_standard_gpu(
    int num_pixels,
    int padding_size
);

void get_CubeMap_geographic_points_gpu(
    int total_pixels,
    int face_size,
    bool is_cmp,
    double* d_lat,
    double* d_lon
);

struct MappedIndices {
    int* d_rows{nullptr};
    int* d_cols{nullptr};
    int size{0};

    // Helper to clean up device memory easily
    void free() {
        if (d_rows) { cudaFree(d_rows); d_rows = nullptr; }
        if (d_cols) { cudaFree(d_cols); d_cols = nullptr; }
        size = 0;
    }
};

// Host wrapper declaration for GPU-resident mapping
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
    bool verbose_debug = false
);

#endif







#endif // PIXEL_ARRANGEMENT_UTILS_HPP