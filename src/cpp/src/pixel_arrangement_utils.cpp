#include <algorithm>
#include <cmath>
#include <iostream>

#include "../include/pixel_arrangement_utils.hpp"

namespace {
    constexpr double PI = 3.14159265358979323846;

    inline double deg2rad(double deg) {
        return deg * (PI / 180.0);
    }

    inline double rad2deg(double rad) {
        return rad * (180.0 / PI);
    }

    // Python-compatible floating-point modulo to handle negative numerators correctly
    // (e.g., Python: -1 % 1.6180339 -> 0.6180339, whereas std::fmod gives -1.0)
    inline double py_mod(double a, double b) {
        double r = std::fmod(a, b);
        return (r < 0.0) ? (r + std::abs(b)) : r;
    }
}

double golden_ratio() {
    static const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    return phi;
}

Vector get_SFL_vector(int index, int num_points) {
    if (num_points <= 0) {
        throw ValidationError("Number of points must be positive");
    }

    if (num_points % 2 == 0) {
        num_points -= 1;
        std::cout << "Warning: The number of points must be odd, so Fibonacci lattice point generation will be done with n=" 
                  << num_points << "\n";
    }

    double phi = golden_ratio();
    int N = (num_points - 1) / 2;
    int i = index - N;

    double lat = rad2deg(std::asin((2.0 * i) / (2.0 * N + 1.0)));
    double lon = py_mod(i, phi) * 360.0 / phi;

    // Normalize longitude to [-180, 180]
    lon = py_mod(lon + 180.0, 360.0) - 180.0;

    return Vector::from_geographic(lon, lat);
}

Point get_projected_point(const GeographicPoint& geographic_point) {
    double theta = geographic_point.lon;
    double phi = 90.0 - geographic_point.lat;

    return Point(theta, std::cos(deg2rad(phi)));
}

Point get_projected_SFL_point(int index, int num_points) {
    int N = (num_points - 1) / 2;
    int i = index - N;
    double phi = golden_ratio();

    double theta = py_mod(i, phi) * 360.0 / phi;
    theta = py_mod(theta + 180.0, 360.0) - 180.0;

    double z = (2.0 * i) / (2.0 * N + 1.0);

    return Point(theta, z);
}

Point get_projected_generator_vector(int k_index, int num_points) {
    if (num_points <= 0) {
        throw ValidationError("Number of points must be positive");
    }

    int N = (num_points - 1) / 2;
    double phi = golden_ratio();

    double frac = py_mod(static_cast<double>(k_index) / phi, 1.0);
    if (frac > 0.5) {
        frac -= 1.0;
    }

    double x = frac * 360.0;
    double y = (2.0 * k_index) / (2.0 * N + 1.0);

    return Point(x, y);
}

std::tuple<double, double, double> generator_vec_geodesic_distance(int k_index, int num_points) {
    Point projected_generator_vector = get_projected_generator_vector(k_index, num_points);

    double y_value = projected_generator_vector.y;
    double clamped_y = std::clamp(y_value, -1.0, 1.0);

    double latitude = rad2deg(std::asin(clamped_y));
    double longitude = projected_generator_vector.x;

    Vector generator_vector_from_reference = Vector::from_geographic(longitude, latitude);
    Vector reference_vector(1.0, 0.0, 0.0);

    double dot_product = generator_vector_from_reference.dot(reference_vector);
    dot_product = std::clamp(dot_product, -1.0, 1.0);

    double geodesic_distance = std::acos(dot_product);

    return {geodesic_distance, projected_generator_vector.x, projected_generator_vector.y};
}

std::pair<GeneratorVectorTuple, GeneratorVectorTuple> find_optimal_generator_vectors(
    int n, 
    SFLPArrangementType arrangement_type,
    int k_max
) {
    if (k_max < 1) {
        k_max = static_cast<int>(3.0 * std::sqrt(n));
    }

    // values = [(k, val[0], val[1], val[2]) for k, val in values]
    std::vector<GeneratorVectorTuple> values;
    values.reserve(k_max);

    for (int k = 1; k <= k_max; ++k) {
        auto [dist, theta, z] = generator_vec_geodesic_distance(k, n);
        values.push_back({k, dist, theta, z});
    }

    // values.sort(key=lambda x: x[1])  # Sort by distance
    std::sort(values.begin(), values.end(), [](const GeneratorVectorTuple& a, const GeneratorVectorTuple& b) {
        return std::get<1>(a) < std::get<1>(b);
    });

    // nearest_neighbors = values[:4]
    std::vector<GeneratorVectorTuple> nearest_neighbors(
        values.begin(), 
        values.begin() + std::min(4, static_cast<int>(values.size()))
    );

    // nearest_neighbors.sort(key=lambda x: x[0], reverse=True)  # Sort by k descending
    std::sort(nearest_neighbors.begin(), nearest_neighbors.end(), [](const GeneratorVectorTuple& a, const GeneratorVectorTuple& b) {
        return std::get<0>(a) > std::get<0>(b);
    });

    int neighbor_one   = std::get<0>(nearest_neighbors[0]);
    int neighbor_two   = std::get<0>(nearest_neighbors[1]);
    int neighbor_three = std::get<0>(nearest_neighbors[2]);

    double n1_distance   = std::get<0>(generator_vec_geodesic_distance(neighbor_one, n));
    double n2n3_distance = std::get<0>(generator_vec_geodesic_distance(neighbor_two - neighbor_three, n));

    int optimal_k_1 = 0;
    int optimal_k_2 = 0;

    if ((n1_distance / n2n3_distance) > 0.7) {
        optimal_k_1 = neighbor_two;
        optimal_k_2 = neighbor_three;
    } else {
        optimal_k_1 = neighbor_one;
        optimal_k_2 = neighbor_two;
    }

    if (arrangement_type == SFLPArrangementType::STANDARD) {
        int min_k = std::min(optimal_k_1, optimal_k_2);
        int max_k = std::max(optimal_k_1, optimal_k_2);

        optimal_k_1 = max_k - min_k;
        optimal_k_2 = min_k;
    }

    // Build final tuple representations for optimal_k_1 and optimal_k_2
    auto [dist1, theta1, z1] = generator_vec_geodesic_distance(optimal_k_1, n);
    GeneratorVectorTuple vec1 = {optimal_k_1, dist1, theta1, z1};

    auto [dist2, theta2, z2] = generator_vec_geodesic_distance(optimal_k_2, n);
    GeneratorVectorTuple vec2 = {optimal_k_2, dist2, theta2, z2};

    return {vec1, vec2};
}

// CPU Implementation of generate_SFLP_arrangement_standard
PixelMatrix<int> generate_SFLP_arrangement_standard_cpu(int num_pixels, int padding_size) {
    // 1. Compute generator vectors and bounds
    int search_bound = 3 * static_cast<int>(std::floor(std::sqrt(num_pixels)));
    std::pair<GeneratorVectorTuple, GeneratorVectorTuple> optimal_index_hops = 
        find_optimal_generator_vectors(num_pixels, SFLPArrangementType::STANDARD, search_bound);
    
    int row_index_hop = std::get<0>(optimal_index_hops.first);
    int col_index_hop = std::get<0>(optimal_index_hops.second);

    int num_columns = col_index_hop;
    int longest_column_length = static_cast<int>(std::floor(static_cast<double>(num_pixels) / col_index_hop));

    int padded_amount = padding_size - (longest_column_length % padding_size);
    longest_column_length += padded_amount;

    PixelMatrix<int> result;
    result.rows = longest_column_length;
    result.cols = num_columns;
    result.data.resize(result.rows * result.cols);

    // 2. Compute indices per element
    std::vector<int> boundary(num_columns, longest_column_length);
    std::vector<bool> column_has_mask(num_columns, false);

    for (int r = 0; r < result.rows; ++r) {
        for (int c = 0; c < result.cols; ++c) {
            int pixel_index = (longest_column_length - 1 - r) * col_index_hop + c * row_index_hop;
            bool mask_too_large = (pixel_index >= num_pixels);

            if (mask_too_large) {
                int fallback_indices = c * row_index_hop - ((r + 1) * col_index_hop);
                pixel_index = fallback_indices;

                if (pixel_index <= 0) {
                    pixel_index = (c * row_index_hop) % col_index_hop;
                }

                if (!column_has_mask[c]) {
                    boundary[c] = r;
                    column_has_mask[c] = true;
                }
            }

            result.at(r, c) = pixel_index;
        }
    }

    // 3. Apply padding mask and edge values substitution
    int half_pad = padded_amount / 2;
    for (int c = 0; c < result.cols; ++c) {
        int b = boundary[c];
        int start_pad_row = b - half_pad;
        int safe_boundary = std::clamp(b, 0, longest_column_length - 1);
        int edge_value = result.at(safe_boundary, c);

        for (int r = 0; r < result.rows; ++r) {
            if (r >= start_pad_row && r < b) {
                result.at(r, c) = edge_value;
            }
        }
    }

    return result;
}




// Standard Arrangement Helper (CPU / GPU dispatcher)
PixelMatrix<int> generate_SFLP_arrangement_standard(int num_pixels, int padding_size, bool gpu_enabled) {
    if (gpu_enabled) {
#ifdef USE_CUDA
        std::cout << "GPU processing enabled. Using CUDA implementation.\n";
        return generate_SFLP_arrangement_standard_gpu(num_pixels, padding_size);
#else
        throw std::runtime_error("GPU processing was requested, but USE_CUDA is not defined.");
#endif
    }
    return generate_SFLP_arrangement_standard_cpu(num_pixels, padding_size);
}

// Dispatcher for generating SFLP pixel arrangement based on arrangement type and GPU availability
PixelMatrix<int> generate_SFLP_arrangement(
    int num_pixels, 
    SFLPArrangementType arrangement_type, 
    int padding_size, 
    bool gpu_enabled
) {
    switch (arrangement_type) {
        case SFLPArrangementType::STANDARD:
            return generate_SFLP_arrangement_standard(num_pixels, padding_size, gpu_enabled);

        case SFLPArrangementType::SEAM_REMOVED:
            throw std::logic_error("generate_SFLP_arrangement_seam_removed is not implemented yet!");

        case SFLPArrangementType::ZIGZAG:
            throw std::logic_error("generate_SFLP_arrangement_zigzag is not implemented yet!");

        default:
            throw std::invalid_argument("Arrangement type is not supported!");
    }
}

