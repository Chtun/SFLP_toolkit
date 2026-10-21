#ifndef FILE_CONVERSION_UTILS_HPP
#define FILE_CONVERSION_UTILS_HPP

#include "pixel_arrangement_utils.hpp"
#include "data_types.hpp"

enum class EncodingProjection {
    ERP,
    CMP,
    EAC,
    EAP,
    SFLP
};

enum class VideoEncoder {
    AV1,
    H264,
    H265
};

struct ProjectionMetadata {
    EncodingProjection projection_type;
    int num_pixels;
    explicit ProjectionMetadata(EncodingProjection type, int n_pixels) :
    projection_type(type),
    num_pixels(n_pixels)
    {}
    virtual ~ProjectionMetadata() = default;
};

struct SFLPMetadata : public ProjectionMetadata {
    SFLPArrangementType arrangement_type;
    std::pair<IndexHop, IndexHop> optimal_index_hops;
    int primary_reference_SFLP_index;
    int secondary_reference_SFLP_index;
    int padding_size;
    
    int non_padded_row_length;
    int non_padded_col_length;

    SFLPMetadata(
        int pixels, 
        SFLPArrangementType arr_type, 
        std::pair<int, int> hops, 
        int primary_idx, 
        int secondary_idx, 
        int pad_size = -1
    ) : ProjectionMetadata(EncodingProjection::SFLP, pixels),
        arrangement_type(arr_type),
        primary_reference_SFLP_index(primary_idx),
        secondary_reference_SFLP_index(secondary_idx),
        padding_size(pad_size) 
    {

        
        Point pt1 = get_projected_generator_vector(hops.first, num_pixels);
        IndexHop hop1{hops.first, pt1.x, pt1.y};

        Point pt2 = get_projected_generator_vector(hops.second, num_pixels);
        IndexHop hop2{hops.second, pt2.x, pt2.y};

        optimal_index_hops = {hop1, hop2};

        if (arrangement_type == SFLPArrangementType::STANDARD) {
            if (padding_size < 0) {
                throw std::invalid_argument("Padding size must be provided for STANDARD arrangement!");
            }
            non_padded_row_length = optimal_index_hops.second.hop_index;
            non_padded_col_length = num_pixels / non_padded_row_length;
            non_padded_col_length = non_padded_col_length + (padding_size - (non_padded_col_length % padding_size));
        } else if (arrangement_type == SFLPArrangementType::SEAM_REMOVED) {
            non_padded_row_length = optimal_index_hops.second.hop_index;
            non_padded_col_length = (num_pixels + non_padded_row_length - 1) / non_padded_row_length;
        } else if (arrangement_type == SFLPArrangementType::ZIGZAG) {
            non_padded_row_length = (optimal_index_hops.first.hop_index + optimal_index_hops.second.hop_index + 1) / 2;
            non_padded_col_length = 2 * ((num_pixels + non_padded_row_length - 1) / non_padded_row_length);
        } else {
            throw std::invalid_argument("Unsupported SFLP Arrangement Type!");
        }
    }

    SFLPMetadata(
        int pixels, 
        SFLPArrangementType arr_type, 
        std::pair<IndexHop, IndexHop> hops, 
        int first_idx, 
        int last_idx, 
        int pad_size = -1
    ) : ProjectionMetadata(EncodingProjection::SFLP, pixels),
        arrangement_type(arr_type),
        optimal_index_hops(hops),
        primary_reference_SFLP_index(first_idx),
        secondary_reference_SFLP_index(last_idx),
        padding_size(pad_size) 
    {
        if (arrangement_type == SFLPArrangementType::STANDARD) {
            if (padding_size < 0) {
                throw std::invalid_argument("Padding size must be provided for STANDARD arrangement!");
            }
            non_padded_row_length = optimal_index_hops.second.hop_index;
            non_padded_col_length = num_pixels / non_padded_row_length;
            non_padded_col_length = non_padded_col_length + (padding_size - (non_padded_col_length % padding_size));
        } else if (arrangement_type == SFLPArrangementType::SEAM_REMOVED) {
            non_padded_row_length = optimal_index_hops.second.hop_index;
            non_padded_col_length = (num_pixels + non_padded_row_length - 1) / non_padded_row_length;
        } else if (arrangement_type == SFLPArrangementType::ZIGZAG) {
            non_padded_row_length = (optimal_index_hops.first.hop_index + optimal_index_hops.second.hop_index + 1) / 2;
            non_padded_col_length = 2 * ((num_pixels + non_padded_row_length - 1) / non_padded_row_length);
        } else {
            throw std::invalid_argument("Unsupported SFLP Arrangement Type!");
        }
    }
};


void compute_sfl_geographic_points_cpu(
    const std::vector<int>& indices, 
    int num_points, 
    std::vector<double>& lat_out, 
    std::vector<double>& lon_out
);



// Helper functions for string and JSON parsing and metadata loading
std::pair<double, double> parse_double_pair(const std::string& str);
std::pair<int, int> parse_int_pair(const std::string& str);
std::string parse_json_string(const std::string& content, const std::string& key);
int parse_json_int(const std::string& content, const std::string& key, int default_val = 0);
std::string read_file_to_string(const std::string& filepath);
std::pair<int, int> parse_json_hops(const std::string& content);
ProjectionMetadata load_metadata_from_json(const std::string& json_path, int fallback_num_pixels = -1);
std::unique_ptr<ProjectionMetadata> load_metadata_ptr_from_json(const std::string& json_path, int fallback_num_pixels);

// Helper to query video dimensions and fps automatically using ffprobe
bool probe_video_metadata(const std::string& ffprobe_path,
    const std::string& video_path,
    int& width,
    int& height,
    int& fps,
    double& duration
);

// Function to convert ERP image to SFLP format using GPU acceleration
PixelMatrix<unsigned char> convert_erp_image_to_sflp_from_path(
    const std::string& input_image_path,
    const ProjectionMetadata& input_metadata,
    const SFLPMetadata& output_metadata
);

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
);

// CUDA wrapper declarations
#ifdef USE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Wrapper for SFLP Geographic Point Conversion
void compute_sfl_geographic_points_gpu(
    const int* d_indices, 
    int num_indices, 
    int num_points, 
    double* d_lat_out, 
    double* d_lon_out
);

// CUDA Wrapper for ERP to ERP/EAP/CMP/EAC/SFLP Image Sampling Pipeline
void convert_ERP_projection_gpu(
    const double* d_lat,
    const double* d_lon,
    int total_pixels,
    const unsigned char* d_erp_image,
    int erp_h,
    int erp_w,
    unsigned char* d_output_rgb_out
);

#ifdef __cplusplus
}
#endif

// CUDA Wrapper for sampling on an SFLP frame using GPU acceleration
void sample_SFLP_gpu(
    const double* d_lat, const double* d_lon, int num_points_l,
    const uint8_t* d_input_image, int img_height, int img_width, int img_channels,
    int num_pixels, SFLPArrangementType arrangement_type,
    const std::pair<IndexHop, IndexHop>& optimal_index_hops,
    int primary_ref_SFLP_index, int secondary_ref_SFLP_index,
    int non_padded_col_length, int non_padded_row_length,
    SamplerResampling sample_type, uint8_t* d_output_rgb_interleaved,
    float& gpu_kernel_time, bool enable_timing, bool verbose_debug = false
);

// CUDA Wrapper for sampling on CMP / EAC frames using GPU acceleration
void sample_CubeMap_gpu(
    const double* d_lat, const double* d_lon, int num_points_l,
    const uint8_t* d_input_image, int img_height, int img_width, int img_channels,
    bool is_CMP, SamplerResampling sample_type, uint8_t* d_output_rgb_interleaved,
    float& gpu_kernel_time, bool enable_timing
);

PixelMatrix<unsigned char> convert_image_projection_gpu(
    const std::string& input_image_path,
    EncodingProjection target_projection,
    int output_num_pixels,
    int padding_size
);

#endif

#endif