#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include "../include/pixel_arrangement_utils.hpp"

namespace fs = std::filesystem;

// Helper to convert SFLPArrangementType enum to string representation
std::string to_string(SFLPArrangementType type) {
    switch (type) {
        case SFLPArrangementType::STANDARD: return "STANDARD";
        case SFLPArrangementType::SEAM_REMOVED: return "SEAM_REMOVED";
        case SFLPArrangementType::ZIGZAG: return "ZIGZAG";
        default: return "UNKNOWN";
    }
}

// Helper function to save a PixelMatrix to a CSV file (no header, no index)
void save_to_csv(const PixelMatrix<int>& matrix, const fs::path& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath.string());
    }

    for (int r = 0; r < matrix.rows; ++r) {
        for (int c = 0; c < matrix.cols; ++c) {
            file << matrix(r, c);
            if (c < matrix.cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
}

// Helper to pad the matrix dimensions up to multiples of padding_size
PixelMatrix<int> create_padded_matrix(const PixelMatrix<int>& src, int pad_H, int pad_W) {
    PixelMatrix<int> padded;
    padded.rows = pad_H;
    padded.cols = pad_W;
    padded.data.resize(pad_H * pad_W);

    for (int r = 0; r < pad_H; ++r) {
        // Clamp source row: if r >= src.rows, duplicate the last row
        int src_r = std::min(r, src.rows - 1);

        for (int c = 0; c < pad_W; ++c) {
            // Clamp source col: if c >= src.cols, duplicate the last column
            int src_c = std::min(c, src.cols - 1);

            padded(r, c) = src(src_r, src_c);
        }
    }
    return padded;
}
// Helper to convert string arguments to lower case
static std::string to_lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { 
        return static_cast<char>(std::tolower(c)); 
    });
    return str;
}

// Helper to parse string representation into SFLPArrangementType
static SFLPArrangementType parse_arrangement_type(const std::string& str) {
    std::string lower_str = to_lower(str);
    if (lower_str == "standard") {
        return SFLPArrangementType::STANDARD;
    } else if (lower_str == "seam_removed" || lower_str == "seamremoved") {
        return SFLPArrangementType::SEAM_REMOVED;
    } else if (lower_str == "zigzag") {
        return SFLPArrangementType::ZIGZAG;
    }
    throw std::invalid_argument("Unknown arrangement_type: " + str + 
                                " (Expected: 'standard', 'seam_removed', or 'zigzag')");
}

int main(int argc, char* argv[]) {
    try {
        // --- Default Values ---
        fs::path output_dir = "../output/layouts";
        fs::path input_dir = "../input";
        int num_points = 647;
        SFLPArrangementType arrangement_type = SFLPArrangementType::STANDARD;
        int padding_size = 2;
        bool gpu_enabled = true;

        // --- CLI Argument Parsing ---
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n\n"
                          << "Options:\n"
                          << "  --num_points <int>         Number of points (default: 647)\n"
                          << "  --arrangement_type <string> Arrangement type: standard, seam_removed, zigzag (default: standard)\n"
                          << "  --padding_size <int>       Padding size (default: 2)\n"
                          << "  --output_dir <path>        Output directory path (default: ../output/layouts)\n"
                          << "  --input_dir <path>         Input directory path (default: ../input)\n"
                          << "  --gpu_enabled <bool>       Enable GPU processing: true/false/1/0 (default: true)\n"
                          << "  --help, -h                 Display this help message\n";
                return 0;
            } 
            else if (arg == "--num_points" && i + 1 < argc) {
                num_points = std::stoi(argv[++i]);
            } 
            else if (arg == "--arrangement_type" && i + 1 < argc) {
                arrangement_type = parse_arrangement_type(argv[++i]);
            } 
            else if (arg == "--padding_size" && i + 1 < argc) {
                padding_size = std::stoi(argv[++i]);
            } 
            else if (arg == "--output_dir" && i + 1 < argc) {
                output_dir = argv[++i];
            } 
            else if (arg == "--input_dir" && i + 1 < argc) {
                input_dir = argv[++i];
            } 
            else if (arg == "--gpu_enabled" && i + 1 < argc) {
                std::string val = to_lower(argv[++i]);
                gpu_enabled = (val == "true" || val == "1" || val == "yes");
            }
        }

        // --- Directory Setup ---
        fs::create_directories(output_dir);

        // --- Generate Pixel Arrangement ---
        PixelMatrix<int> arranged_pixel_matrix = generate_SFLP_arrangement(
            num_points, 
            arrangement_type, 
            padding_size, 
            gpu_enabled
        );

        std::cout << "Successfully generated pixel arrangement with " << num_points << " points.\n";

        // Save unpadded CSV
        std::string filename_base = "arrangement_" + to_string(arrangement_type) + 
                                    "-360_No_Tiling-" + std::to_string(num_points);
        fs::path unpadded_path = output_dir / (filename_base + ".csv");
        save_to_csv(arranged_pixel_matrix, unpadded_path);

        // Calculate Padded Dimensions
        int H = arranged_pixel_matrix.rows;
        int W = arranged_pixel_matrix.cols;

        // Round up to multiple of padding_size
        int pad_H = ((H + padding_size - 1) / padding_size) * padding_size;
        int pad_W = ((W + padding_size - 1) / padding_size) * padding_size;

        if (pad_H != H || pad_W != W) {
            arranged_pixel_matrix = create_padded_matrix(arranged_pixel_matrix, pad_H, pad_W);
        }

        // Save Padded CSV
        fs::path output_path = output_dir / (filename_base + "_padded.csv");
        save_to_csv(arranged_pixel_matrix, output_path);

        std::cout << "layout saved to: " << output_path << std::endl;

        // Validate Pixel Index Occurrence Counts
        std::map<int, int> indices_found;
        for (int i = 0; i < num_points; ++i) {
            indices_found[i] = 0;
        }

        for (int val : arranged_pixel_matrix.data) {
            if (val != -1) {
                if (indices_found.find(val) != indices_found.end()) {
                    indices_found[val]++;
                }
            }
        }

        int multiple_copies = 0;
        for (const auto& [val, count] : indices_found) {
            if (count == 0) {
                std::cout << val << " was not found in the table!" << std::endl;
            } else if (count > 1) {
                multiple_copies += (count - 1);
            }
        }

        std::cout << multiple_copies << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}