#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <filesystem>

#include "../include/data_types.hpp"
#include "../include/file_conversion_utils.hpp"
#include "../include/pixel_arrangement_utils.hpp"


std::string get_base_filename(const std::string& path_str) {
    std::filesystem::path p(path_str);
    return p.stem().string();
}

int main(int argc, char* argv[]) {
    try {
        std::string input_image_path = "../../../input/ERP-CableCar-snapshot.jpg";
        std::string input_metadata_path = "../../../input/config/ERP_metadata.json";
        std::string output_dir = "../../../output/images";
        std::string output_image_name = ""; // Empty by default to trigger auto-generation
        EncodingProjection output_projection_type = EncodingProjection::SFLP;
        int output_num_pixels = 18000001;
        int padding_size = 2;

        // Parse command-line flags
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::cout << "SFLP VR Toolkit - Image Converter\n\n"
                          << "Usage: " << argv[0] << " [options]\n\n"
                          << "Options:\n"
                          << "  --help, -h                         Show this help message and exit\n"
                          << "  --input_image_path <path>          Path to the input image\n"
                          << "  --input_metadata_path <path>       Path to input image's metadata\n"
                          << "  --output_dir <path>                Directory to save the output image (default: ../../../output/images)\n"
                          << "  --output_image_name <name>         Explicit filename to save the output image (overrides auto-generation)\n"
                          << "  --output_projection_type <type>    Target projection type (SFLP, EAC, CMP)\n"
                          << "  --output_image_num_pixels <num>    Target number of pixels for SFLP / face sizing\n"
                          << "  --padding_size <num>               Target padding size (default: 2)\n";
                return 0;
            } else if (arg == "--input_image_path" && i + 1 < argc) {
                input_image_path = argv[++i];
            } else if (arg == "--input_metadata_path" && i + 1 < argc) {
                input_metadata_path = argv[++i];
            } else if (arg == "--output_dir" && i + 1 < argc) {
                output_dir = argv[++i];
            } else if (arg == "--output_image_name" && i + 1 < argc) {
                output_image_name = argv[++i];
            } else if (arg == "--output_projection_type" && i + 1 < argc) {
                std::string pt_str = argv[++i];
                if (pt_str == "SFLP" || pt_str == "sflp") {
                    output_projection_type = EncodingProjection::SFLP;
                } else if (pt_str == "CMP" || pt_str == "cmp") {
                    output_projection_type = EncodingProjection::CMP;
                } else if (pt_str == "EAC" || pt_str == "eac") {
                    output_projection_type = EncodingProjection::EAC;
                } else {
                    std::cerr << "Warning: Unknown projection type '" << pt_str 
                              << "'. Defaulting to SFLP.\n";
                    output_projection_type = EncodingProjection::SFLP;
                }
            } else if (arg == "--output_image_num_pixels" && i + 1 < argc) {
                output_num_pixels = std::stoi(argv[++i]);
            } else if (arg == "--padding_size" && i + 1 < argc) {
                padding_size = std::stoi(argv[++i]);
            }
        }

        // Ensure output directory exists
        std::filesystem::path dir_path(output_dir);
        std::filesystem::create_directories(dir_path);

        // Auto-generate output filename if not explicitly provided via command line
        if (output_image_name.empty()) {
            std::string proj_str = "SFLP";
            if (output_projection_type == EncodingProjection::CMP) proj_str = "CMP";
            else if (output_projection_type == EncodingProjection::EAC) proj_str = "EAC";

            std::string input_stem = get_base_filename(input_image_path);
            output_image_name = proj_str + "-" + input_stem + "-" + std::to_string(output_num_pixels) + "-STANDARD.jpeg";
        }

        std::string output_image_path = (dir_path / output_image_name).string();

        std::cout << "Using input image: " << input_image_path << "\n";
        std::cout << "Using output image: " << output_image_path << "\n";

        // Call the unified conversion function
        PixelMatrix<unsigned char> result_image = convert_image_projection_gpu(
            input_image_path,
            output_projection_type,
            output_num_pixels,
            padding_size
        );

        // Save output image using stb_image_write
        auto write_start = std::chrono::high_resolution_clock::now();
        int success = stbi_write_jpg(
            output_image_path.c_str(), 
            result_image.cols, 
            result_image.rows, 
            3, 
            result_image.data.data(), 
            90 // Quality factor
        );

        if (!success) {
            success = stbi_write_png(
                output_image_path.c_str(), 
                result_image.cols, 
                result_image.rows, 
                3, 
                result_image.data.data(), 
                result_image.cols * 3
            );
        }
        auto write_end = std::chrono::high_resolution_clock::now();
        double write_ms = std::chrono::duration<double, std::milli>(write_end - write_start).count();

        if (!success) {
            throw std::runtime_error("Failed to write output image to path: " + output_image_path);
        }

        std::cout << "Image Disk Write Time:     " << write_ms << " ms\n";
        std::cout << "Successfully generated output at: " << output_image_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}