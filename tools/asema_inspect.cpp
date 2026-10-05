#include "../include/asema/manifest.hpp"
#include "../include/asema/storage.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: asema-inspect <path_to_manifest.json or model_dir>\n";
        return 1;
    }

    std::string path = argv[1];
    std::string manifest_file = path;
    if (path.find(".json") == std::string::npos) {
        manifest_file = path + "/manifest.json";
    }

    std::ifstream f(manifest_file);
    if (!f.is_open()) {
        std::cerr << "[Error] Cannot open manifest file: " << manifest_file << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string content = buffer.str();

    try {
        auto manifest = asema::ModelManifest::from_json(content);
        std::cout << "====================================================\n";
        std::cout << "             ASEMA-SSF Model Inspection             \n";
        std::cout << "====================================================\n";
        std::cout << "Format:               " << manifest.format << " v" << manifest.version << "\n";
        std::cout << "Model Name:           " << manifest.model_name << "\n";
        std::cout << "Architecture:         " << manifest.base_architecture << "\n";
        std::cout << "Total Layers:         " << manifest.num_layers << "\n";
        std::cout << "Experts per Layer:    " << manifest.experts_per_layer << "\n";
        std::cout << "Active Top-K:         " << manifest.active_experts_per_token << "\n";
        std::cout << "Hidden Dim:           " << manifest.hidden_dim << "\n";
        std::cout << "FFN Dim:              " << manifest.ffn_dim << "\n";
        std::cout << "Total Parameters:     " << manifest.total_parameters << "\n";
        std::cout << "Total Storage Size:   " << std::fixed << std::setprecision(2) 
                  << (manifest.total_storage_bytes / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "Indexed Experts:      " << manifest.lookup_index.size() << "\n";
        std::cout << "----------------------------------------------------\n";

        // Sample layer summary
        for (const auto& layer : manifest.layers) {
            std::cout << "  Layer " << std::setw(2) << layer.layer_id 
                      << ": " << layer.experts.size() << " experts";
            if (!layer.experts.empty()) {
                std::cout << " (Sample E0: offset=" << layer.experts[0].storage_offset 
                          << ", size=" << layer.experts[0].storage_length 
                          << " B, crc32=0x" << std::hex << layer.experts[0].checksum_crc32 << std::dec << ")";
            }
            std::cout << "\n";
        }
        std::cout << "====================================================\n";
    } catch (const std::exception& e) {
        std::cerr << "[Exception] Failed to parse manifest: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
