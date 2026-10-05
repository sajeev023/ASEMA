#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_multi_volume.hpp"
#include <iostream>

int main() {
    auto vol_mgr = std::make_shared<asema::m8::M8MultiVolumeManager>();
    vol_mgr->register_volume(asema::m8::paths::primary_shards());
    // vol_mgr->load_index("examples/real_model/DeepSeek-V4.1-Flash/hf/model.safetensors.index.json");

    auto loc0 = vol_mgr->locate_expert(0, 0);
    std::cout << "Expert 0: present=" << loc0.all_tensors_present 
              << " path=" << loc0.physical_path 
              << " w1_scale_off=" << loc0.w1_scale.offset_in_file 
              << " w1_weight_off=" << loc0.w1_weight.offset_in_file << "\n";

    for (int id : { 218, 247, 319, 203, 16, 201 }) {
        auto loc = vol_mgr->locate_expert(0, id);
        std::cout << "Expert " << id << ": present=" << loc.all_tensors_present 
                  << " path=" << loc.physical_path 
                  << " w1_scale_off=" << loc.w1_scale.offset_in_file 
                  << " w1_weight_off=" << loc.w1_weight.offset_in_file << "\n";
    }

    auto loc1_0 = vol_mgr->locate_expert(1, 0);
    std::cout << "Layer 1 Expert 0: present=" << loc1_0.all_tensors_present 
              << " path=" << loc1_0.physical_path 
              << " w1_scale_off=" << loc1_0.w1_scale.offset_in_file 
              << " w1_weight_off=" << loc1_0.w1_weight.offset_in_file << "\n";

    return 0;
}
