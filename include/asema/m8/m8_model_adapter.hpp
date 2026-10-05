#pragma once

// ASEMA v0.2 — M8 Real-Model Integration: model adapter
// -----------------------------------------------------------------------------
// Adapter that exposes DeepSeek-V4.1-Flash's metadata, tokenizer, and tensor
// index to the ASEMA runtime without depending on HuggingFace transformers /
// vLLM. Pure C++17; only the safetensors and JSON readers are dependencies.
//
// What this header DOES:
//   - Parse config.json (text + vision sub-configs).
//   - Parse model.safetensors.index.json (tensor -> shard map).
//   - Provide a stable ExpertCoord mapping for routed experts + shared experts.
//   - Provide the tokenizer (loaded from tokenizer.json — read-only).
//   - Provide a reference forward stub that uses the official DeepSeek
//     math (kernel.py ported to C++) on demand.
//
// What this header does NOT do (yet, in this host-constrained session):
//   - Download the 510 GB checkpoint (BLOCKED by disk capacity).
//   - Run actual inference (BLOCKED by lack of GPU + CPU-only host).
// -----------------------------------------------------------------------------

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace asema {
namespace m8 {

// -----------------------------------------------------------------------------
// Stable ASEMA coordinate for an expert tensor in DeepSeek-V4.1-Flash.
// model_id:    "deepseek-v4.1-flash"
// kind:        0 = routed expert, 1 = shared expert, 2 = engram, 3 = dspark
// layer_id:    0..num_hidden_layers-1 for routed/shared/dspark;
//              for engram, this is the index into engram_layer_ids.
// expert_id:   0..n_routed_experts-1 for routed; 0 for shared; 0..dspark_n_experts-1
// -----------------------------------------------------------------------------
struct DeepSeekExpertCoord {
    int kind{0};
    int layer_id{0};
    int expert_id{0};
};

// Shard map entry: tensor name -> shard filename.
using ShardMap = std::unordered_map<std::string, std::string>;

// Per-tensor metadata.
struct TensorMeta {
    std::string name;
    int layer_id{-1};
    int expert_id{-1};           // -1 if not an expert tensor
    int kind{0};                 // 0 routed, 1 shared, 2 engram, 3 dspark
    std::string shard;
    uint64_t offset_in_shard{0};
    uint64_t byte_size{0};
    std::vector<int64_t> shape;
    std::string dtype;          // "F8_E4M3", "BF16", "F32", etc.
};

// Group tensors by conceptual layer.
struct LayerGroup {
    int layer_id{0};
    std::vector<TensorMeta> attention_tensors;
    std::vector<TensorMeta> hc_tensors;          // hyper-connection
    std::vector<TensorMeta> shared_expert_tensors;
    std::vector<TensorMeta> routed_expert_tensors; // 384 per layer
    std::vector<TensorMeta> norm_tensors;
};

// -----------------------------------------------------------------------------
// Model metadata (parsed from config.json).
// -----------------------------------------------------------------------------
struct DeepSeekV41Config {
    std::string model_type{"deepseek_v41"};
    std::string architectures{"DeepseekV41ForCausalLM"};
    std::string torch_dtype{"bfloat16"};
    std::string quantization_method{"fp8"};

    // Text config
    int vocab_size{129280};
    int hidden_size{5120};
    int moe_intermediate_size{2304};
    int num_hidden_layers{40};
    int num_attention_heads{64};
    int num_key_value_heads{1};
    int head_dim{512};
    int qk_rope_head_dim{64};
    int q_lora_rank{1280};
    int o_lora_rank{1024};
    int o_groups{8};
    int max_position_embeddings{1048576};
    int sliding_window{128};

    int n_routed_experts{384};
    int n_shared_experts{1};
    int num_experts_per_tok{6};
    std::string scoring_func{"sqrtsoftplus"};
    std::string topk_method{"noaux_tc"};
    bool norm_topk_prob{true};
    double routed_scaling_factor{1.5};

    // Engram
    std::vector<int> engram_layer_ids{1, 14};
    std::vector<uint64_t> engram_num_embeddings{384006168, 384016682};
    int engram_max_ngram_size{4};
    uint64_t engram_vocab_size{16000000};
    int engram_n_heads{8};
    int engram_head_dim{256};
    int engram_compressed_vocab_size{99092};

    // DSpark
    int num_nextn_predict_layers{3};
    int dspark_block_size{5};
    std::vector<int> dspark_target_layer_ids{37, 38, 39};
    int dspark_n_routed_experts{128};
    int dspark_num_experts_per_tok{3};

    // Hyper-Connections
    int hc_mult{4};
    int hc_sinkhorn_iters{20};
    double hc_eps{1e-06};

    // Sparse attention / CSA2
    std::vector<int> compress_ratios;
    std::vector<int> kv_source_layer_ids{2, 8, 14, 20};
    std::vector<int> index_source_layer_ids{2, 8, 14, 20, 24, 28, 32, 36};

    // Image token id (for multimodal)
    int image_token_id{129264};

    // Vision config (subset)
    int vision_hidden_size{1024};
    int vision_num_hidden_layers{32};
    int vision_intermediate_size{2816};
    int vision_num_attention_heads{16};
    int vision_patch_size{14};
    int vision_max_image_tokens{1024};
};

// -----------------------------------------------------------------------------
// Adapter interface.
// -----------------------------------------------------------------------------
class M8ModelAdapter {
public:
    virtual ~M8ModelAdapter() = default;

    // Load config.json + index.json from `hf_root` (the local HF directory).
    virtual bool load(const std::string& hf_root) = 0;

    virtual const DeepSeekV41Config& config() const = 0;

    // Total number of expert tensors (routed + shared).
    virtual int total_routed_expert_tensors() const = 0;
    virtual int total_shared_expert_tensors() const = 0;

    // Build the per-layer grouping.
    virtual std::vector<LayerGroup> group_layers() const = 0;

    // Stable ASEMA ExpertCoord for one of the 384 routed experts in a layer.
    virtual DeepSeekExpertCoord routed_expert_coord(int layer_id, int expert_id) const = 0;
    virtual DeepSeekExpertCoord shared_expert_coord(int layer_id) const = 0;
    virtual DeepSeekExpertCoord engram_coord(int engram_index) const = 0;

    // Estimate per-expert bytes (FP8 + FP4 expert mix).
    virtual uint64_t bytes_per_routed_expert_fp8() const = 0;

    // Look up the safetensors shard for a given expert coordinate.
    virtual std::string shard_for_routed_expert(int layer_id, int expert_id) const = 0;
    virtual std::vector<std::string> shard_for_expert_tensors(int layer_id, int expert_id) const = 0;

    // Summary for the M8 architecture.json artifact.
    virtual std::string summary_json() const = 0;

    // Build a human-readable architecture report.
    virtual std::string report_markdown() const = 0;
};

// Concrete adapter for the local HF directory layout.
std::unique_ptr<M8ModelAdapter> make_deepseek_v41_adapter();

// ---------------------------------------------------------------------------
// Tokenizer (lightweight; does not require model weights).
// Reads tokenizer.json and exposes encode / decode.
// ---------------------------------------------------------------------------
class Tokenizer {
public:
    bool load(const std::string& tokenizer_json_path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const { return vocab_size_; }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int pad_id() const { return pad_id_; }

private:
    int vocab_size_{0};
    int bos_id_{0};
    int eos_id_{1};
    int pad_id_{2};
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> bpe_ranks_;
    std::unordered_map<uint32_t, uint8_t> u2b_;
    std::unordered_map<uint8_t, std::string> b2u_;
    std::vector<std::string> special_tokens_;

    void init_byte_mappings();
    std::vector<std::string> bpe_word(const std::string& word) const;
};

} // namespace m8
} // namespace asema
