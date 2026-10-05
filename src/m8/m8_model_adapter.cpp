// ASEMA v0.2 — M8 DeepSeek-V4.1-Flash model adapter implementation.
// -----------------------------------------------------------------------------
// Pure C++17 adapter. Parses the official Hugging Face config.json and
// model.safetensors.index.json into structured types using nlohmann/json.
// Does NOT depend on transformers / vLLM. The full safetensors shards are
// NOT downloaded (BLOCKED by disk capacity in this host environment).
// -----------------------------------------------------------------------------

#include "../../include/asema/m8/m8_model_adapter.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <regex>
#include <set>
#include <sstream>

namespace asema {
namespace m8 {

// -----------------------------------------------------------------------------
// Adapter implementation
// -----------------------------------------------------------------------------
class DeepSeekV41AdapterImpl : public M8ModelAdapter {
public:
    bool load(const std::string& hf_root) override {
        hf_root_ = hf_root;

        // Load config.json
        std::ifstream f(hf_root + "/config.json");
        if (!f.is_open()) return false;
        nlohmann::json j;
        try {
            f >> j;
        } catch (...) {
            return false;
        }

        // Top-level fields
        if (j.contains("model_type"))      cfg_.model_type = j["model_type"].get<std::string>();
        if (j.contains("dtype"))           cfg_.torch_dtype = j["dtype"].get<std::string>();
        if (j.contains("quantization_config")) {
            const auto& qc = j["quantization_config"];
            if (qc.contains("quant_method")) cfg_.quantization_method = qc["quant_method"].get<std::string>();
        }
        // architectures is an array; take first
        if (j.contains("architectures") && j["architectures"].is_array() && !j["architectures"].empty()) {
            cfg_.architectures = j["architectures"][0].get<std::string>();
        }

        // Text config (flat keys live alongside the nested text_config)
        auto get_int = [&](const std::string& k, int fb) {
            return j.contains(k) && j[k].is_number_integer() ? j[k].get<int>() : fb;
        };
        auto get_double = [&](const std::string& k, double fb) {
            return j.contains(k) && j[k].is_number() ? j[k].get<double>() : fb;
        };
        auto get_str = [&](const std::string& k, const std::string& fb) {
            return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : fb;
        };
        auto get_bool = [&](const std::string& k, bool fb) {
            return j.contains(k) && j[k].is_boolean() ? j[k].get<bool>() : fb;
        };
        auto get_int_arr = [&](const std::string& k, std::vector<int>& dst) {
            if (j.contains(k) && j[k].is_array()) {
                for (auto& v : j[k]) if (v.is_number_integer()) dst.push_back(v.get<int>());
            }
        };

        cfg_.vocab_size              = get_int("vocab_size", cfg_.vocab_size);
        cfg_.hidden_size             = get_int("hidden_size", cfg_.hidden_size);
        cfg_.moe_intermediate_size   = get_int("moe_intermediate_size", cfg_.moe_intermediate_size);
        cfg_.num_hidden_layers       = get_int("num_hidden_layers", cfg_.num_hidden_layers);
        cfg_.num_attention_heads     = get_int("num_attention_heads", cfg_.num_attention_heads);
        cfg_.num_key_value_heads     = get_int("num_key_value_heads", cfg_.num_key_value_heads);
        cfg_.head_dim                = get_int("head_dim", cfg_.head_dim);
        cfg_.qk_rope_head_dim         = get_int("qk_rope_head_dim", cfg_.qk_rope_head_dim);
        cfg_.q_lora_rank              = get_int("q_lora_rank", cfg_.q_lora_rank);
        cfg_.o_lora_rank              = get_int("o_lora_rank", cfg_.o_lora_rank);
        cfg_.o_groups                 = get_int("o_groups", cfg_.o_groups);
        cfg_.max_position_embeddings  = get_int("max_position_embeddings", cfg_.max_position_embeddings);
        cfg_.sliding_window           = get_int("sliding_window", cfg_.sliding_window);
        cfg_.n_routed_experts         = get_int("n_routed_experts", cfg_.n_routed_experts);
        cfg_.n_shared_experts         = get_int("n_shared_experts", cfg_.n_shared_experts);
        cfg_.num_experts_per_tok      = get_int("num_experts_per_tok", cfg_.num_experts_per_tok);
        cfg_.scoring_func             = get_str("scoring_func", cfg_.scoring_func);
        cfg_.topk_method              = get_str("topk_method", cfg_.topk_method);
        cfg_.norm_topk_prob           = get_bool("norm_topk_prob", cfg_.norm_topk_prob);
        cfg_.routed_scaling_factor    = get_double("routed_scaling_factor", cfg_.routed_scaling_factor);
        cfg_.image_token_id           = get_int("image_token_id", cfg_.image_token_id);
        get_int_arr("engram_layer_ids", cfg_.engram_layer_ids);
        get_int_arr("compress_ratios", cfg_.compress_ratios);
        get_int_arr("kv_source_layer_ids", cfg_.kv_source_layer_ids);
        get_int_arr("index_source_layer_ids", cfg_.index_source_layer_ids);
        if (j.contains("engram_num_embeddings") && j["engram_num_embeddings"].is_array()) {
            for (auto& v : j["engram_num_embeddings"]) {
                if (v.is_number_integer()) cfg_.engram_num_embeddings.push_back(v.get<uint64_t>());
            }
        }
        cfg_.num_nextn_predict_layers  = get_int("num_nextn_predict_layers", 3);
        cfg_.dspark_block_size         = get_int("dspark_block_size", 5);
        get_int_arr("dspark_target_layer_ids", cfg_.dspark_target_layer_ids);
        cfg_.dspark_n_routed_experts   = get_int("dspark_n_routed_experts", 128);
        cfg_.dspark_num_experts_per_tok = get_int("dspark_num_experts_per_tok", 3);
        cfg_.hc_mult            = get_int("hc_mult", 4);
        cfg_.hc_sinkhorn_iters  = get_int("hc_sinkhorn_iters", 20);
        cfg_.hc_eps             = get_double("hc_eps", 1e-6);
        cfg_.vision_hidden_size        = get_int("vision_hidden_size", cfg_.vision_hidden_size);
        cfg_.vision_num_hidden_layers   = get_int("vision_num_hidden_layers", cfg_.vision_num_hidden_layers);
        cfg_.vision_intermediate_size  = get_int("vision_intermediate_size", cfg_.vision_intermediate_size);
        cfg_.vision_num_attention_heads = get_int("vision_num_attention_heads", cfg_.vision_num_attention_heads);
        cfg_.vision_patch_size          = get_int("vision_patch_size", cfg_.vision_patch_size);
        cfg_.vision_max_image_tokens    = get_int("vision_max_image_tokens", cfg_.vision_max_image_tokens);

        // Load index.json
        std::ifstream fi(hf_root + "/model.safetensors.index.json");
        if (!fi.is_open()) return false;
        nlohmann::json ij;
        try {
            fi >> ij;
        } catch (...) {
            return false;
        }
        if (!ij.contains("weight_map") || !ij["weight_map"].is_object()) return false;

        for (auto& [name, shard] : ij["weight_map"].items()) {
            std::string s = shard.get<std::string>();
            shards_[name] = s;
            TensorMeta t;
            t.name = name;
            t.shard = s;
            classify_tensor(t);
            tensors_.push_back(t);
        }
        return true;
    }

    const DeepSeekV41Config& config() const override { return cfg_; }

    int total_routed_expert_tensors() const override {
        return cfg_.n_routed_experts * cfg_.num_hidden_layers * 3;
    }
    int total_shared_expert_tensors() const override {
        return cfg_.n_shared_experts * cfg_.num_hidden_layers * 3;
    }

    std::vector<LayerGroup> group_layers() const override {
        std::vector<LayerGroup> groups(cfg_.num_hidden_layers);
        for (int i = 0; i < cfg_.num_hidden_layers; ++i) groups[i].layer_id = i;
        for (const auto& t : tensors_) {
            if (t.layer_id < 0 || t.layer_id >= cfg_.num_hidden_layers) continue;
            if (t.name.find("hc_") != std::string::npos) {
                groups[t.layer_id].hc_tensors.push_back(t);
            } else if (t.name.find("attn") != std::string::npos) {
                groups[t.layer_id].attention_tensors.push_back(t);
            } else if (t.name.find("shared_experts") != std::string::npos) {
                groups[t.layer_id].shared_expert_tensors.push_back(t);
            } else if (t.name.find("experts.") != std::string::npos) {
                groups[t.layer_id].routed_expert_tensors.push_back(t);
            } else if (t.name.find("_norm") != std::string::npos) {
                groups[t.layer_id].norm_tensors.push_back(t);
            }
        }
        return groups;
    }

    DeepSeekExpertCoord routed_expert_coord(int layer_id, int expert_id) const override {
        return {0, layer_id, expert_id};
    }
    DeepSeekExpertCoord shared_expert_coord(int layer_id) const override {
        return {1, layer_id, 0};
    }
    DeepSeekExpertCoord engram_coord(int engram_index) const override {
        return {2, engram_index, 0};
    }

    uint64_t bytes_per_routed_expert_fp8() const override {
        uint64_t per_tensor = static_cast<uint64_t>(cfg_.hidden_size) *
                              static_cast<uint64_t>(cfg_.moe_intermediate_size);
        return per_tensor * 3ull;
    }

    std::string shard_for_routed_expert(int layer_id, int expert_id) const override {
        // Weight-map keys are like "layers.0.ffn.experts.0.w1.weight" (no "model." prefix).
        std::string w1 = "layers." + std::to_string(layer_id) + ".ffn.experts." +
                          std::to_string(expert_id) + ".w1.weight";
        auto it = shards_.find(w1);
        return it == shards_.end() ? "" : it->second;
    }

    std::vector<std::string> shard_for_expert_tensors(int layer_id, int expert_id) const override {
        std::vector<std::string> out;
        for (const char* t : {"w1", "w3", "w2"}) {
            std::string w = "layers." + std::to_string(layer_id) + ".ffn.experts." +
                             std::to_string(expert_id) + "." + t + ".weight";
            auto it = shards_.find(w);
            if (it != shards_.end()) out.push_back(it->second);
        }
        return out;
    }

    std::string summary_json() const override {
        nlohmann::json j;
        j["model_type"] = cfg_.model_type;
        j["architectures"] = cfg_.architectures;
        j["vocab_size"] = cfg_.vocab_size;
        j["hidden_size"] = cfg_.hidden_size;
        j["num_hidden_layers"] = cfg_.num_hidden_layers;
        j["n_routed_experts"] = cfg_.n_routed_experts;
        j["n_shared_experts"] = cfg_.n_shared_experts;
        j["num_experts_per_tok"] = cfg_.num_experts_per_tok;
        j["moe_intermediate_size"] = cfg_.moe_intermediate_size;
        j["scoring_func"] = cfg_.scoring_func;
        j["topk_method"] = cfg_.topk_method;
        j["quantization_method"] = cfg_.quantization_method;
        j["bytes_per_routed_expert_fp8"] = bytes_per_routed_expert_fp8();
        j["total_routed_expert_tensors"] = total_routed_expert_tensors();
        j["total_tensors_indexed"] = tensors_.size();
        j["unique_shards"] = unique_shard_count();
        j["hf_root"] = hf_root_;
        return j.dump(2);
    }

    std::string report_markdown() const override {
        std::ostringstream os;
        os << "# DeepSeek-V4.1-Flash — ASEMA M8 Architecture Report\n\n";
        os << "Source: `deepseek-ai/DeepSeek-V4.1-Flash` (commit `dba1be0a`)\n\n";
        os << "## Model\n\n";
        os << "| Field | Value |\n|---|---|\n";
        os << "| Architecture | " << cfg_.architectures << " |\n";
        os << "| Total params (HF) | 763 B |\n";
        os << "| Active per token (prefill) | 8 B |\n";
        os << "| Active per token (decode)  | 16 B |\n";
        os << "| Checkpoint size | 510 GB / 48 shards |\n";
        os << "| Tensors indexed by adapter | " << tensors_.size() << " |\n";
        os << "| Unique shards in index     | " << unique_shard_count() << " |\n\n";
        os << "## Text config\n\n";
        os << "| Field | Value |\n|---|---|\n";
        os << "| vocab_size | " << cfg_.vocab_size << " |\n";
        os << "| hidden_size | " << cfg_.hidden_size << " |\n";
        os << "| moe_intermediate_size | " << cfg_.moe_intermediate_size << " |\n";
        os << "| num_hidden_layers | " << cfg_.num_hidden_layers << " |\n";
        os << "| num_attention_heads | " << cfg_.num_attention_heads << " |\n";
        os << "| num_key_value_heads (MLA) | " << cfg_.num_key_value_heads << " |\n";
        os << "| head_dim | " << cfg_.head_dim << " |\n";
        os << "| q_lora_rank | " << cfg_.q_lora_rank << " |\n";
        os << "| o_lora_rank | " << cfg_.o_lora_rank << " |\n";
        os << "| max_position_embeddings | " << cfg_.max_position_embeddings << " |\n";
        os << "| n_routed_experts | " << cfg_.n_routed_experts << " |\n";
        os << "| n_shared_experts | " << cfg_.n_shared_experts << " |\n";
        os << "| num_experts_per_tok | " << cfg_.num_experts_per_tok << " |\n";
        os << "| scoring_func | " << cfg_.scoring_func << " |\n";
        os << "| topk_method | " << cfg_.topk_method << " |\n";
        os << "| quantization | " << cfg_.quantization_method << " (block 32x32, ue8m0) |\n\n";
        os << "## ASEMA-coordinate mapping\n\n";
        os << "| Kind | Layer | Expert | Total tensors |\n|---|---|---|---|\n";
        os << "| Routed expert | 0.." << (cfg_.num_hidden_layers - 1) << " | 0.."
           << (cfg_.n_routed_experts - 1) << " | " << total_routed_expert_tensors() << " |\n";
        os << "| Shared expert | 0.." << (cfg_.num_hidden_layers - 1) << " | 0 | "
           << total_shared_expert_tensors() << " |\n";
        os << "| Engram | " << cfg_.engram_layer_ids.size() << " entries | 0 | (not in weight_map) |\n";
        os << "| DSpark draft MoE | " << cfg_.num_nextn_predict_layers << " | 0.."
           << (cfg_.dspark_n_routed_experts - 1) << " | (not in weight_map) |\n";
        os << "\n";
        os << "Per routed expert: 6 tensors (w1/w3/w2 each with weight + scale).\n";
        os << "Total routed expert tensors indexed: "
           << cfg_.n_routed_experts * cfg_.num_hidden_layers * 6 << "\n";
        os << "\n## Estimated per-expert size (FP8)\n\n";
        os << "- gate (w1): " << cfg_.hidden_size << " x " << cfg_.moe_intermediate_size
           << " = " << (cfg_.hidden_size * cfg_.moe_intermediate_size) << " B FP8\n";
        os << "- up   (w3): " << cfg_.hidden_size << " x " << cfg_.moe_intermediate_size
           << " = " << (cfg_.hidden_size * cfg_.moe_intermediate_size) << " B FP8\n";
        os << "- down (w2): " << cfg_.moe_intermediate_size << " x " << cfg_.hidden_size
           << " = " << (cfg_.moe_intermediate_size * cfg_.hidden_size) << " B FP8\n";
        os << "- **per-expert total ~= "
           << (bytes_per_routed_expert_fp8() / (1024.0 * 1024.0))
           << " MB FP8**\n";
        os << "- Total routed expert weights: "
           << (cfg_.n_routed_experts * cfg_.num_hidden_layers *
               bytes_per_routed_expert_fp8() / (1024.0 * 1024.0 * 1024.0))
           << " GB\n";
        os << "\n## Status on this host\n\n";
        os << "- Config + tokenizer + index: **downloaded** ("
           << tensors_.size() << " tensors across " << unique_shard_count() << " shards indexed)\n";
        os << "- Safetensors shards: **NOT downloaded** (510 GB; host has 71 GB free)\n";
        os << "- Reference forward: **BLOCKED** (no GPU; CPU insufficient for 552 B params)\n";
        return os.str();
    }

private:
    void classify_tensor(TensorMeta& t) {
        std::regex layer_re("layers\\.(\\d+)\\.");
        std::smatch m;
        if (std::regex_search(t.name, m, layer_re)) {
            t.layer_id = std::stoi(m[1].str());
        }
        if (t.name.find("experts.") != std::string::npos) {
            std::regex expert_re("experts\\.(\\d+)\\.");
            if (std::regex_search(t.name, m, expert_re)) {
                t.expert_id = std::stoi(m[1].str());
            }
            t.kind = 0;
        } else if (t.name.find("shared_experts") != std::string::npos) {
            t.kind = 1;
        } else if (t.name.find("engram") != std::string::npos) {
            t.kind = 2;
        } else if (t.name.find("dspark") != std::string::npos) {
            t.kind = 3;
        }
    }

    int unique_shard_count() const {
        std::set<std::string> s;
        for (const auto& kv : shards_) s.insert(kv.second);
        return static_cast<int>(s.size());
    }

    DeepSeekV41Config cfg_;
    ShardMap shards_;
    std::vector<TensorMeta> tensors_;
    std::string hf_root_;
};

std::unique_ptr<M8ModelAdapter> make_deepseek_v41_adapter() {
    return std::make_unique<DeepSeekV41AdapterImpl>();
}

// -----------------------------------------------------------------------------
// Byte-Level BPE Tokenizer.
// -----------------------------------------------------------------------------
void Tokenizer::init_byte_mappings() {
    u2b_.clear();
    b2u_.clear();
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    std::vector<bool> in_bs(256, false);
    for (int b : bs) in_bs[b] = true;

    for (int b = 0; b < 256; ++b) {
        if (!in_bs[b]) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    auto codepoint_to_utf8 = [](uint32_t cp) -> std::string {
        std::string s;
        if (cp <= 0x7F) {
            s += static_cast<char>(cp);
        } else if (cp <= 0x7FF) {
            s += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp <= 0xFFFF) {
            s += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp <= 0x10FFFF) {
            s += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
            s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return s;
    };

    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t b = static_cast<uint8_t>(bs[i]);
        uint32_t cp = static_cast<uint32_t>(cs[i]);
        b2u_[b] = codepoint_to_utf8(cp);
        u2b_[cp] = b;
    }
}

std::vector<std::string> Tokenizer::bpe_word(const std::string& word_str) const {
    std::vector<std::string> word;
    size_t idx = 0;
    while (idx < word_str.size()) {
        size_t char_len = 1;
        unsigned char c = static_cast<unsigned char>(word_str[idx]);
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;
        if (idx + char_len > word_str.size()) char_len = 1;
        word.push_back(word_str.substr(idx, char_len));
        idx += char_len;
    }

    while (word.size() > 1) {
        int min_rank = 1000000000;
        int min_idx = -1;
        for (size_t i = 0; i < word.size() - 1; ++i) {
            std::string pair = word[i] + "\x1f" + word[i+1];
            auto it = bpe_ranks_.find(pair);
            if (it != bpe_ranks_.end()) {
                if (it->second < min_rank) {
                    min_rank = it->second;
                    min_idx = static_cast<int>(i);
                }
            }
        }
        if (min_idx == -1) break;

        std::vector<std::string> new_word;
        new_word.reserve(word.size() - 1);
        for (size_t i = 0; i < word.size(); ++i) {
            if (static_cast<int>(i) == min_idx) {
                new_word.push_back(word[i] + word[i+1]);
                ++i;
            } else {
                new_word.push_back(word[i]);
            }
        }
        word = std::move(new_word);
    }
    return word;
}

bool Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path);
    if (!f.is_open()) return false;
    nlohmann::json j;
    try { f >> j; } catch (...) { return false; }

    if (j.contains("bos_token_id")) bos_id_ = j["bos_token_id"].get<int>();
    if (j.contains("eos_token_id")) eos_id_ = j["eos_token_id"].get<int>();
    if (j.contains("pad_token_id")) pad_id_ = j["pad_token_id"].get<int>();

    id_to_token_.clear();
    id_to_token_.resize(300000, "");
    token_to_id_.clear();

    special_tokens_.clear();
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (auto& tok : j["added_tokens"]) {
            if (!tok.is_object()) continue;
            if (!tok.contains("id") || !tok.contains("content")) continue;
            int id = tok["id"].get<int>();
            std::string content = tok["content"].get<std::string>();
            if (id >= 0 && id < (int)id_to_token_.size()) {
                id_to_token_[id] = content;
                token_to_id_[content] = id;
                if (content.rfind("<｜", 0) == 0 || content.rfind("<|", 0) == 0 ||
                    content == "<think>" || content == "</think>") {
                    special_tokens_.push_back(content);
                }
            }
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](const std::string& a, const std::string& b) {
        return a.size() > b.size();
    });

    if (j.contains("model") && j["model"].is_object() && j["model"].contains("vocab")) {
        for (auto it = j["model"]["vocab"].begin(); it != j["model"]["vocab"].end(); ++it) {
            std::string key = it.key();
            int id = -1;
            try {
                if (it.value().is_number_integer()) {
                    id = it.value().get<int>();
                } else if (it.value().is_string()) {
                    id = std::stoi(it.value().get<std::string>());
                }
            } catch (...) { continue; }
            if (id >= 0 && id < (int)id_to_token_.size() && id_to_token_[id].empty()) {
                id_to_token_[id] = key;
                token_to_id_[key] = id;
            }
        }
    }

    bpe_ranks_.clear();
    if (j.contains("model") && j["model"].is_object() && j["model"].contains("merges") && j["model"]["merges"].is_array()) {
        int rank = 0;
        for (const auto& m_val : j["model"]["merges"]) {
            if (!m_val.is_string()) continue;
            std::string m = m_val.get<std::string>();
            size_t sp = m.find(' ');
            if (sp != std::string::npos) {
                std::string p1 = m.substr(0, sp);
                std::string p2 = m.substr(sp + 1);
                bpe_ranks_[p1 + "\x1f" + p2] = rank++;
            }
        }
    }

    init_byte_mappings();

    // The table was over-allocated while loading; the real vocabulary is the highest populated
    // id + 1. Trim so vocab_size() and decode()'s bounds check reflect the actual vocabulary.
    int highest = -1;
    for (int i = static_cast<int>(id_to_token_.size()) - 1; i >= 0; --i) {
        if (!id_to_token_[i].empty()) { highest = i; break; }
    }
    id_to_token_.resize(static_cast<size_t>(highest + 1));
    id_to_token_.shrink_to_fit();
    vocab_size_ = static_cast<int>(id_to_token_.size());
    return vocab_size_ > 0 && !bpe_ranks_.empty();
}

static bool is_ws_char(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static bool is_ascii_letter_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool is_digit_char(char c) {
    return c >= '0' && c <= '9';
}

static size_t utf8_char_len(unsigned char c) {
    if (c >= 0xF0) return 4;
    if (c >= 0xE0) return 3;
    if (c >= 0xC0) return 2;
    return 1;
}

static bool is_letter_or_unicode(const std::string& text, size_t i) {
    if (i >= text.size()) return false;
    unsigned char c = static_cast<unsigned char>(text[i]);
    return is_ascii_letter_char(text[i]) || (c >= 128);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    size_t pos = 0;
    size_t n = text.size();

    while (pos < n) {
        // 1. Check for special tokens
        bool is_special = false;
        if (text[pos] == '<') {
            for (const auto& spec : special_tokens_) {
                if (pos + spec.size() <= n && text.compare(pos, spec.size(), spec) == 0) {
                    auto it = token_to_id_.find(spec);
                    if (it != token_to_id_.end()) {
                        out.push_back(it->second);
                        pos += spec.size();
                        is_special = true;
                        break;
                    }
                }
            }
        }
        if (is_special) continue;

        // 2. Extract regular text segment up to next special token or end
        size_t next_spec = text.find("<｜", pos);
        if (next_spec == std::string::npos) next_spec = text.find("<|", pos);
        if (next_spec == std::string::npos) next_spec = n;

        std::string regular_text = text.substr(pos, next_spec - pos);
        pos = next_spec;

        // 3. Tokenize regular text segment
        std::vector<std::string> chunks;
        size_t i = 0;
        size_t rn = regular_text.size();
        while (i < rn) {
            // Contractions: 're, 've, 'll, 's, 't, 'm, 'd
            if (regular_text[i] == '\'' && i + 1 < rn) {
                std::string rest = regular_text.substr(i);
                for (auto& c : rest) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
                const char* conts[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
                std::string matched_cont;
                for (const char* c : conts) {
                    size_t clen = strlen(c);
                    if (rest.rfind(c, 0) == 0) {
                        matched_cont = regular_text.substr(i, clen);
                        break;
                    }
                }
                if (!matched_cont.empty()) {
                    chunks.push_back(matched_cont);
                    i += matched_cont.size();
                    continue;
                }
            }

            size_t start = i;
            bool has_space = false;
            if (regular_text[i] == ' ') {
                has_space = true;
                i++;
                if (i == rn) {
                    chunks.push_back(" ");
                    break;
                }
            }

            if (i < rn && is_ws_char(regular_text[i])) {
                while (i < rn && is_ws_char(regular_text[i])) ++i;
                chunks.push_back(regular_text.substr(start, i - start));
                continue;
            }

            if (i < rn && is_letter_or_unicode(regular_text, i)) {
                while (i < rn && is_letter_or_unicode(regular_text, i)) {
                    i += utf8_char_len(static_cast<unsigned char>(regular_text[i]));
                }
                chunks.push_back(regular_text.substr(start, i - start));
                continue;
            }

            if (i < rn && is_digit_char(regular_text[i])) {
                int d_count = 0;
                while (i < rn && is_digit_char(regular_text[i]) && d_count < 3) {
                    ++i;
                    ++d_count;
                }
                chunks.push_back(regular_text.substr(start, i - start));
                continue;
            }

            if (i < rn && !is_ws_char(regular_text[i])) {
                while (i < rn && !is_letter_or_unicode(regular_text, i) && !is_digit_char(regular_text[i]) && !is_ws_char(regular_text[i])) {
                    ++i;
                }
                chunks.push_back(regular_text.substr(start, i - start));
                continue;
            }

            if (has_space && i == start + 1) {
                chunks.push_back(" ");
            }
        }

        for (const auto& chunk : chunks) {
            std::string b_str;
            for (unsigned char b : chunk) {
                auto it_b = b2u_.find(b);
                if (it_b != b2u_.end()) {
                    b_str += it_b->second;
                } else {
                    b_str += static_cast<char>(b);
                }
            }
            auto subwords = bpe_word(b_str);
            for (const auto& sw : subwords) {
                auto it = token_to_id_.find(sw);
                if (it != token_to_id_.end()) {
                    out.push_back(it->second);
                }
            }
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out_bytes;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_token_.size()) continue;
        if (id == bos_id_ || id == eos_id_) continue;
        const std::string& tok = id_to_token_[id];
        if (tok.empty()) continue;

        size_t i = 0;
        while (i < tok.size()) {
            unsigned char c = static_cast<unsigned char>(tok[i]);
            uint32_t cp = 0;
            size_t clen = 1;
            if (c < 0x80) {
                cp = c;
                clen = 1;
            } else if ((c & 0xE0) == 0xC0) {
                if (i + 1 < tok.size()) {
                    cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(tok[i + 1]) & 0x3F);
                    clen = 2;
                }
            } else if ((c & 0xF0) == 0xE0) {
                if (i + 2 < tok.size()) {
                    cp = ((c & 0x0F) << 12) | ((static_cast<unsigned char>(tok[i + 1]) & 0x3F) << 6) | (static_cast<unsigned char>(tok[i + 2]) & 0x3F);
                    clen = 3;
                }
            } else if ((c & 0xF8) == 0xF0) {
                if (i + 3 < tok.size()) {
                    cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(tok[i + 1]) & 0x3F) << 12) | ((static_cast<unsigned char>(tok[i + 2]) & 0x3F) << 6) | (static_cast<unsigned char>(tok[i + 3]) & 0x3F);
                    clen = 4;
                }
            }
            i += clen;

            auto it = u2b_.find(cp);
            if (it != u2b_.end()) {
                out_bytes += static_cast<char>(it->second);
            } else {
                if (cp <= 0x7F) out_bytes += static_cast<char>(cp);
                else {
                    out_bytes += tok.substr(i - clen, clen);
                }
            }
        }
    }
    return out_bytes;
}

} // namespace m8
} // namespace asema
