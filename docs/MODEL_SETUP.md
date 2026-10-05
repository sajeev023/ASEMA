# Model setup

ASEMA does not ship model weights. Obtain the DeepSeek-V4.1-Flash checkpoint from its official
source and accept its license. The checkpoint is 48 `model-XXXXX-of-00048.safetensors` shards
(about 510 GB), plus `model.safetensors.index.json`, `tokenizer.json` and `config.json`.

## Layout

Shards may be placed in one directory or split across two (for example across two drives).
ASEMA resolves each tensor to whichever directory contains its shard.

```
<shards_primary>/model-00001-of-00048.safetensors ...
<shards_secondary>/model-00033-of-00048.safetensors ...   (optional)
<model_root>/model.safetensors.index.json
<hf_dir>/tokenizer.json, config.json
```

## Configure

Copy `asema.config.example` to `asema.config`, or set environment variables (they take precedence):

| Key | Environment variable |
|---|---|
| `shards_primary` | `ASEMA_SHARDS_PRIMARY` |
| `shards_secondary` | `ASEMA_SHARDS_SECONDARY` |
| `model_root` | `ASEMA_MODEL_ROOT` |
| `hf_dir` | `ASEMA_HF_DIR` |

Then run `asema verify-model` and `asema doctor`.

## Storage advice

Throughput of the drive holding the shards is the dominant factor in decode speed
(see PERFORMANCE.md). NVMe was measured at about 6x a SATA SSD for expert-sized random reads.
If you split shards over two drives, put as many as possible on the faster one.
