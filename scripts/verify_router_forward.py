import os
SHARDS = os.environ.get("ASEMA_SHARDS_PRIMARY", "models/DeepSeek-V4.1-Flash/shards")
import numpy as np

# Load router weights & biases
with open(os.path.join(SHARDS, "l0_gate_weight.bin"), "rb") as f:
    buf_w = f.read()

with open(os.path.join(SHARDS, "l0_gate_bias.bin"), "rb") as f:
    buf_b = f.read()

# BF16 weight
u16 = np.frombuffer(buf_w, dtype=np.uint16)
u32 = (u16.astype(np.uint32) << 16)
weights = u32.view(np.float32).reshape(384, 5120)

# FP32 bias
# Note: gate_bias.bin may contain bias and bias_vl. Let's check size:
print(f"Bias buffer size: {len(buf_b)} bytes")
biases = np.frombuffer(buf_b, dtype=np.float32)
bias = biases[:384]

# Load reference x
x = np.fromfile("m8_ref_x.bin", dtype=np.float32)

# Compute logits
logits = np.dot(weights, x)

# sqrtsoftplus: scores = sqrt(softplus(logits)) = sqrt(log(1 + exp(logits)))
# Stable computation: where logits > 20, softplus(logits) ≈ logits
softplus = np.where(logits > 20.0, logits, np.log1p(np.exp(np.clip(logits, -80.0, 20.0))))
scores = np.sqrt(softplus)

# Bias steers selection only
selection = scores + bias

# Top-6 selection
topk = 6
top_indices = np.argsort(selection)[::-1][:topk]
top_scores = scores[top_indices]

# Normalize top weights
top_weights = top_scores / (np.sum(top_scores) + 1e-20)
# Route scale 1.5
top_weights = top_weights * 1.5

print(f"Top-{topk} selected expert indices: {list(top_indices)}")
print(f"Top-{topk} router weights: {[round(float(w), 6) for w in top_weights]}")

np.save("m8_ref_router_indices.npy", top_indices.astype(np.int32))
np.save("m8_ref_router_weights.npy", top_weights.astype(np.float32))
top_indices.astype(np.int32).tofile("m8_ref_router_indices.bin")
top_weights.astype(np.float32).tofile("m8_ref_router_weights.bin")
print("Saved router reference outputs to m8_ref_router_indices.bin and m8_ref_router_weights.bin")
