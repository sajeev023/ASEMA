import os
SHARDS = os.environ.get("ASEMA_SHARDS_PRIMARY", "models/DeepSeek-V4.1-Flash/shards")
import numpy as np

def unpack_fp4(packed_bytes, shape):
    u8 = packed_bytes.reshape(shape[0], shape[1] // 2)
    low_nib = (u8 & 0x0F)
    high_nib = ((u8 >> 4) & 0x0F)
    fp4_lut = np.array([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=np.float32)
    unpacked = np.empty(shape, dtype=np.float32)
    unpacked[:, 0::2] = fp4_lut[low_nib]
    unpacked[:, 1::2] = fp4_lut[high_nib]
    return unpacked

def dequantize(packed_bytes, scale_bytes, shape):
    rows, cols = shape
    blocks = cols // 32
    scales = (2.0 ** (scale_bytes.reshape(rows, blocks).astype(np.float32) - 127.0))
    fp4 = unpack_fp4(packed_bytes, shape)
    w_dequant = np.empty(shape, dtype=np.float32)
    for b in range(blocks):
        w_dequant[:, b*32 : (b+1)*32] = fp4[:, b*32 : (b+1)*32] * scales[:, b:b+1]
    return w_dequant

# Load scales
with open(os.path.join(SHARDS, "e0_scales.bin"), "rb") as f:
    scales_raw = f.read()

# w1.scale: 368640 bytes [2304, 160]
w1_scale = np.frombuffer(scales_raw[0:368640], dtype=np.uint8)
# w2.scale: 368640 bytes [5120, 72]
w2_scale = np.frombuffer(scales_raw[368640:368640*2], dtype=np.uint8)
# w3.scale: 368640 bytes [2304, 160]
w3_scale = np.frombuffer(scales_raw[368640*2:368640*3], dtype=np.uint8)

# Load weights
with open(os.path.join(SHARDS, "e0_weights.bin"), "rb") as f:
    weights_raw = f.read()

sz = 5898240
# w1.weight: 5898240 bytes [2304, 2560]
w1_weight = np.frombuffer(weights_raw[0:sz], dtype=np.uint8)
# w2.weight: 5898240 bytes [5120, 1152]
w2_weight = np.frombuffer(weights_raw[sz:sz*2], dtype=np.uint8)
# w3.weight: 5898240 bytes [2304, 2560]
w3_weight = np.frombuffer(weights_raw[sz*2:sz*3], dtype=np.uint8)

print("Dequantizing w1 [2304, 5120]...")
w1 = dequantize(w1_weight, w1_scale, (2304, 5120))
print("Dequantizing w2 [5120, 2304]...")
w2 = dequantize(w2_weight, w2_scale, (5120, 2304))
print("Dequantizing w3 [2304, 5120]...")
w3 = dequantize(w3_weight, w3_scale, (2304, 5120))

# Test input vector x (5120 elements)
import os
if os.path.exists("m8_ref_x.bin"):
    x = np.fromfile("m8_ref_x.bin", dtype=np.float32)
else:
    np.random.seed(42)
    x = np.random.randn(5120).astype(np.float32) * 0.05

# Forward pass:
# gate = w1 @ x
gate = np.dot(w1, x)
# up = w3 @ x
up = np.dot(w3, x)
# SwiGLU activation with clamp (matching dims_.swiglu_limit = 2.0)
gate = np.minimum(gate, 2.0)
up = np.clip(up, -2.0, 2.0)
silu_gate = gate / (1.0 + np.exp(-gate))
act = silu_gate * up
# out = w2 @ act
out = np.dot(w2, act)

print(f"Expert 0 Output shape: {out.shape}")
print(f"Output Mean: {float(np.mean(out)):.8f}")
print(f"Output Std: {float(np.std(out)):.8f}")
print(f"Output Min: {float(np.min(out)):.8f}, Max: {float(np.max(out)):.8f}")
print("First 8 output values:", [round(float(v), 6) for v in out[:8]])

# Save reference input and output for C++ equivalence test!
np.save("m8_ref_x.npy", x)
np.save("m8_ref_out.npy", out)
x.tofile("m8_ref_x.bin")
out.tofile("m8_ref_out.bin")
print("Saved m8_ref_x.bin and m8_ref_out.bin for C++ verification.")
