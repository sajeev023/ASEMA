import numpy as np

with open("gate_weight.bin", "rb") as f:
    buf = f.read()

# BF16 is the high 16 bits of FP32
u16 = np.frombuffer(buf, dtype=np.uint16)
u32 = (u16.astype(np.uint32) << 16)
f32 = u32.view(np.float32).reshape(384, 5120)

print("Router gate weight shape:", f32.shape)
print("Mean:", float(np.mean(f32)))
print("Std:", float(np.std(f32)))
print("Min / Max:", float(np.min(f32)), "/", float(np.max(f32)))
print("First 5 weights for expert 0:", [round(float(x), 6) for x in f32[0, :5]])
