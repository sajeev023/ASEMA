import urllib.request
import struct
import json

url = 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/model-00003-of-00048.safetensors'
req = urllib.request.Request(url, headers={'Range': 'bytes=0-1048575', 'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = resp.read()

hlen = struct.unpack('<Q', data[:8])[0]
print(f'model-00003-of-00048.safetensors header_len = {hlen}')
header = json.loads(data[8:8+hlen].decode('utf-8'))

tensor_names = [k for k in header.keys() if k != '__metadata__']
print(f'Total tensors in Layer 0 shard: {len(tensor_names)}')

# Categorize tensors
attn_tensors = [k for k in tensor_names if 'self_attn' in k or 'attn' in k]
norm_tensors = [k for k in tensor_names if 'norm' in k]
router_tensors = [k for k in tensor_names if 'gate' in k or 'router' in k]
shared_tensors = [k for k in tensor_names if 'shared' in k]
expert_tensors = [k for k in tensor_names if 'experts' in k]

print(f'  Attention tensors: {len(attn_tensors)}')
print(f'  Norm tensors: {len(norm_tensors)}')
print(f'  Router tensors: {len(router_tensors)}')
print(f'  Shared expert tensors: {len(shared_tensors)}')
print(f'  Routed expert tensors: {len(expert_tensors)}')

print('\nSample tensor definitions:')
for k in (norm_tensors + router_tensors[:2] + shared_tensors[:3] + expert_tensors[:3]):
    m = header[k]
    print(f"  {k}: dtype={m['dtype']}, shape={m['shape']}, offsets={m['data_offsets']}")
