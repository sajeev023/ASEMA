import os
import struct
import json

for name in ['model-00002-of-00048.safetensors', 'model-00043-of-00048.safetensors']:
    path = os.path.join(os.environ.get("ASEMA_SHARDS_PRIMARY", "models/DeepSeek-V4.1-Flash/shards"), name)
    with open(path, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen).decode('utf-8'))
        print(f'{name}: header_len={hlen}')
        for k in header:
            if k != '__metadata__':
                meta = header[k]
                print(f"  {k}: dtype={meta['dtype']}, shape={meta['shape']}, offsets={meta['data_offsets']}")
