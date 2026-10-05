# Security

## Status

ASEMA is experimental research software and has had no formal security review. It is a local
command-line tool, not a network service; it opens no listening sockets.

## Trust model

- Treat the checkpoint directories and `asema.config` as **trusted input**. The engine reads
  safetensors headers and memory-maps shard files. Tensor offsets and sizes from a header are
  bounds-checked against the mapped file before a pointer is handed out, and the header size is
  capped, but the parser has not been fuzz-tested. Do not point ASEMA at checkpoint files from
  untrusted sources.
- No path sanitization or sandboxing is implemented for configured paths or prompts.
- Prompts are tokenized and processed locally; nothing is sent over the network by the inference
  path. The `download` helpers and scripts under `scripts/` contact Hugging Face to fetch model
  files and should only be run deliberately.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Report it privately through
GitHub: **Security tab → "Report a vulnerability"** on https://github.com/sajeev023/ASEMA
(<https://github.com/sajeev023/ASEMA/security/advisories/new>). Include the affected version, a
description, and reproduction steps. This is a one-person research project, so there is no
guaranteed response time.

## Hygiene for contributors

Never commit credentials, access tokens, model weights or machine-specific paths. Before
publishing, run the checks listed in docs/OPEN_SOURCE_READINESS.md.
