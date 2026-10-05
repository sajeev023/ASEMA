# ASEMA v0.1 — Hardware & Toolchain Environment Report

**Date**: 2026-09-25  
**Platform**: Windows 11 Pro 64-bit (Build 10.0.26200)

---

## 1. Hardware Specification

| Component | Specification | Notes |
| :--- | :--- | :--- |
| **CPU** | AMD Ryzen 7 5700X 8-Core Processor | 8 Physical Cores, 16 Logical Processors, AVX2 support |
| **System RAM** | 31.9 GB Total Visible Physical RAM (~21.1 GB Available) | DDR4 |
| **GPU** | AMD Radeon RX 580 2048SP (Polaris 20) | 8 GB Physical VRAM reported |
| **GPU Driver** | AMD Proprietary Driver 31.0.21925.1001 / Vulkan Driver 2.0.279 | Supports Vulkan 1.3.260 |
| **Storage (C:)** | NVMe / Fast SSD | 475.9 GB Total, 77.1 GB Free (OS & Scratch) |
| **Storage (D:)** | NVMe / Fast SSD | 476.9 GB Total, 476.8 GB Free (Dedicated Model Tier) |
| **Storage (E:)** | NVMe / Fast SSD | 237.8 GB Total, 237.7 GB Free |

---

## 2. Software & Toolchains Detected

| Tool | Status / Path | Version / Capabilities |
| :--- | :--- | :--- |
| **Python** | Installed (`python.exe` on PATH) | 3.11.9 64-bit |
| **CMake** | Installed (`C:\Program Files\CMake\bin\cmake.EXE`) | 3.x Native Windows |
| **Git** | Installed (`C:\Program Files\Git\cmd\git.EXE`) | Available |
| **Vulkan Runtime**| Installed (`C:\WINDOWS\system32\vulkaninfo.EXE`) | Vulkan 1.3.301 Instance, AMD RX 580 2048SP |
| **C/C++ Compiler** | LLVM-MinGW (Clang/Clang++ 19+, UCRT, LLD) | Standalone x86_64 toolchain |
| **PyTorch** | Installed | 2.14.0+cpu with Transformers 5.16, Safetensors 0.8 |
| **Package Mgr** | Windows Package Manager (`winget.exe`) | Operational |

---

## 3. Storage Hierarchy Mapping for ASEMA

- **Tier 0 (Persistent Storage)**: Fast SSD on `D:\` (476 GB unconstrained free space) for persistent `.asema` expert containers (`ASEMA-SSF`).
- **Tier 1 (Warm RAM Cache)**: Up to 16 GB configurable RAM budget out of 32 GB total system memory.
- **Tier 2 (Hot VRAM Cache)**: RX 580 8 GB VRAM. Optional Vulkan compute / memory upload cache.
- **Tier 3 (Compute)**:
  - Primary CPU execution: Ryzen 7 5700X with multi-threaded vectorized kernels.
  - Optional GPU acceleration: Vulkan compute backend.
