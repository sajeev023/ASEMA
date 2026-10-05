# ASEMA Installation & Toolchain Guide

## Prerequisites

- **OS:** Windows 10/11 64-bit (Pro/Home)
- **Compiler:** Clang++ (LLVM-MinGW UCRT x86_64 or MSVC 2022) with C++20 support
- **Build System:** CMake >= 3.20, Ninja >= 1.10
- **DirectX:** Windows 11 SDK (Direct3D 11, D3DCompiler, DXGI runtime libraries)
- **Hardware:**
  - CPU: AVX2 and FMA supported x86_64 processor (AMD Ryzen or Intel Core)
  - GPU: Direct3D 11 Compute Shader 5.0 capable GPU (AMD Radeon RX 400/500/Vega/Navi, NVIDIA GTX 900+)
  - Storage: about 510 GB for the checkpoint on one or two drives; NVMe strongly recommended (see PERFORMANCE.md)

---

## Build Steps

```powershell
# Clone or navigate to the ASEMA repository
cd <path-to-your-asema-checkout>

# Generate build configuration
cmake -B build -G Ninja

# Compile core runtime, tests, and unified CLI
ninja -C build asema

# Verify compilation
.\build\asema.exe doctor
```

---

## Automated Deployment Verification

```powershell
# Run the complete automated test suite
ninja -C build test_m8_production_suite
.\build\test_m8_production_suite.exe
```
