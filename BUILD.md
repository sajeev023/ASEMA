# Build

## Prerequisites

- Windows 10/11 x64 (the engine uses Win32 memory mapping and Direct3D 11).
- Compiler: Clang/LLVM with C++20. Tested with LLVM-MinGW (UCRT, x86_64). `CMakeLists.txt` passes
  GCC/Clang-style flags (`-O3 -mavx2 -mfma`); MSVC flags exist but are untested.
- CMake >= 3.20 and Ninja.
- Windows SDK headers/libs for Direct3D 11 / DXGI / D3DCompiler (linked as `d3d11 d3dcompiler dxgi`).
- CPU with AVX2 and FMA.

## Build

```
scripts/build.bat
```

The script expects `clang++`, `ninja` and `cmake` on `PATH`; if they are not, set `CLANG_BIN` and/or
`NINJA_BIN` to their directories first. Equivalent manual commands:

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

## Test

```
ctest --test-dir build -LE requires-data     # no model files needed (26 tests)
ctest --test-dir build -L requires-data      # needs the checkpoint / synthetic model (17 tests)
```

Tests run with the source directory as their working directory, so `asema.config` there is found.
The `requires-data` tests do not skip gracefully: without the checkpoint (configured through
`asema.config` or `ASEMA_*` variables, see MODEL_SETUP.md) they fail, abort or time out. Some run
real inference and take minutes and several GB of RAM. `test_m8_fp8_gemv` and
`test_m8_tokenizer_roundtrip` use real files when present and otherwise run their synthetic parts.

## Targets

- `asema`: the command-line tool (`build/asema.exe`): `doctor`, `verify-model`, `generate`,
  `chat`, `profile`, `bench`.
- `asema_core`: static library with the runtime, loader, GPU kernels and tokenizer.
- `test_*`, `asema-*`: test and benchmark executables in `build/`.
