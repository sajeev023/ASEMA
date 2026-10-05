@echo off
setlocal

rem Toolchain: clang++, ninja and cmake must be on PATH.
rem If they are not, set CLANG_BIN and/or NINJA_BIN to their directories before running this script.
if defined CLANG_BIN set PATH=%CLANG_BIN%;%PATH%
if defined NINJA_BIN set PATH=%NINJA_BIN%;%PATH%

echo [ASEMA BUILD] Configuring CMake with Clang++ and Ninja...
cmake -B build -G Ninja ^
  -DCMAKE_CXX_COMPILER=clang++ ^
  -DCMAKE_C_COMPILER=clang ^
  -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% NEQ 0 (
    echo [ASEMA BUILD] CMake configuration failed!
    exit /b %ERRORLEVEL%
)

echo [ASEMA BUILD] Building targets with Ninja...
cmake --build build --config Release

if %ERRORLEVEL% NEQ 0 (
    echo [ASEMA BUILD] Compilation failed!
    exit /b %ERRORLEVEL%
)

echo [ASEMA BUILD] Build succeeded!
