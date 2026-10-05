cmake_minimum_required(VERSION 3.20)
project(asema VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization and linking flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-O3 -mavx2 -mfma -Wall -Wextra -Wno-unused-parameter -mconsole)
    add_link_options(-static -mconsole -Wl,--subsystem,console)
elseif (MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
endif()

include_directories(include)

# CTest runs tests from the build directory by default. ASEMA resolves model locations
# (asema.config, examples/...) relative to the source tree, so register every test with the
# source directory as its working directory.
macro(add_test)
  _add_test(${ARGN})
  set(_asema_add_test_args ${ARGN}) # macros substitute ARGN textually; copy it to a real variable
  list(FIND _asema_add_test_args NAME _asema_name_idx)
  if(NOT _asema_name_idx EQUAL -1)
    math(EXPR _asema_name_idx "${_asema_name_idx}+1")
    list(GET _asema_add_test_args ${_asema_name_idx} _asema_test_name)
    set_tests_properties(${_asema_test_name} PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  endif()
endmacro()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# AGENT #2 ADDITIONS - Milestones 3, 4, 5
# (Do not modify this block from another agent)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

set(ASEMA_AGENT2_SOURCES
    src/cache/cache.cpp
    src/queue/request_queue.cpp
    src/prefetch/prefetch.cpp
    src/telemetry/telemetry.cpp
    src/runtime/runtime.cpp
    src/runtime/loader_adapter.cpp
)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# AGENT #3 ADDITIONS - Milestones 6, 7
# (Do not modify this block from another agent)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

set(ASEMA_AGENT3_SOURCES
    src/cli/cli.cpp
    src/experiments/experiments.cpp
    src/experiments/synthetic_gen.cpp
)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# AGENT #5 ADDITIONS - Milestone 8 (Real-Model Integration)
# (Do not modify this block from another agent)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

set(ASEMA_AGENT5_SOURCES
    src/m8/m8_model_adapter.cpp
    src/m8/m8_router.cpp
    src/m8/m8_expert_kernel.cpp
    src/m8/m8_gpu_expert_kernel.cpp
    src/m8/m8_gpu_mla_kernel.cpp
    src/m8/m8_multi_volume.cpp
    src/m8/m8_mla_attention.cpp
    src/m8/m8_byte_loader.cpp
    src/m8/m8_transformer_layer.cpp
    src/m8/m8_model_runner.cpp
    src/m8/m8_memory_tracker.cpp
    src/m8/m8_storage_forensics.cpp
    src/m8/m8_expert_cache.cpp
    src/m8/m8_routing_prefetcher.cpp
    src/m8/m8_storage_scheduler.cpp
    src/m8/m8_iocp_storage.cpp
    src/m8/m8_gpu_memory_arena.cpp
    src/m8/m8_gpu_command_scheduler.cpp
    src/m8/m8_hybrid_scheduler.cpp
    src/m8/m8_model_manifest.cpp
    src/m8/m8_storage_planner.cpp
    src/m8/m8_model_verifier.cpp
    src/m8/m8_download_engine.cpp
    src/m8/m8_model_doctor.cpp
    src/m8/m8_fault_tolerance.cpp
    src/m8/m8_generation_api.cpp
    src/m8/m8_safetensors_index.cpp
    src/m8/m8_reference_engine.cpp
    src/m8/m8_resource_governor.cpp
)

# Core ASEMA Runtime Library (all sources aggregated BEFORE add_library)
set(ASEMA_SOURCES
    src/storage/storage_ssf.cpp
    src/loader/async_loader.cpp
    ${ASEMA_AGENT2_SOURCES}
    ${ASEMA_AGENT3_SOURCES}
    ${ASEMA_AGENT5_SOURCES}
)

add_library(asema_core STATIC ${ASEMA_SOURCES})
target_link_libraries(asema_core PUBLIC d3d11 d3dcompiler dxgi)

# Tools
add_executable(asema-inspect tools/asema_inspect.cpp)
target_link_libraries(asema-inspect PRIVATE asema_core)

add_executable(asema-async-bench tools/asema_async_bench.cpp)
target_link_libraries(asema-async-bench PRIVATE asema_core)

# Tests
enable_testing()

add_executable(test_storage tests/test_storage.cpp)
target_link_libraries(test_storage PRIVATE asema_core)
add_test(NAME test_storage COMMAND test_storage)

add_executable(test_async_loader tests/test_async_loader.cpp)
target_link_libraries(test_async_loader PRIVATE asema_core)
add_test(NAME test_async_loader COMMAND test_async_loader)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# AGENT #2 ADDITIONS - Milestones 3, 4, 5 (tests)
# (Do not modify this block from another agent)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

add_executable(test_cache tests/test_cache.cpp)
target_link_libraries(test_cache PRIVATE asema_core)
add_test(NAME test_cache COMMAND test_cache)

add_executable(test_queue tests/test_queue.cpp)
target_link_libraries(test_queue PRIVATE asema_core)
add_test(NAME test_queue COMMAND test_queue)

add_executable(test_prefetch tests/test_prefetch.cpp)
target_link_libraries(test_prefetch PRIVATE asema_core)
add_test(NAME test_prefetch COMMAND test_prefetch)

add_executable(test_telemetry tests/test_telemetry.cpp)
target_link_libraries(test_telemetry PRIVATE asema_core)
add_test(NAME test_telemetry COMMAND test_telemetry)

add_executable(test_runtime tests/test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE asema_core)
add_test(NAME test_runtime COMMAND test_runtime)

add_executable(asema-full-bench tools/asema_full_bench.cpp)
target_link_libraries(asema-full-bench PRIVATE asema_core)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# AGENT #3 ADDITIONS - Milestones 6, 7 (CLI + tests)
# (Do not modify this block from another agent)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

add_executable(asema-pack tools/asema_pack.cpp)
target_link_libraries(asema-pack PRIVATE asema_core)

add_executable(asema-run tools/asema_run.cpp)
target_link_libraries(asema-run PRIVATE asema_core)

add_executable(asema-bench tools/asema_bench.cpp)
target_link_libraries(asema-bench PRIVATE asema_core)

add_executable(test_cli tests/test_cli.cpp)
target_link_libraries(test_cli PRIVATE asema_core)
add_test(NAME test_cli COMMAND test_cli)

add_executable(test_synthetic_gen tests/test_synthetic_gen.cpp)
target_link_libraries(test_synthetic_gen PRIVATE asema_core)
add_test(NAME test_synthetic_gen COMMAND test_synthetic_gen)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# AGENT #5 ADDITIONS - Milestone 8 (Real-Model Integration)
# (Do not modify this block from another agent)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# M8 CLI tools
add_executable(m8-inspect tools/m8/m8_inspect.cpp)
target_link_libraries(m8-inspect PRIVATE asema_core)

add_executable(test_m8_run tools/m8/m8_run.cpp)
target_link_libraries(test_m8_run PRIVATE asema_core)
add_test(NAME test_m8_run COMMAND test_m8_run)

# M8 tests
add_executable(test_m8_adapter tests/test_m8_adapter.cpp)
target_link_libraries(test_m8_adapter PRIVATE asema_core)
add_test(NAME test_m8_adapter COMMAND test_m8_adapter)

add_executable(test_m8_equivalence tests/test_m8_equivalence.cpp)
target_link_libraries(test_m8_equivalence PRIVATE asema_core)
add_test(NAME test_m8_equivalence COMMAND test_m8_equivalence)

add_executable(test_m8_full_execution tests/test_m8_full_execution.cpp)
target_link_libraries(test_m8_full_execution PRIVATE asema_core)
add_test(NAME test_m8_full_execution COMMAND test_m8_full_execution)

add_executable(test_m8_gpu_acceleration tests/test_m8_gpu_acceleration.cpp)
target_link_libraries(test_m8_gpu_acceleration PRIVATE asema_core)
add_test(NAME test_m8_gpu_acceleration COMMAND test_m8_gpu_acceleration)

add_executable(test_m8_benchmark tests/test_m8_benchmark.cpp)
target_link_libraries(test_m8_benchmark PRIVATE asema_core)
add_test(NAME test_m8_benchmark COMMAND test_m8_benchmark)

add_executable(test_m8_gpu_mla tests/test_m8_gpu_mla.cpp)
target_link_libraries(test_m8_gpu_mla PRIVATE asema_core)
add_test(NAME test_m8_gpu_mla COMMAND test_m8_gpu_mla)

add_executable(test_m8_mla_benchmark tests/test_m8_mla_benchmark.cpp)
target_link_libraries(test_m8_mla_benchmark PRIVATE asema_core)
add_test(NAME test_m8_mla_benchmark COMMAND test_m8_mla_benchmark)

add_executable(test_m8_memory_forensics tests/test_m8_memory_forensics.cpp)
target_link_libraries(test_m8_memory_forensics PRIVATE asema_core psapi)
add_test(NAME test_m8_memory_forensics COMMAND test_m8_memory_forensics)

add_executable(test_m8_storage_forensics tests/test_m8_storage_forensics.cpp)
target_link_libraries(test_m8_storage_forensics PRIVATE asema_core)
add_test(NAME test_m8_storage_forensics COMMAND test_m8_storage_forensics)

add_executable(test_m8_expert_cache tests/test_m8_expert_cache.cpp)
target_link_libraries(test_m8_expert_cache PRIVATE asema_core)
add_test(NAME test_m8_expert_cache COMMAND test_m8_expert_cache)

add_executable(test_m8_routing_prefetch tests/test_m8_routing_prefetch.cpp)
target_link_libraries(test_m8_routing_prefetch PRIVATE asema_core)
add_test(NAME test_m8_routing_prefetch COMMAND test_m8_routing_prefetch)

add_executable(test_m8_storage_scheduler tests/test_m8_storage_scheduler.cpp)
target_link_libraries(test_m8_storage_scheduler PRIVATE asema_core)
add_test(NAME test_m8_storage_scheduler COMMAND test_m8_storage_scheduler)

add_executable(test_m8_iocp_benchmark tests/test_m8_iocp_benchmark.cpp)
target_link_libraries(test_m8_iocp_benchmark PRIVATE asema_core)
add_test(NAME test_m8_iocp_benchmark COMMAND test_m8_iocp_benchmark)

add_executable(test_m8_gpu_arena tests/test_m8_gpu_arena.cpp)
target_link_libraries(test_m8_gpu_arena PRIVATE asema_core d3d11 dxgi)
add_test(NAME test_m8_gpu_arena COMMAND test_m8_gpu_arena)

add_executable(test_m8_gpu_command_scheduler tests/test_m8_gpu_command_scheduler.cpp)
target_link_libraries(test_m8_gpu_command_scheduler PRIVATE asema_core d3d11 dxgi)
add_test(NAME test_m8_gpu_command_scheduler COMMAND test_m8_gpu_command_scheduler)

add_executable(test_m8_expert_optimization tests/test_m8_expert_optimization.cpp)
target_link_libraries(test_m8_expert_optimization PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_expert_optimization COMMAND test_m8_expert_optimization)

add_executable(test_m8_tokenizer_roundtrip tests/test_m8_tokenizer_roundtrip.cpp)
target_link_libraries(test_m8_tokenizer_roundtrip PRIVATE asema_core)
add_test(NAME test_m8_tokenizer_roundtrip COMMAND test_m8_tokenizer_roundtrip)

add_executable(test_m8_fp8_gemv tests/test_m8_fp8_gemv.cpp)
target_link_libraries(test_m8_fp8_gemv PRIVATE asema_core)
add_test(NAME test_m8_fp8_gemv COMMAND test_m8_fp8_gemv)

add_executable(test_m8_dequantization tests/test_m8_dequantization.cpp)
target_link_libraries(test_m8_dequantization PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_dequantization COMMAND test_m8_dequantization)

add_executable(test_m8_swiglu_fusion tests/test_m8_swiglu_fusion.cpp)
target_link_libraries(test_m8_swiglu_fusion PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_swiglu_fusion COMMAND test_m8_swiglu_fusion)

add_executable(test_m8_mla_optimization tests/test_m8_mla_optimization.cpp)
target_link_libraries(test_m8_mla_optimization PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_mla_optimization COMMAND test_m8_mla_optimization)

add_executable(test_m8_residual_fusion tests/test_m8_residual_fusion.cpp)
target_link_libraries(test_m8_residual_fusion PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_residual_fusion COMMAND test_m8_residual_fusion)

add_executable(test_m8_gpu_rmsnorm tests/test_m8_gpu_rmsnorm.cpp)
target_link_libraries(test_m8_gpu_rmsnorm PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_gpu_rmsnorm COMMAND test_m8_gpu_rmsnorm)

add_executable(test_m8_complete_pipeline tests/test_m8_complete_pipeline.cpp)
target_link_libraries(test_m8_complete_pipeline PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_complete_pipeline COMMAND test_m8_complete_pipeline)

add_executable(test_m8_router_optimization tests/test_m8_router_optimization.cpp)
target_link_libraries(test_m8_router_optimization PRIVATE asema_core)
add_test(NAME test_m8_router_optimization COMMAND test_m8_router_optimization)

add_executable(test_m8_hybrid_scheduler tests/test_m8_hybrid_scheduler.cpp)
target_link_libraries(test_m8_hybrid_scheduler PRIVATE asema_core)
add_test(NAME test_m8_hybrid_scheduler COMMAND test_m8_hybrid_scheduler)

add_executable(test_m8_storage_pipeline_opt tests/test_m8_storage_pipeline_opt.cpp)
target_link_libraries(test_m8_storage_pipeline_opt PRIVATE asema_core)
add_test(NAME test_m8_storage_pipeline_opt COMMAND test_m8_storage_pipeline_opt)

add_executable(test_m8_generation_engine tests/test_m8_generation_engine.cpp)
target_link_libraries(test_m8_generation_engine PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_generation_engine COMMAND test_m8_generation_engine)

add_executable(test_m8_model_installation tests/test_m8_model_installation.cpp)
target_link_libraries(test_m8_model_installation PRIVATE asema_core)
add_test(NAME test_m8_model_installation COMMAND test_m8_model_installation)

add_executable(test_m8_real_checkpoint_pipeline tests/test_m8_real_checkpoint_pipeline.cpp)
target_link_libraries(test_m8_real_checkpoint_pipeline PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_real_checkpoint_pipeline COMMAND test_m8_real_checkpoint_pipeline)

add_executable(asema tools/asema_cli.cpp)
target_link_libraries(asema PRIVATE asema_core d3d11 d3dcompiler dxgi)

# CLI behaviour tests: these run the real asema binary and need no model data.
add_test(NAME cli_version COMMAND asema --version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "^asema [0-9]+\\.[0-9]+\\.[0-9]+")
add_test(NAME cli_help COMMAND asema --help)
set_tests_properties(cli_help PROPERTIES PASS_REGULAR_EXPRESSION "Made by Ajay.*Usage: asema")
add_test(NAME cli_unknown_command COMMAND asema definitely-not-a-command)
set_tests_properties(cli_unknown_command PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_chat_rejects_bad_option COMMAND asema chat --no-such-option)
set_tests_properties(cli_chat_rejects_bad_option PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_models_missing_checkpoint COMMAND ${CMAKE_COMMAND} -E env ASEMA_SHARDS_PRIMARY=nonexistent_dir ASEMA_SHARDS_SECONDARY= ASEMA_HF_DIR=nonexistent_dir ASEMA_MODEL_ROOT=nonexistent_dir $<TARGET_FILE:asema> models)
set_tests_properties(cli_models_missing_checkpoint PROPERTIES PASS_REGULAR_EXPRESSION "MODEL NOT FOUND")
add_test(NAME cli_doctor_reports_problem COMMAND ${CMAKE_COMMAND} -E env ASEMA_SHARDS_PRIMARY=nonexistent_dir ASEMA_SHARDS_SECONDARY= ASEMA_HF_DIR=nonexistent_dir ASEMA_MODEL_ROOT=nonexistent_dir $<TARGET_FILE:asema> doctor)
set_tests_properties(cli_doctor_reports_problem PROPERTIES PASS_REGULAR_EXPRESSION "PROBLEM.*failed; fixes are listed above" WILL_FAIL FALSE)
# With the real checkpoint configured this must report READY (needs the model, so it is labelled).
add_test(NAME cli_models_ready COMMAND asema models)
set_tests_properties(cli_models_ready PROPERTIES PASS_REGULAR_EXPRESSION "READY +48/48")

add_executable(test_m8_production_suite tests/test_m8_production_suite.cpp)
target_link_libraries(test_m8_production_suite PRIVATE asema_core d3d11 d3dcompiler dxgi)
add_test(NAME test_m8_production_suite COMMAND test_m8_production_suite)

add_executable(test_m8_tokenizer_bpe tests/test_m8_tokenizer_bpe.cpp)
target_link_libraries(test_m8_tokenizer_bpe PRIVATE asema_core)
add_test(NAME test_m8_tokenizer_bpe COMMAND test_m8_tokenizer_bpe)

add_executable(test_m8_reference_engine tests/test_m8_reference_engine.cpp)
target_link_libraries(test_m8_reference_engine PRIVATE asema_core)
add_test(NAME test_m8_reference_engine COMMAND test_m8_reference_engine)

add_executable(test_m8_trace_diagnostics tests/test_m8_trace_diagnostics.cpp)
target_link_libraries(test_m8_trace_diagnostics PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_inspect_tokens tests/test_inspect_tokens.cpp)
target_link_libraries(test_inspect_tokens PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_layer_evolution tests/test_layer_evolution.cpp)
target_link_libraries(test_layer_evolution PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_attn_norm tests/test_attn_norm.cpp)
target_link_libraries(test_attn_norm PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_scale_sweep tests/test_scale_sweep.cpp)
target_link_libraries(test_scale_sweep PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_real_40layer_generation tests/test_real_40layer_generation.cpp)
target_link_libraries(test_real_40layer_generation PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_layer0_moe tests/test_layer0_moe.cpp)
target_link_libraries(test_layer0_moe PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_instruct_generation tests/test_instruct_generation.cpp)
target_link_libraries(test_instruct_generation PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_trace_generation tests/test_trace_generation.cpp)
target_link_libraries(test_trace_generation PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_evolution_trace tests/test_evolution_trace.cpp)
target_link_libraries(test_evolution_trace PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_greedy_vs_rep tests/test_greedy_vs_rep.cpp)
target_link_libraries(test_greedy_vs_rep PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_completion tests/test_completion.cpp)
target_link_libraries(test_completion PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_attn_context tests/test_attn_context.cpp)
target_link_libraries(test_attn_context PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_residual_scale tests/test_residual_scale.cpp)
target_link_libraries(test_residual_scale PRIVATE asema_core d3d11 d3dcompiler dxgi)

add_executable(test_acceptance_prompts tests/test_acceptance_prompts.cpp)
target_link_libraries(test_acceptance_prompts PRIVATE asema_core d3d11 d3dcompiler dxgi)

# Tests that need the model checkpoint (see asema.config.example) and/or the locally generated
# synthetic model. A fresh checkout without those files should run:  ctest -LE requires-data
set_tests_properties(
    test_storage test_async_loader test_runtime test_m8_run test_m8_adapter test_m8_equivalence
    test_m8_full_execution test_m8_gpu_acceleration test_m8_benchmark test_m8_gpu_mla
    test_m8_mla_benchmark test_m8_memory_forensics test_m8_storage_forensics
    test_m8_generation_engine test_m8_real_checkpoint_pipeline test_m8_production_suite
    test_m8_tokenizer_bpe cli_models_ready
    PROPERTIES LABELS "requires-data")



















