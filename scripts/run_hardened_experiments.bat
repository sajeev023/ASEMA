@echo off
setlocal

set ROOT=%~dp0..
set BENCH=%ROOT%\build\asema-bench.exe
set T1=%ROOT%\examples\synthetic_moe\model_synthetic
set T2=%ROOT%\examples\synthetic_moe\model_tier2
set T3=%ROOT%\examples\synthetic_moe\model_tier3

set TIER1_OUT=%ROOT%\reports\runs\hardened_tier1
set TIER2_OUT=%ROOT%\reports\runs\hardened_tier2
set TIER3_OUT=%ROOT%\reports\runs\hardened_tier3

echo === Hardened M7 batch run ===

REM Tier 1 baseline + cache sweep with N=10
echo [T1 baseline]
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache 0 --lookahead 0 --workers 4 --seed 42 --output "%TIER1_OUT%\baseline" --quiet
echo [T1 cache-sweep K=2]
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache-sweep --lookahead 2 --workers 4 --seed 42 --output "%TIER1_OUT%\cache_sweep_K2" --quiet

REM Tier 2 baseline + cache sweep with N=10
echo [T2 baseline]
"%BENCH%" --model "%T2%" --tokens 32 --warmup 3 --iterations 10 --cache 0 --lookahead 0 --workers 4 --seed 42 --output "%TIER2_OUT%\baseline" --quiet
echo [T2 cache-sweep K=2]
"%BENCH%" --model "%T2%" --tokens 32 --warmup 3 --iterations 10 --cache-sweep --lookahead 2 --workers 4 --seed 42 --output "%TIER2_OUT%\cache_sweep_K2" --quiet

REM Tier 3 baseline + cache sweep with N=5 (Tier 3 is much slower)
echo [T3 baseline]
"%BENCH%" --model "%T3%" --tokens 16 --warmup 2 --iterations 5 --cache 0 --lookahead 0 --workers 4 --seed 42 --output "%TIER3_OUT%\baseline" --quiet
echo [T3 cache-sweep K=2]
"%BENCH%" --model "%T3%" --tokens 16 --warmup 2 --iterations 5 --cache-sweep --lookahead 2 --workers 4 --seed 42 --output "%TIER3_OUT%\cache_sweep_K2" --quiet

REM Locality sweep on Tier 1 (cache 8MB, K=2)
echo [T1 locality sweep]
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache 8388608 --lookahead 2 --locality-high --workers 4 --seed 42 --output "%TIER1_OUT%\locality_high" --quiet
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache 8388608 --lookahead 2 --locality-medium --workers 4 --seed 42 --output "%TIER1_OUT%\locality_medium" --quiet
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache 8388608 --lookahead 2 --locality-low --workers 4 --seed 42 --output "%TIER1_OUT%\locality_low" --quiet
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache 8388608 --lookahead 2 --locality-random --workers 4 --seed 42 --output "%TIER1_OUT%\locality_random" --quiet
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache 8388608 --lookahead 2 --locality-adversarial --workers 4 --seed 42 --output "%TIER1_OUT%\locality_adversarial" --quiet

REM Prefetch sweep with NEW metrics (cache 8MB)
echo [T1 prefetch sweep]
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache 8388608 --lookahead-sweep --workers 4 --seed 42 --output "%TIER1_OUT%\prefetch_sweep" --quiet

REM Worker sweep (Tier 1, cache 8MB, K=2)
echo [T1 worker sweep]
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache 8388608 --lookahead 2 --workers-sweep --seed 42 --output "%TIER1_OUT%\worker_sweep" --quiet

REM Seed sweep (Tier 1, cache 8MB, K=2)
echo [T1 seed sweep]
"%BENCH%" --model "%T1%" --tokens 32 --warmup 3 --iterations 10 --cache 8388608 --lookahead 2 --workers 4 --seed-sweep --output "%TIER1_OUT%\seed_sweep" --quiet

echo === Hardened M7 batch DONE ===
