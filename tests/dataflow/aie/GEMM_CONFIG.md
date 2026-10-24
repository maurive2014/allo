# Running a specific int16 GEMM configuration

`benchmark_gemm_config.py` is the same runner in `~/allo` and `~/allo-my-fork`.
It uses the compiler and library in the active environment, prints `allo.__file__`,
and does not replace or patch the library GEMM. Verify that printed path points
at the checkout you intend to test.

The runner accepts **tile dimensions**:

- `--tmc`: output tile rows, TmC.
- `--tk`: reduction tile length, Tk.
- `--tn`: output tile columns, Tn.
- `--rho`: 1 for symmetric buffering, 2 or 4 for ATB; TmA = TmC/rho.

The older `test_temporal_atb_gemm.py` instead accepts **partition counts**:
`pm=M/TmC`, `pn=N/Tn`, `pk=K/Tk`. Its executable entry point hardcodes rho=2.
The number of logical partitions is not the physical core count: the mapping
primitives bundle/chain the work onto 16 physical cores. This runner requires
M divisible by 4*TmC, N by 4*Tn, and K by Tk.

## Main checkout: existing symmetric library GEMM

```bash
conda activate /opt/anaconda3/envs/allo-base
cd ~/allo
python3 tests/dataflow/aie/benchmark_gemm_config.py \
    --frontend library \
    --m 1024 --k 1024 --n 1024 \
    --tmc 32 --tk 64 --tn 128 --rho 1 \
    --project symmetric_main_1024.prj
```

This reproduces the **tile dimensions** of `sym_temporal_32_64_128` using main's
existing library frontend. Main has no equivalent of the fork's temporal
frontend. Call this configuration `sym_32_64_128`, not `sym_temporal_32_64_128`.
No temporal-sharding compiler changes are copied into main.

## Fork: enter the user's AIE container

```bash
cd ~/allo-my-fork
docker run --rm -it \
    --device /dev/accel/accel0:/dev/accel/accel0 \
    --ulimit memlock=-1 \
    -v "$(pwd):/ryzers/allo-my-fork" \
    -w /ryzers/allo-my-fork \
    shihanfang/allo-ci:aie-v1.0 bash
source activate allo
python3 -m pip install --no-build-isolation -e .
```

Then run one configuration at a time inside the container.

The same symmetric library GEMM as main, now compiled by the fork:

```bash
python3 tests/dataflow/aie/benchmark_gemm_config.py \
    --frontend library \
    --m 1024 --k 1024 --n 1024 \
    --tmc 32 --tk 64 --tn 128 --rho 1 \
    --project symmetric_fork_1024.prj
```

The exact `sym_temporal_32_64_128` frontend/tile configuration:

```bash
python3 tests/dataflow/aie/benchmark_gemm_config.py \
    --frontend temporal \
    --m 1024 --k 1024 --n 1024 \
    --tmc 32 --tk 64 --tn 128 --rho 1 \
    --project symmetric_temporal_1024.prj
```

The requested `(M,K,N)=(1024,2048,2048)`, `atb_128_64_64_r4` configuration:

```bash
python3 tests/dataflow/aie/benchmark_gemm_config.py \
    --frontend temporal \
    --m 1024 --k 2048 --n 2048 \
    --tmc 128 --tk 64 --tn 64 --rho 4 \
    --project atb_1024_2048_2048_r4.prj
```

That last configuration has TmA=32, TmC=128, Tk=64, Tn=64, and partition counts
pm=8, pn=32, pk=32. Identifiers in the study encode `TmC_Tk_Tn`, not partition
counts. Dimensions in study tables are `(M,K,N)`; the CLI names each explicitly.

## Timing and verification

Defaults: int16 A/B/C, DMA port spreading enabled, 16 cores, three profiling
invocations, each with 200 warmups and 1,000 measured launches. The same seeded
inputs in [-2,2] are used across invocations and configurations of a workload.
Each invocation's entire final output is checked exactly against a bounded
integer reference computed using float64 BLAS and narrowed to int16.

Use the median of the three printed `Avg NPU execution time` values; these are
microseconds, including host submission through NPU completion and programmed
DMA, excluding compilation and reference computation. TOPS is
`2*M*K*N/(latency_us*1e6)`. Avoid comparing Python process wall-clock time.

Optional flags:

- `--dry-run`: print dimensions, partition counts and settings, without building.
- `--build-only`: compile without device execution.
- `--runs 1`: one profiling invocation for a quicker check.
- `--warmup 200 --num-iters 1000 --runs 3`: explicitly request the study protocol.
- `--seed 20261017`: default reproducible seed.

Each run builds its project. Use distinct project directories to preserve
artifacts. Configuration and successful verification metadata are saved there.
Compiler/timing output can also be retained with `2>&1 | tee run.log`.

Compare `library` against `library` across branches to keep the source model
constant. Compare `library` against `temporal --rho 1` within the fork to isolate
the frontend/lowering difference. Your host-main and container-fork workflows
also differ in compiler environment, so a latency difference alone does not
prove it was caused by the branch. Run hardware measurements serially.

Some tiles will exceed local memory or instruction-memory capacity; changing
CLI dimensions does not guarantee feasibility. In particular main's statically
chained long reductions can overflow program memory where the fork's temporal
frontend still fits.
