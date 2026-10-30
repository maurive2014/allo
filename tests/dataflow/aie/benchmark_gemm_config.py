# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one int16 GEMM configuration on 16 AIE cores, with exact verification.

The library frontend works on main and the temporal-sharding-atb fork.
The temporal frontend requires that fork; rho=1 is symmetric buffering.
Tile arguments are dimensions, not partition counts. Use a new process for
each configuration because Allo captures Python frontend globals.
"""

import argparse
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontend", choices=("library", "temporal"), default="library"
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--tmc", type=int, default=32)
    parser.add_argument("--tk", type=int, default=64)
    parser.add_argument("--tn", type=int, default=128)
    parser.add_argument("--rho", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--c-depth",
        "--output-fifo-depth",
        "--output_fifo_depth",
        dest="output_fifo_depth",
        type=int,
        choices=(1, 2),
        default=2,
        help="Compute-side output C FIFO depth (default: 2; override requires fork)",
    )
    parser.add_argument(
        "--fifo",
        "--fifo-lowering",
        "--fifo_lowering",
        dest="fifo_lowering",
        choices=("auto", "static", "dynamic"),
        default="auto",
        help="FIFO lowering policy (default: auto; override requires fork)",
    )
    parser.add_argument(
        "--kernel",
        "--kernel-variant",
        "--kernel_variant",
        dest="kernel_variant",
        choices=("default", "unroll", "full_unroll_k"),
        default="default",
        help="Experimental K-loop unrolling: unroll = full_unroll_k",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument(
        "--project", help="Build directory; defaults to configuration name"
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configuration only"
    )
    args = parser.parse_args()
    if args.kernel_variant == "unroll":
        args.kernel_variant = "full_unroll_k"
    if min(args.m, args.k, args.n, args.tmc, args.tk, args.tn) <= 0:
        parser.error("Matrix and tile dimensions must be positive")
    if args.frontend == "library" and args.rho != 1:
        parser.error("The library frontend is symmetric: use --rho 1")
    if args.tmc % args.rho or any(
        size % 4 for size in (args.tmc // args.rho, args.tk, args.tn)
    ):
        parser.error("TmA=TmC/rho, Tk and Tn must be integer multiples of four")
    if args.m % (4 * args.tmc) or args.n % (4 * args.tn) or args.k % args.tk:
        parser.error("Require M divisible by 4*TmC, N by 4*Tn, and K by Tk (16 cores)")
    if args.warmup < 0 or args.num_iters <= 0 or args.runs <= 0:
        parser.error("Warmup must be nonnegative; num-iters and runs must be positive")
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    # For inputs in [-2,2], every intermediate integer sum is exactly represented
    # by the float64 reference and also fits int32 before int16 narrowing.
    if args.k > np.iinfo(np.int32).max // 4:
        parser.error("K exceeds the bounded reference's int32 range")
    if args.project is None:
        args.project = (
            f"gemm_{args.frontend}_{args.m}_{args.k}_{args.n}_"
            f"c{args.tmc}_k{args.tk}_n{args.tn}_r{args.rho}"
        )
        if (args.output_fifo_depth, args.fifo_lowering, args.kernel_variant) != (
            2,
            "auto",
            "default",
        ):
            args.project += f"_cd{args.output_fifo_depth}_{args.fifo_lowering}_{args.kernel_variant}"
        args.project += ".prj"
    return args


def fifo_build_options(args, build_function):
    """Preserve main defaults; never silently discard a requested policy."""
    supported = inspect.signature(build_function).parameters
    options = {}
    for name, default, flag in (
        ("output_fifo_depth", 2, "--c-depth"),
        ("fifo_lowering", "auto", "--fifo"),
    ):
        value = getattr(args, name)
        if name in supported:
            options[name] = value
        elif value != default:
            raise ValueError(
                f"This Allo checkout does not support {flag} {value}; "
                "use allo-my-fork for FIFO overrides or keep main's defaults"
            )
    return options


def unroll_matmul_reduction(code):
    """Apply the single-C study's experimental int16 K-loop pragma locally.

    Keep arithmetic and kernel selection unchanged. The generated external*.cc
    retains the transformed source; an upstream loop change fails explicitly.
    """
    start = code.index("static inline void matmul_vectorized_2x2_mmul(")
    end = code.index("  event1();", start)
    kernel = code[start:end]
    loop = "          for (unsigned i = 1; i < colA; ++i)"
    if kernel.count(loop) != 1:
        raise ValueError("Unrecognized 2x2 matmul reduction loop for unrolling")
    kernel = kernel.replace(loop, "          #pragma clang loop unroll(full)\n" + loop)
    return code[:start] + kernel + code[end:]


def build(args):
    import allo
    import allo.dataflow as df
    import allo.backend.aie as aie_backend
    from allo.backend.aie import is_available
    from allo.ir.types import int16

    print(f"Allo loaded from: {allo.__file__}", flush=True)
    fifo_options = fifo_build_options(args, df.build)
    if not is_available():
        raise RuntimeError(
            "AIE toolchain unavailable; activate the correct environment"
        )
    if args.frontend == "library":
        from allo.library.aie.modules.gemm import GEMM

        region, mapping = GEMM(
            args.m,
            args.n,
            args.k,
            args.m // args.tmc,
            args.n // args.tn,
            args.k // args.tk,
            int16,
            int16,
        )
    else:
        if not Path(__file__).with_name("test_temporal_atb_gemm.py").is_file():
            raise RuntimeError(
                "Temporal frontend requires allo-my-fork; on main use --frontend library"
            )
        from test_temporal_atb_gemm import make_atb_gemm, gen_atb_mapping_primitives

        region = make_atb_gemm(
            args.m,
            args.n,
            args.k,
            Pm=args.m // args.tmc,
            Pn=args.n // args.tn,
            Pk=args.k // args.tk,
            rho=args.rho,
        )
        mapping = gen_atb_mapping_primitives(args.m // args.tmc, args.n // args.tn)
    # Match the study and the original large-GEMM benchmark on both branches.
    os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
    original_codegen = aie_backend.codegen_external_kernels

    def unrolled_codegen(*codegen_args, **codegen_kwargs):
        return unroll_matmul_reduction(
            original_codegen(*codegen_args, **codegen_kwargs)
        )

    context = (
        patch.object(aie_backend, "codegen_external_kernels", unrolled_codegen)
        if args.kernel_variant == "full_unroll_k"
        else nullcontext()
    )
    with context:
        return df.build(
            region,
            target="aie",
            project=args.project,
            mapping_primitives=mapping,
            profile=True,
            warmup=args.warmup,
            num_iters=args.num_iters,
            **fifo_options,
        )


def main():
    args = parse_args()
    config = dict(vars(args))
    config.update(
        tma=args.tmc // args.rho,
        pm=args.m // args.tmc,
        pn=args.n // args.tn,
        pk=args.k // args.tk,
        physical_cores=16,
        precision="int16 A/B/C",
        dma_port_spreading=True,
    )
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        return
    module = build(args)
    project = Path(args.project)
    (project / "benchmark_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.build_only:
        print("BUILD PASSED (no hardware execution)", flush=True)
        return
    rng = np.random.default_rng(args.seed)
    left = rng.integers(-2, 3, (args.m, args.k), dtype=np.int16)
    right = rng.integers(-2, 3, (args.k, args.n), dtype=np.int16)
    reference = (
        (left.astype(np.float64) @ right.astype(np.float64))
        .astype(np.int32)
        .astype(np.int16)
    )
    for run in range(args.runs):
        output = np.full((args.m, args.n), 123, dtype=np.int16)
        print(
            f"Run {run + 1}/{args.runs}: {args.warmup} warmups, "
            f"{args.num_iters} timed launches",
            flush=True,
        )
        module(left, right, output)
        np.testing.assert_array_equal(output, reference)
        print(f"Run {run + 1}: exact full-output verification PASSED", flush=True)
    (project / "verification.json").write_text(
        json.dumps({"passed": True, "runs": args.runs, "seed": args.seed}, indent=2)
        + "\n"
    )
    print(
        "Use the median of the printed Avg NPU execution times. "
        f"TOPS = {2 * args.m * args.k * args.n} / (latency_us * 1e6).",
        flush=True,
    )


if __name__ == "__main__":
    main()
