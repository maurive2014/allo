# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one int16 GEMM configuration on 16 AIE cores, with exact verification.

The library frontend works on main and the temporal-sharding-atb fork.
The temporal frontend requires that fork; rho=1 is symmetric buffering.
Tile arguments are dimensions, not partition counts. Use a new process for
each configuration because Allo captures Python frontend globals.
"""

import argparse
import json
import os
from pathlib import Path

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
            f"c{args.tmc}_k{args.tk}_n{args.tn}_r{args.rho}.prj"
        )
    return args


def build(args):
    import allo
    import allo.dataflow as df
    from allo.backend.aie import is_available
    from allo.ir.types import int16

    print(f"Allo loaded from: {allo.__file__}", flush=True)
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
    return df.build(
        region,
        target="aie",
        project=args.project,
        mapping_primitives=mapping,
        profile=True,
        warmup=args.warmup,
        num_iters=args.num_iters,
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
