# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and inspect one fixed-tile int16 GEMM benchmark (no device execution).

Use an isolated process per configuration: Allo captures frontend globals.
The external study driver supplies inputs and profiles the generated host.
"""
import argparse
from contextlib import nullcontext
from unittest.mock import patch
import json
import math
import os
from pathlib import Path
import re
import time

import allo.dataflow as df
import allo.backend.aie as aie_backend
from allo.ir.types import int16
from allo.library.aie.modules.gemm import GEMM
from test_temporal_atb_gemm import make_atb_gemm, gen_atb_mapping_primitives


def inspect_project(project, spec):
    ir = (project / "top.mlir").read_text()
    physical = (project / "top.mlir.prj/input_with_addresses.mlir").read_text()
    traffic = {}
    descriptor_count = 0
    for arg, sizes in re.findall(
        r"aiex.npu.dma_memcpy_nd\(%arg(\d+)\[[^]]+\]\[([^]]+)\]", ir
    ):
        dims = [int(x) for x in sizes.split(", ")]
        assert dims[2] <= 1023, dims
        traffic[arg] = traffic.get(arg, 0) + 2 * math.prod(dims)
        descriptor_count += 1
    buffers = {}
    for tile, dims in re.findall(
        r"aie.buffer\(%((?:mem_)?tile_\d_\d)\)[^\n]*memref<([\dx]+)xi16>",
        physical,
    ):
        buffers[tile] = buffers.get(tile, 0) + 2 * math.prod(map(int, dims.split("x")))
    core_bytes = [v for k, v in buffers.items() if k.startswith("tile_")]
    m, k, n = spec["m"], spec["k"], spec["n"]
    tmc, tk, tn, rho = (spec[x] for x in ("tmc", "tk", "tn", "rho"))
    calls = (m // tmc) * (n // tn) * (k // tk) * rho
    assert ir.count(" = aie.core(") == 16
    expected_traffic = {
        "0": 2 * m * k * (n // tn) // 4,
        "2": 2 * k * n * (m // tmc) // 4,
        "1": 2 * m * n,
    }
    assert traffic == expected_traffic, (traffic, expected_traffic)
    assert len(core_bytes) == 16
    assert min(core_bytes) == max(core_bytes)
    output_depth = spec.get("output_fifo_depth", 2)
    assert max(core_bytes) == (
        4 * ((tmc // rho) * tk + tk * tn) + 2 * output_depth * tmc * tn
    )
    return {
        "physical_cores": 16,
        "core_buffer_bytes_min": min(core_bytes),
        "core_buffer_bytes_max": max(core_bytes),
        "buffers_by_tile": buffers,
        "core_memref_copies": ir.count("memref.copy"),
        "private_buffers": ir.count(" = aie.buffer("),
        "dma_bytes_by_runtime_argument": traffic,
        "dma_bytes_a": traffic["0"],
        "dma_bytes_b": traffic["2"],
        "dma_bytes_c": traffic["1"],
        "dma_bytes": sum(traffic.values()),
        "dma_descriptors": descriptor_count,
        "dma_waits": ir.count("aiex.npu.dma_wait"),
        "microkernel_calls_total": calls,
        "microkernel_calls_per_core": calls // 16,
        "ops": 2 * m * k * n,
        "programmed_intensity_ops_per_byte": 2 * m * k * n / sum(traffic.values()),
    }


def unroll_matmul_reduction(code):
    """Experimental Peano scheduling ablation for the benchmark's int16 kernel.

    Keep upstream kernel selection and arithmetic unchanged. Fail explicitly if
    the installed source no longer has the expected 2x2 MMUL reduction loop.
    The generated external*.cc files retain the exact source used for a run.
    """
    start = code.index("static inline void matmul_vectorized_2x2_mmul(")
    end = code.index("  event1();", start)
    kernel = code[start:end]
    loop = "          for (unsigned i = 1; i < colA; ++i)"
    if kernel.count(loop) != 1:
        raise ValueError("Unrecognized 2x2 matmul reduction loop for unrolling")
    kernel = kernel.replace(loop, "          #pragma clang loop unroll(full)\n" + loop)
    return code[:start] + kernel + code[end:]


def build_one(spec, project):
    # Keep frontend factory names out of the caller's globals: otherwise
    # dimension names in the benchmark harness can shadow captured constants.
    assert spec["kind"] in ("symmetric", "atb")
    assert type(spec["rho"]) is int and spec["rho"] >= 1
    assert spec["tmc"] % spec["rho"] == 0
    assert spec["m"] % (4 * spec["tmc"]) == 0
    assert spec["n"] % (4 * spec["tn"]) == 0
    assert spec["k"] % spec["tk"] == 0
    assert spec.get("frontend", "library") in ("library", "temporal")
    if spec["kind"] == "symmetric" and spec.get("frontend") != "temporal":
        assert spec["rho"] == 1
        region, mapping = GEMM(
            spec["m"],
            spec["n"],
            spec["k"],
            spec["m"] // spec["tmc"],
            spec["n"] // spec["tn"],
            spec["k"] // spec["tk"],
            int16,
            int16,
        )
    else:
        if spec["kind"] == "symmetric":
            assert spec["rho"] == 1
        region = make_atb_gemm(
            spec["m"],
            spec["n"],
            spec["k"],
            Pm=spec["m"] // spec["tmc"],
            Pn=spec["n"] // spec["tn"],
            Pk=spec["k"] // spec["tk"],
            rho=spec["rho"],
        )
        mapping = gen_atb_mapping_primitives(
            spec["m"] // spec["tmc"], spec["n"] // spec["tn"]
        )
    os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
    kernel_variant = spec.get("kernel_variant", "default")
    assert kernel_variant in ("default", "full_unroll_k")
    original_codegen = aie_backend.codegen_external_kernels

    def unrolled_codegen(*args, **kwargs):
        return unroll_matmul_reduction(original_codegen(*args, **kwargs))

    context = (
        patch.object(aie_backend, "codegen_external_kernels", unrolled_codegen)
        if kernel_variant == "full_unroll_k"
        else nullcontext()
    )
    with context:
        df.build(
            region,
            target="aie",
            project=str(project),
            mapping_primitives=mapping,
            output_fifo_depth=spec.get("output_fifo_depth", 2),
            fifo_lowering=spec.get("fifo_lowering", "auto"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="JSON configuration")
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    specification = json.loads(args.spec)
    destination = Path(args.project)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "spec.json").write_text(json.dumps(specification, indent=2) + "\n")
    started = time.monotonic()
    build_one(specification, destination)
    metadata = inspect_project(destination, specification)
    metadata["build_seconds"] = time.monotonic() - started
    metadata["spec"] = specification
    (destination / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("BUILD_AND_INSPECTION_PASSED", json.dumps(metadata), flush=True)
