# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEMM with spatial M/N sharding and temporal K/rho sharding (ATB).

Run with --check-only for frontend, transfer-plan, and AIE optimizer checks.
Run with --build-only to compile the complete AIE project without execution.
Run without arguments in the AIE container to also build and execute on device.
Each core retains a full C tile while consuming half-height A tiles and
reusing each B tile across both rho steps. The default problem is 128x128x128.
Larger problems select additional logical M/N tiles and bundle them onto a
4x4 physical mesh; --m 1024 --n 1024 --k 1024 needs no manual mapping options.
"""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import re

import numpy as np

import allo
import allo.dataflow as df
from allo.backend.aie import AIE_MLIRModule, is_available
from allo.ir.types import int16
from allo.ir.utils import get_global_vars
from allo.memory import Axis, DTensor, Layout

S = Layout.Shard


def gen_atb_mapping_primitives(Pm, Pn, row_num=4, col_num=4):
    """Bundle the logical M/N mesh onto a bounded physical mesh.

    K and rho already run inside each kernel. Only independent output tiles
    are bundled; no temporal kernels or reduction chains are created.
    """
    if any(type(x) is not int or x <= 0 for x in (Pm, Pn, row_num, col_num)):
        raise ValueError("Logical and physical mesh extents must be positive integers")
    primitives = []
    for m in range(min(Pm, row_num)):
        for n in range(min(Pn, col_num)):
            kernels = [
                f"gemm_{i}_{j}"
                for i in range(m, Pm, row_num)
                for j in range(n, Pn, col_num)
            ]
            if len(kernels) > 1:
                primitives.append(("bundle", kernels))
    return primitives


def make_atb_gemm(M, N, K, Pn=4, Pk=4, rho=2, Pm=4):
    """Return a GEMM region with A[M/(Pm*rho),K/Pk] and C[M/Pm,N/Pn].

    One logical instance per M/N partition; temporal coordinates do not create
    kernel instances and consequently need no chain mapping primitives.
    """
    if any(type(x) is not int or x <= 0 for x in (M, N, K, Pm, Pn, Pk, rho)):
        raise ValueError(
            "Dimensions, partition counts, and rho must be positive integers"
        )
    if M % (Pm * rho) or N % Pn or K % Pk:
        raise ValueError("Require M % (Pm * rho) == N % Pn == K % Pk == 0")
    Mt, At, Nt, Kt = M // Pm, M // (Pm * rho), N // Pn, K // Pk
    Sm = Axis.Spatial(Pm, name="m")
    Sn = Axis.Spatial(Pn, name="n")
    Tk = Axis.Temporal(Pk, name="k")
    Tr = Axis.Temporal(rho, name="rho")
    LyA = [S(0) * S(3), S(2)]
    LyB = [S(2), S(1)]
    LyC = [S(0), S(1)]

    @df.region()
    def top(A: int16[M, K], B: int16[K, N], C: int16[M, N]):
        @df.kernel(mapping=[Sm, Sn, Tk, Tr], args=[A, B, C])
        def gemm(
            A_port: int16[M, K] @ LyA,
            B_port: int16[K, N] @ LyB,
            C_port: int16[M, N] @ LyC,
        ):
            acc: int16[rho, At, Nt] = 0
            for k in range(Pk):
                # B is invariant over rho: fetch outside the inner loop.
                b: int16[Kt, Nt] = B_port.get()
                for r in range(rho):
                    a: int16[At, Kt] = A_port.get()
                    current: int16[At, Nt]
                    current[:, :] = acc[r, :, :]
                    partial: int16[At, Nt] = allo.matmul(a, b)
                    acc[r, :, :] = allo.add(current, partial)
            C_port.put(allo.view(acc, (Mt, Nt)))

    return top


def check_atb_frontend(rho):
    """Compile the actual port-based kernel, including nested B reuse."""
    top = make_atb_gemm(128, 128, 256, rho=rho)
    schedule = allo.customize(
        top,
        global_vars=get_global_vars(top),
        unroll=False,
        typing_rule_set="cpp-style",
    )
    assert schedule.module.operation.verify()


def check_atb_transfer_plan(rho, Pk):
    """Replay compiler-derived port tile IDs and compare with full GEMM.

    This checks traversal and arithmetic in NumPy, not AIE execution or the
    unsupported simulator lowering of tensor ports.
    """
    M, N, K, Pm, Pn = 32, 24, 24, 2, 2
    Mt, At, Nt, Kt = M // Pm, M // (Pm * rho), N // Pn, K // Pk
    mapping = [
        Axis.Spatial(Pm, "m"),
        Axis.Spatial(Pn, "n"),
        Axis.Temporal(Pk, "k"),
        Axis.Temporal(rho, "rho"),
    ]
    tensors = [
        DTensor(2, mapping, (M, K), int16, Layout([S(0) * S(3), S(2)])),
        DTensor(2, mapping, (K, N), int16, Layout([S(2), S(1)])),
        DTensor(2, mapping, (M, N), int16, Layout([S(0), S(1)])),
    ]
    assert [t.get_local_shape() for t in tensors] == [(At, Kt), (Kt, Nt), (Mt, Nt)]
    rng = np.random.default_rng(42)
    A = rng.integers(-3, 4, (M, K), dtype=np.int16)
    B = rng.integers(-3, 4, (K, N), dtype=np.int16)
    C = np.zeros((M, N), dtype=np.int16)
    for m, n in np.ndindex(Pm, Pn):
        a_ids, b_ids, c_ids = [t.PE_tile_id_to_tensor_tile_id((m, n)) for t in tensors]
        assert a_ids == [(m * rho + r, k) for k in range(Pk) for r in range(rho)]
        assert b_ids == [(k, n) for k in range(Pk)]
        assert c_ids == [(m, n)]
        a_cursor = iter(a_ids)
        acc = np.zeros((Mt, Nt), dtype=np.int16)
        for bk, bn in b_ids:
            b = B[bk * Kt : (bk + 1) * Kt, bn * Nt : (bn + 1) * Nt]
            for r in range(rho):
                ar, ak = next(a_cursor)
                a = A[ar * At : (ar + 1) * At, ak * Kt : (ak + 1) * Kt]
                acc[r * At : (r + 1) * At] += a @ b
        assert next(a_cursor, None) is None
        C[m * Mt : (m + 1) * Mt, n * Nt : (n + 1) * Nt] = acc
    np.testing.assert_array_equal(
        C.astype(np.int32), A.astype(np.int32) @ B.astype(np.int32)
    )


def test_atb_frontend():
    for rho in (1, 2, 4):
        check_atb_frontend(rho)


def test_atb_transfer_plan():
    for rho in (1, 2, 4):
        for Pk in (1, 3):
            check_atb_transfer_plan(rho, Pk)


def test_atb_mapping_primitives():
    """Cover each logical tile exactly once, including uneven logical meshes."""
    for Pm, Pn in ((4, 4), (16, 16), (16, 8), (5, 7), (1, 12)):
        primitives = gen_atb_mapping_primitives(Pm, Pn)
        all_kernels = {f"gemm_{m}_{n}" for m in range(Pm) for n in range(Pn)}
        bundled = []
        for kind, kernels in primitives:
            assert kind == "bundle"
            assert len(kernels) > 1
            bundled.extend(kernels)
        assert len(bundled) == len(set(bundled))
        assert set(bundled) <= all_kernels
        remaining = all_kernels - set(bundled)
        assert len(primitives) + len(remaining) == min(Pm, 4) * min(Pn, 4)
    large = gen_atb_mapping_primitives(16, 16)
    assert len(large) == 16
    assert all(len(kernels) == 16 for _, kernels in large)


def test_atb_bundled_codegen():
    """Compile 256 logical kernels onto 16 cores without executing hardware."""
    if not is_available():
        import pytest

        pytest.skip("AIE code generation requires the AIE toolchain")
    with TemporaryDirectory(prefix="atb-bundled-") as project:
        with patch.object(AIE_MLIRModule, "post_codegen_build"):
            mod = df.build(
                make_atb_gemm(1024, 1024, 1024, Pm=16, Pn=16, Pk=32),
                target="aie",
                project=project,
                mapping_primitives=gen_atb_mapping_primitives(16, 16),
            )
        assert mod.aie_module.operation.verify()
        ir = str(mod.aie_module)
        assert ir.count(" = aie.core(") == 16
        assert ir.count("dynamic_objfifo_lowering = true") == 16
        assert "memref.copy" not in ir
        assert " = aie.buffer(" not in ir
        # Bundling must preserve ATB and reuse the local buffers for every tile.
        assert (
            len(
                re.findall(
                    r"aie.objectfifo @fifo_\d+\(%mem_tile[^\n]+memref<32x32xi16>", ir
                )
            )
            == 4
        )
        assert (
            len(
                re.findall(
                    r"aie.objectfifo @fifo_\d+\(%shim_noc_tile[^\n]+memref<1x1x64x32xi16>",
                    ir,
                )
            )
            == 4
        )
        opt = Path(project, "original_opt.mlir").read_text(encoding="utf-8")
        assert opt.count("func.func @gemm_") == 16
        nodes = mod.virtual_computation_graph.nodes
        assert len(nodes) == 16
        assert all(name.endswith("x16") for name in nodes)
        assert all(len(node.meta_data.df_kernels) == 16 for node in nodes.values())


def test_atb_dma_port_spreading():
    """Check legal descriptors, port budgets, multicast, and K/rho ordering."""
    if not is_available():
        import pytest

        pytest.skip("AIE code generation requires the AIE toolchain")
    from allo.backend.aie.mlir_codegen import CodeGenerator

    generators = []
    original_codegen = CodeGenerator.aie_codegen

    def capture_codegen(generator, *args, **kwargs):
        result = original_codegen(generator, *args, **kwargs)
        generators.append(generator)
        return result

    with TemporaryDirectory(prefix="atb-dma-") as project:
        with (
            patch.dict(os.environ, {"ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH": "1"}),
            patch.object(AIE_MLIRModule, "post_codegen_build"),
            patch.object(CodeGenerator, "aie_codegen", capture_codegen),
        ):
            mod = df.build(
                make_atb_gemm(1024, 1024, 1024, Pm=32, Pn=16, Pk=16),
                target="aie",
                project=project,
                mapping_primitives=gen_atb_mapping_primitives(32, 16),
            )
        assert mod.aie_module.operation.verify()
        ir = str(mod.aie_module)
        assert ir.count(" = aie.core(") == 16
        assert "memref.copy" not in ir
        assert " = aie.buffer(" not in ir
        generator = generators[0]
        for tile in generator.used_shim_tiles + generator.used_mem_tiles:
            assert len(tile.send_ports) <= tile.max_send
            assert len(tile.recv_ports) <= tile.max_recv
        assert [len(tile.send_ports) for tile in generator.used_shim_tiles] == [2] * 4

        # B stays 64x64 in L2. Each column gets its own route; four compute
        # cores share each B object. A retains the full 32-row L2 object.
        for shape in ("32x64", "64x64"):
            routes = re.findall(
                r"aie.objectfifo @(fifo_\d+)\(%shim_noc_tile_(\d)_0, "
                r"\{%mem_tile_(\d)_1\}, 2 : i32\) : "
                rf"!aie.objectfifo<memref<1x1x{shape}xi16>>",
                ir,
            )
            assert len(routes) == 4
            assert {(int(shim), int(mem)) for _, shim, mem in routes} == {
                (i, i) for i in range(4)
            }
            for fifo, col, _ in routes:
                downstream = re.search(
                    rf"aie.objectfifo.link \[@{fifo}\] -> \[@(fifo_\d+)\]", ir
                )[1]
                multicast = next(
                    line
                    for line in ir.splitlines()
                    if line.strip().startswith(f"aie.objectfifo @{downstream}(")
                )
                destinations = {
                    (int(x), int(y))
                    for x, y in re.findall(r"%tile_(\d)_(\d)", multicast)
                }
                expected_destinations = (
                    {(int(col), row) for row in range(2, 6)}
                    if shape == "32x64"
                    else {(column, int(col) + 2) for column in range(4)}
                )
                assert destinations == expected_destinations
                descriptors = re.findall(
                    r"aiex.npu.dma_memcpy_nd\(%arg\d+\[([^]]+)\]"
                    r"\[([^]]+)\]\[([^]]+)\]\) \{[^}]*metadata = @" + fifo + r"\}",
                    ir,
                )
                assert descriptors
                addresses = []
                for offsets, sizes, strides in descriptors:
                    offsets, sizes, strides = (
                        [int(v) for v in field.split(", ")]
                        for field in (offsets, sizes, strides)
                    )
                    assert 0 < sizes[2] <= 1023
                    assert 0 < sizes[0] <= 64
                    base = sum(o * st for o, st in zip(offsets, strides))
                    indices = np.indices(sizes, dtype=np.int64)
                    addresses.append(
                        (base + sum(indices[i] * strides[i] for i in range(4))).ravel()
                    )
                actual = np.concatenate(addresses)
                # Reconstruct the consumer's stream independently from the
                # descriptors. Bundles traverse M, N, K, then rho for A only.
                if shape == "64x64":
                    expected = np.concatenate(
                        [
                            (
                                (np.arange(1024)[:, None] * 1024)
                                + (int(col) + 4 * n) * 64
                                + np.arange(64)
                            ).ravel()
                            for _ in range(8)
                            for n in range(4)
                        ]
                    )
                else:
                    expected = np.concatenate(
                        [
                            (
                                ((int(col) + 4 * m) * 32 + np.arange(32)[:, None])
                                * 1024
                                + k * 64
                                + np.arange(64)
                            ).ravel()
                            for m in range(8)
                            for _ in range(4)
                            for k in range(16)
                        ]
                    )
                np.testing.assert_array_equal(actual, expected)


def test_atb_aie_optimization():
    """Exercise the optimizer that frontend-only verification does not run.

    Stop after allo_opt, before device code generation or hardware execution.
    In particular this catches matmul fusion using a later accumulator buffer.
    """
    if not is_available():
        import pytest

        pytest.skip("AIE optimization check requires the AIE toolchain")

    class OptimizationChecked(Exception):
        pass

    original_opt = AIE_MLIRModule.allo_opt

    def verify_optimization(module):
        original_opt(module)
        assert module.allo_module.operation.verify()
        raise OptimizationChecked

    cases = (
        (128, 128, 128, 4, 2),  # complete-build smoke case
        (128, 128, 256, 8, 1),
        (128, 128, 256, 8, 2),  # reported failing shape
        (128, 128, 256, 8, 4),
    )
    for M, N, K, Pk, rho in cases:
        with TemporaryDirectory(prefix="atb-opt-") as project:
            with patch.object(AIE_MLIRModule, "allo_opt", verify_optimization):
                try:
                    df.build(
                        make_atb_gemm(M, N, K, Pk=Pk, rho=rho),
                        target="aie",
                        project=project,
                    )
                except OptimizationChecked:
                    continue
                raise AssertionError("AIE optimization was not checked")


def test_atb_aie_codegen():
    """Check physical buffers and loop placement without building a bitstream."""
    if not is_available():
        import pytest

        pytest.skip("AIE code generation requires the AIE toolchain")
    with TemporaryDirectory(prefix="atb-codegen-") as project:
        with patch.object(AIE_MLIRModule, "post_codegen_build"):
            mod = df.build(make_atb_gemm(128, 128, 128), target="aie", project=project)
        assert mod.aie_module.operation.verify()
        ir = str(mod.aie_module)
        assert ir.count(" = aie.core(") == 16
        assert ir.count("dynamic_objfifo_lowering = true") == 16
        assert "memref.copy" not in ir
        assert " = aie.buffer(" not in ir
        # Four A multicasts: L2 keeps 32 rows; each L1 object contains 16.
        a_l1 = re.findall(
            r"aie.objectfifo @fifo_\d+\(%mem_tile[^\n]+memref<16x32xi16>", ir
        )
        a_l2 = re.findall(
            r"aie.objectfifo @fifo_\d+\(%shim_noc_tile[^\n]+memref<1x1x32x32xi16>", ir
        )
        assert len(a_l1) == len(a_l2) == 4
        opt = Path(project, "original_opt.mlir").read_text(encoding="utf-8")
        # B is acquired in K's loop, A in rho's loop; only spatial PEs exist.
        kernel = opt[
            opt.index("func.func @gemm_0_0") : opt.index("func.func @gemm_0_1")
        ]
        assert "allo.transform_layout" not in kernel
        assert "memref.copy" not in kernel
        assert "_slice_2x16x32" in kernel
        assert kernel.count("allo.stream_get") == 2
        assert kernel.count("allo.stream_put") == 1
        assert kernel.index('name = "b"') < kernel.index('name = "a"')


def run_aie(
    build_only=False,
    project="temporal_atb_gemm.prj",
    M=128,
    N=128,
    K=128,
    Pm=None,
    Pn=None,
    Pk=None,
    output_fifo_depth=2,
):
    """Explicit hardware entry point; deliberately not collected by pytest."""
    if not is_available():
        raise RuntimeError(
            "AIE toolchain unavailable; use --check-only for software checks"
        )
    # Bound the local C tile as global matrices grow. Preserve the 4x4
    # defaults for the original 128/256 cases; bundle larger logical meshes.
    if Pm is None:
        Pm = max(4, (M + 63) // 64)
    if Pn is None:
        Pn = max(4, (N + 63) // 64)
    if Pk is None:
        if K % 32:
            raise ValueError(
                "Default K tile is 32; choose K divisible by 32 or specify --pk"
            )
        Pk = K // 32
    top = make_atb_gemm(M, N, K, Pm=Pm, Pn=Pn, Pk=Pk, rho=2)
    mapping_primitives = gen_atb_mapping_primitives(Pm, Pn)
    physical_cores = min(Pm, 4) * min(Pn, 4)
    print(
        f"Logical mesh {Pm}x{Pn} -> {physical_cores} cores; "
        f"{len(mapping_primitives)} spatial bundles",
        flush=True,
    )
    mod = df.build(
        top,
        target="aie",
        project=project,
        mapping_primitives=mapping_primitives,
        output_fifo_depth=output_fifo_depth,
    )
    if build_only:
        print("AIE ATB GEMM build passed (rho=2); hardware execution not requested")
        return
    rng = np.random.default_rng(7)
    for run in range(3):
        A = rng.integers(-3, 4, (M, K), dtype=np.int16)
        B = rng.integers(-3, 4, (K, N), dtype=np.int16)
        if run == 2:
            A.fill(0)
        C = np.full((M, N), 123, dtype=np.int16)
        mod(A, B, C)
        np.testing.assert_array_equal(
            C.astype(np.int32), A.astype(np.int32) @ B.astype(np.int32)
        )
    print(f"AIE ATB GEMM passed: M={M}, N={N}, K={K}, rho=2, mesh={Pm}x{Pn} (3 runs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--build-only", action="store_true")
    parser.add_argument("--project", default="temporal_atb_gemm.prj")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument(
        "--pm",
        type=int,
        default=None,
        help="Logical M partitions (default: max(4, ceil(M/64)))",
    )
    parser.add_argument(
        "--pn",
        type=int,
        default=None,
        help="Logical N partitions (default: max(4, ceil(N/64)))",
    )
    parser.add_argument(
        "--pk", type=int, default=None, help="Temporal K partitions (default: K/32)"
    )
    parser.add_argument("--output-fifo-depth", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    test_atb_frontend()
    test_atb_transfer_plan()
    test_atb_mapping_primitives()
    print("Frontend and NumPy transfer-plan checks passed")
    if args.check_only:
        if is_available():
            test_atb_aie_optimization()
            test_atb_aie_codegen()
            test_atb_bundled_codegen()
            print(
                "AIE optimization and code generation checks passed; no hardware executed"
            )
        else:
            print("AIE optimization checks skipped: toolchain unavailable")
    else:
        run_aie(
            build_only=args.build_only,
            project=args.project,
            M=args.m,
            N=args.n,
            K=args.k,
            Pm=args.pm,
            Pn=args.pn,
            Pk=args.pk,
            output_fifo_depth=args.output_fifo_depth,
        )
