# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np
import allo
import allo.dataflow as df
from allo.backend.aie import AIE_MLIRModule, is_available
from allo.ir.types import int16, Stream
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


@pytest.mark.parametrize("rho", [1, 2, 4])
def test_atb_onchip_tiling(rho):
    Ty = int16
    M, N, K = 64, 16, 16
    assert M % rho == 0
    Ma = M // rho

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe_a: Stream[Ty[Ma, K], 2][rho]
        pipe_b: Stream[Ty[K, N], 2][rho]
        pipe_c: Stream[Ty[Ma, N], 2][rho]

        @df.kernel(mapping=[1], args=[A])
        def load_a(local_A: Ty[M, K]):
            with allo.meta_for(rho) as i:
                pipe_a[i].put(local_A[i * Ma : (i + 1) * Ma, :])

        @df.kernel(mapping=[1], args=[B])
        def load_b(local_B: Ty[K, N]):
            with allo.meta_for(rho) as i:
                pipe_b[i].put(local_B)

        @df.kernel(mapping=[rho])
        def compute():
            pk = df.get_pid()
            local_A: Ty[Ma, K] = pipe_a[pk].get()
            local_B: Ty[K, N] = pipe_b[pk].get()
            pipe_c[pk].put(allo.matmul(local_A, local_B))

        @df.kernel(mapping=[1], args=[C])
        def store_c(local_C: Ty[M, N]):
            with allo.meta_for(rho) as i:
                local_C[i * Ma : (i + 1) * Ma, :] = pipe_c[i].get()

    mapping_primitives = None
    if rho > 1:
        mapping_primitives = [("bundle", [f"compute_{i}" for i in range(rho)])]

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)

    if is_available():
        os.environ["FORCE_UNROLL_INDEX"] = "1"
        mod = df.build(top, target="aie", mapping_primitives=mapping_primitives)
        mod(A, B, C)
        del os.environ["FORCE_UNROLL_INDEX"]
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print(f"rho={rho} PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


@pytest.mark.parametrize("rho", [1, 2, 4])
def test_atb(rho):
    Ty = int16
    M, N, K = 64, 16, 16
    assert M % rho == 0
    Ma = M // rho

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipeB: Stream[Ty[K, N], 1][rho]
        pipeC: Stream[Ty[Ma, N], 1][rho]

        @df.kernel(mapping=[1], args=[B])
        def loadB(local_B: Ty[K, N]):
            with allo.meta_for(rho) as i:
                pipeB[i].put(local_B)

        @df.kernel(mapping=[rho], args=[A])
        def compute(local_A: Ty[M, K] @ [S(0), R]):
            pk = df.get_pid()
            c = allo.matmul(local_A, pipeB[pk].get())
            pipeC[pk].put(c)

        @df.kernel(mapping=[1], args=[C])
        def store(local_C: Ty[M, N]):
            with allo.meta_for(rho) as i:
                local_C[i * Ma : (i + 1) * Ma, :] = pipeC[i].get()

    mapping_primitives = None
    if rho > 1:
        mapping_primitives = [("bundle", [f"compute_{i}" for i in range(rho)])]

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)

    if is_available():
        os.environ["FORCE_UNROLL_INDEX"] = "1"
        mod = df.build(top, target="aie", mapping_primitives=mapping_primitives)
        mod(A, B, C)
        del os.environ["FORCE_UNROLL_INDEX"]
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print(f"rho={rho} PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


@pytest.mark.skipif(not is_available(), reason="MLIR-AIE is unavailable")
def test_bundle_chain_bufferizes_all_stream_aliases(tmp_path, monkeypatch):
    """Bundled stream aliases must all become accesses to the same local buffer."""
    Ty = int16
    rho = 4
    M, N, K = 64, 16, 16
    Ma = M // rho

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipeB: Stream[Ty[K, N], 1][rho]
        pipeC: Stream[Ty[Ma, N], 1][rho]

        @df.kernel(mapping=[1], args=[B])
        def loadB(local_B: Ty[K, N]):
            with allo.meta_for(rho) as i:
                pipeB[i].put(local_B)

        @df.kernel(mapping=[rho], args=[A])
        def compute(local_A: Ty[M, K] @ [S(0), R]):
            pk = df.get_pid()
            pipeC[pk].put(allo.matmul(local_A, pipeB[pk].get()))

        @df.kernel(mapping=[1], args=[C])
        def store(local_C: Ty[M, N]):
            with allo.meta_for(rho) as i:
                local_C[i * Ma : (i + 1) * Ma, :] = pipeC[i].get()

    monkeypatch.setenv("FORCE_UNROLL_INDEX", "1")
    monkeypatch.setattr(
        AIE_MLIRModule,
        "post_codegen_build",
        lambda _self, _external_cc_list: None,
    )
    project = tmp_path / "bundle-chain-bufferization.prj"
    core_name = "compute_0x4-store_0"
    mod = df.build(
        top,
        target="aie",
        project=str(project),
        mapping_primitives=[
            ("bundle", [f"compute_{i}" for i in range(rho)]),
            ("chain", ["compute_0x4", "store_0"]),
        ],
    )

    original_mlir = (project / "original.mlir").read_text(encoding="utf-8")
    core_mlir = original_mlir.split(f'func.func @"{core_name}"', 1)[1]
    core_mlir = core_mlir.split("func.func @top", 1)[0]
    assert core_mlir.count("allo.stream_get") == 1  # external pipeB
    assert "allo.stream_put" not in core_mlir
    assert "memref<4x16x16xi16>" in core_mlir
    assert "memref.subview %alloc[%" in core_mlir
    for slot in range(rho):
        assert f"[{slot}, 0, 0] [1, 16, 16]" in core_mlir

    optimized_mlir = (project / "original_opt.mlir").read_text(encoding="utf-8")
    assert "memref.collapse_shape" in optimized_mlir
    assert (
        "memref<4x16x16xi16> into memref<64x16xi16>" in optimized_mlir
    )

    remaining_stream_names = set()
    for arg_info in mod.core_func_args[core_name].values():
        arguments = arg_info[0] if isinstance(arg_info[0], list) else [arg_info[0]]
        remaining_stream_names.update(
            argument.stream.name
            for argument in arguments
            if argument.stream is not None
        )
    assert not any(name.startswith("pipeC") for name in remaining_stream_names)
    assert mod.aie_module is not None
    assert "aie.core" in str(mod.aie_module)


if __name__ == "__main__":
    RHO_VALUES = [1, 2, 4]  # Change rho value here
    for rho in RHO_VALUES:
        test_atb_onchip_tiling(rho)
    for rho in RHO_VALUES:
        test_atb(rho)
