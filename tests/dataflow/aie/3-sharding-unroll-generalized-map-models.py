# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import allo
import allo.dataflow as df
import numpy as np
from allo.backend.aie import is_available
from allo.ir.types import Stream, int16
from allo.memory import Layout


S = Layout.Shard
R = Layout.Replicate

Ty = int16
M, N, K = 64, 16, 16
# RHO_VALUES = [1, 2, 4, 8]

# Mapping model:
# 1 = bundle
# 2 = bundle + chain(compute -> store)
# 3 = bundle + chain(load_b -> compute) 
# 4 = bundle + both chains
MODEL = 2

def make_atb_top(rho):
    assert M % rho == 0
    Ma = M // rho

    @df.region()
    def top(B: Ty[K, N], A: Ty[M, K], C: Ty[M, N]):
        pipeB: Stream[Ty[K, N], 1][rho]
        pipeC: Stream[Ty[Ma, N], 1][rho]

        @df.kernel(mapping=[1], args=[B])
        def load_b(local_B: Ty[K, N]):
            b = local_B
            with allo.meta_for(rho) as i:
                pipeB[i].put(b)

        @df.kernel(mapping=[rho], args=[A])
        def compute(local_A: Ty[M, K] @ [S(0), R]):
            pk = df.get_pid()
            c = allo.matmul(local_A, pipeB[pk].get())
            pipeC[pk].put(c)

        @df.kernel(mapping=[1], args=[C])
        def store_c(local_C: Ty[M, N]):
            with allo.meta_for(rho) as i:
                local_C[i * Ma : (i + 1) * Ma, :] = pipeC[i].get()

    return top

def run_atb(rho):
    top = make_atb_top(rho)
    mapping_primitives = None
    mapping_primitives = []

    # Bundle the replicated compute kernels when rho > 1
    if rho > 1:
        mapping_primitives.append(("bundle", [f"compute_{i}" for i in range(rho)]))

    # Name of the compute kernel after bundling if rho > 1.
    compute_name = f"compute_0x{rho}" if rho > 1 else "compute_0"
    load_b_name = "load_b_0"
    store_c_name = "store_c_0"
    load_b_compute_name = f"{load_b_name}-{compute_name}"

    # -------------------------------------------------------
    # Mapping models
    # -------------------------------------------------------

    # Model 1:
    # load_b -> compute -> store
    if MODEL == 1:
        pass

    # Model 2:
    # load_b -> (compute + store)
    elif MODEL == 2:
        mapping_primitives.append(("chain", [compute_name, store_c_name]))

    # Model 3:
    # (load_b + compute) -> store
    elif MODEL == 3:
        mapping_primitives.append(("chain", [load_b_name, compute_name]))

    # Model 4:
    # (load_b + compute + store)
    elif MODEL == 4:
        mapping_primitives.append(("chain", [load_b_name, compute_name]))
        mapping_primitives.append(("chain", [load_b_compute_name, store_c_name]))

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)

    if is_available():
        os.environ["FORCE_UNROLL_INDEX"] = "1"
        mod = df.build(
            top,
            target="aie",
            project="rho-model-loopv2.prj",
            mapping_primitives=mapping_primitives,
        )
        mod(B, A, C)
        del os.environ["FORCE_UNROLL_INDEX"]
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print(f"rho={rho} PASSED! for model {MODEL}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


run_atb(4) 
# for rho 1 model 1,2,3,4 work 
# for rho 2 model 1,2 work
# for rho 4 model 1 work
# rho = 8 doesn't work in any model