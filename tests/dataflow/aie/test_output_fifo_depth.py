# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output FIFO depth is shared by library and temporal GEMM lowering."""
import re
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

import allo.dataflow as df
from allo.backend.aie import AIE_MLIRModule, is_available
from allo.ir.types import int16
from allo.library.aie.modules.gemm import GEMM
from test_temporal_atb_gemm import make_atb_gemm


@pytest.mark.parametrize("depth", [0, 3, True, 1.0])
def test_invalid_output_depth(depth):
    with pytest.raises(ValueError, match="output_fifo_depth must be 1 or 2"):
        df.build(None, target="aie", output_fifo_depth=depth)


def test_output_depth_requires_aie():
    with pytest.raises(ValueError, match="only supported for AIE"):
        df.build(None, target="simulator", output_fifo_depth=1)


@pytest.mark.parametrize("frontend", ["library", "temporal_sym", "atb"])
@pytest.mark.parametrize("depth", [1, 2])
def test_output_fifo_depth(frontend, depth):
    if not is_available():
        pytest.skip("AIE code generation requires the AIE toolchain")
    if frontend == "library":
        region, mapping = GEMM(256, 256, 128, 4, 4, 2, int16, int16)
    else:
        region = make_atb_gemm(256, 256, 128, Pk=2, rho=2 if frontend == "atb" else 1)
        mapping = []
    with TemporaryDirectory(prefix="output-depth-") as project:
        with patch.object(AIE_MLIRModule, "post_codegen_build"):
            mod = df.build(
                region,
                target="aie",
                project=project,
                mapping_primitives=mapping,
                output_fifo_depth=depth,
            )
        assert mod.aie_module.operation.verify()
        ir = str(mod.aie_module)
        fifos = [line for line in ir.splitlines() if "aie.objectfifo @" in line]
        outputs = [line for line in fifos if re.search(r"\(%tile_\d_\d,", line)]
        assert len(outputs) == 16
        expected = "[1 : i32, 2 : i32]" if depth == 1 else "2 : i32"
        assert all(expected in line for line in outputs)
        # Input and shim/memory endpoints retain double buffering.
        assert all("2 : i32" in line for line in fifos if line not in outputs)
        assert "memref.copy" not in ir
        assert " = aie.buffer(" not in ir


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_fifo_lowering_override(mode):
    if not is_available():
        pytest.skip("AIE code generation requires the AIE toolchain")
    with TemporaryDirectory(prefix="fifo-lowering-") as project:
        with patch.object(AIE_MLIRModule, "post_codegen_build"):
            mod = df.build(
                make_atb_gemm(128, 128, 128),
                target="aie",
                project=project,
                output_fifo_depth=1,
                fifo_lowering=mode,
            )
        assert mod.aie_module.operation.verify()
        expected = "true" if mode == "dynamic" else "false"
        assert str(mod.aie_module).count(f"dynamic_objfifo_lowering = {expected}") == 16


def test_invalid_fifo_lowering():
    with pytest.raises(ValueError, match="fifo_lowering must be"):
        df.build(None, target="aie", fifo_lowering="unknown")
    with pytest.raises(ValueError, match="only supported for AIE"):
        df.build(None, target="simulator", fifo_lowering="static")
