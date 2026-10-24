# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legality regressions for packed accumulators and FIFO view lifetimes."""
from types import SimpleNamespace

import pytest

pytest.importorskip("aie")
from allo._mlir import ir
from allo._mlir.dialects import allo as allo_d
from allo.backend.aie import _promote_sliced_matmul_accumulators
from allo.backend.aie.external_kernel import ExternalModuleBase
from allo.backend.aie.mlir_codegen import _output_view_allocation, _input_fifo_use_scope
import aie.ir as aie_ir


PACKED_INPUT = """
module {
  func.func private @matmul_i16_i16(memref<16x64xi16>, memref<64x64xi16>, memref<16x64xi16>)
  func.func @kernel(%a: memref<16x64xi16>, %b: memref<64x64xi16>, %out: !allo.stream<memref<32x64xi16>, 2>) {
    %zero = arith.constant 0 : i16
    %acc = memref.alloc() : memref<2x16x64xi16>
    linalg.fill ins(%zero : i16) outs(%acc : memref<2x16x64xi16>)
    affine.for %k = 0 to 3 {
      affine.for %r = 0 to 2 {
        %current = memref.alloc() : memref<16x64xi16>
        %read = memref.subview %acc[%r, 0, 0] [1, 16, 64] [1, 1, 1] : memref<2x16x64xi16> to memref<16x64xi16, strided<[64, 1], offset: ?>>
        memref.copy %read, %current : memref<16x64xi16, strided<[64, 1], offset: ?>> to memref<16x64xi16>
        %packed = allo.transform_layout %current offsets = [0, 0, 0, 0] sizes = [4, 16, 4, 4] strides = [256, 4, 64, 1] {layout_hint = "C_to_16x64x64_4x4x4"} : memref<16x64xi16> -> memref<16x64xi16>
        func.call @matmul_i16_i16(%a, %b, %packed) : (memref<16x64xi16>, memref<64x64xi16>, memref<16x64xi16>) -> ()
        %unpacked = allo.transform_layout %packed offsets = [0, 0, 0, 0] sizes = [4, 4, 16, 4] strides = [256, 4, 16, 1] {layout_hint = "C_from_16x64x64_4x4x4"} : memref<16x64xi16> -> memref<16x64xi16>
        %write = memref.subview %acc[%r, 0, 0] [1, 16, 64] [1, 1, 1] : memref<2x16x64xi16> to memref<16x64xi16, strided<[64, 1], offset: ?>>
        memref.copy %unpacked, %write : memref<16x64xi16> to memref<16x64xi16, strided<[64, 1], offset: ?>>
      }
    }
    %full = memref.reinterpret_cast %acc to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<2x16x64xi16> to memref<32x64xi16>
    allo.stream_put(%out, [], %full) : !allo.stream<memref<32x64xi16>, 2> contains memref<32x64xi16>
    return
  }
}
"""


@pytest.mark.parametrize(
    "variant", ["valid", "nonzero", "extra_read", "wrong_inverse", "aliased_input"]
)
def test_packed_accumulator_legality(variant):
    source = PACKED_INPUT
    if variant == "nonzero":
        source = source.replace("arith.constant 0 : i16", "arith.constant 1 : i16")
    elif variant == "extra_read":
        source = source.replace(
            "    %full =",
            "    %i = arith.constant 0 : index\n"
            "    %v = memref.load %acc[%i, %i, %i] : memref<2x16x64xi16>\n"
            "    %full =",
        )
    elif variant == "wrong_inverse":
        source = source.replace("strides = [256, 4, 16, 1]", "strides = [256, 4, 8, 1]")
    elif variant == "aliased_input":
        source = source.replace(
            "@matmul_i16_i16(%a, %b, %packed)", "@matmul_i16_i16(%packed, %b, %packed)"
        )
    with ir.Context() as ctx, ir.Location.unknown():
        allo_d.register_dialect(ctx)
        parsed = ir.Module.parse(source)
        function = list(parsed.body.operations)[1]
        module = SimpleNamespace(
            device="npu1",
            injected_external_kernels={
                "matmul_i16_i16": ExternalModuleBase("matmul_i16_i16", [0, 1], [2])
            },
            include_src={"matmul_i16_i16": '#include "aie2/mm.cc"\n'},
            virtual_computation_graph=SimpleNamespace(
                nodes={
                    "kernel": SimpleNamespace(
                        meta_data=SimpleNamespace(used_external_kernel=set())
                    )
                }
            ),
        )
        before = str(parsed)
        assert _promote_sliced_matmul_accumulators(module, function) == (
            variant == "valid"
        )
        assert parsed.operation.verify()
        if variant != "valid":
            assert str(parsed) == before
        else:
            result = str(function)
            assert "memref.copy" not in result
            assert "memref.subview" not in result
            assert "_slice_2x16x64" in result
            assert result.count("allo.transform_layout") == 1
            assert result.index("allo.transform_layout") > result.index(
                "memref.reinterpret_cast"
            )


@pytest.mark.parametrize("variant", ["valid", "post_use", "escape", "partial"])
def test_output_fifo_view_lifetime(variant):
    extra = ""
    post = ""
    shape = "32x64"
    sizes = "32, 64"
    if variant == "post_use":
        post = "func.call @fill_zeros_i16_32_64_vector_planes_2(%acc) : (memref<2x16x64xi16>) -> ()"
    elif variant == "escape":
        extra = "func.call @escape(%acc) : (memref<2x16x64xi16>) -> ()"
    elif variant == "partial":
        shape, sizes = "16x64", "16, 64"
    source = f"""
module {{
  func.func private @fill_zeros_i16_32_64_vector_planes_2(memref<2x16x64xi16>)
  func.func private @escape(memref<2x16x64xi16>)
  func.func @kernel(%out: memref<{shape}xi16>) {{
    %acc = memref.alloc() : memref<2x16x64xi16>
    func.call @fill_zeros_i16_32_64_vector_planes_2(%acc) : (memref<2x16x64xi16>) -> ()
    {extra}
    %full = memref.reinterpret_cast %acc to offset: [0], sizes: [{sizes}], strides: [64, 1] : memref<2x16x64xi16> to memref<{shape}xi16>
    memref.copy %full, %out : memref<{shape}xi16> to memref<{shape}xi16>
    {post}
    return
  }}
}}
"""
    with aie_ir.Context(), aie_ir.Location.unknown():
        parsed = aie_ir.Module.parse(source)
        assert parsed.operation.verify()
        function = list(parsed.body.operations)[2]
        copy = next(
            op
            for op in function.body.blocks[0].operations
            if op.operation.name == "memref.copy"
        )
        assert (_output_view_allocation(copy) is not None) == (variant == "valid")


@pytest.mark.parametrize(
    "variant", ["valid", "write", "unknown", "shared", "multiple", "conditional"]
)
def test_nested_input_fifo_lifetime(variant):
    name = "unknown" if variant == "unknown" else "matmul_i16_i16"
    operands = "%a, %c, %b" if variant == "write" else "%a, %b, %c"
    call = f"func.call @{name}({operands}) : (memref<16x16xi16>, memref<16x16xi16>, memref<16x16xi16>) -> ()"
    post = call if variant == "multiple" else ""
    if variant == "conditional":
        call = f"scf.if %cond {{ {call} }}"
    source = f"""
module {{
  func.func private @{name}(memref<16x16xi16>, memref<16x16xi16>, memref<16x16xi16>)
  func.func @kernel(%fifo: memref<16x16xi16>, %a: memref<16x16xi16>, %c: memref<16x16xi16>, %cond: i1) {{
    %b = memref.alloc() : memref<16x16xi16>
    memref.copy %fifo, %b : memref<16x16xi16> to memref<16x16xi16>
    affine.for %r = 0 to 2 {{
      affine.for %s = 0 to 2 {{ {call} }}
    }}
    {post}
    return
  }}
}}
"""
    with aie_ir.Context(), aie_ir.Location.unknown():
        parsed = aie_ir.Module.parse(source)
        assert parsed.operation.verify()
        function = list(parsed.body.operations)[1]
        copy = next(
            op
            for op in function.body.blocks[0].operations
            if op.operation.name == "memref.copy"
        )
        scope = _input_fifo_use_scope(copy, variant != "shared")
        if variant == "valid":
            assert scope.name == "affine.for"
            assert scope.parent == copy.parent
        else:
            assert scope is None
