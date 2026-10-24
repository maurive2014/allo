# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=c-extension-no-member, too-many-instance-attributes, too-many-nested-blocks, no-name-in-module, unsupported-binary-operation

import os
import re
import subprocess
import shutil
import numpy as np

try:
    import aie.ir as aie_ir
    import aie.passmanager as aie_pass_manager
except ImportError:
    pass

import allo._mlir._mlir_libs._mlir as allo_ir
from allo._mlir.exceptions import AlloWarning
from ..._mlir.dialects import (
    allo as allo_d,
    func as allo_func_d,
    _memref_ops_gen as allo_memref_d,
)
from ..._mlir.ir import (
    Type,
    StringAttr,
    InsertionPoint,
    FlatSymbolRefAttr,
    MemRefType,
)
from ..._mlir.passmanager import PassManager as mlir_pass_manager
from ...memory import DTensor
from ...utils import construct_kernel_name
from .external_kernel import ExternalModule, ExternalModuleBase
from .mlir_codegen import CodeGenerator, _is_builtin_matmul
from .vliw import vliw
from .utils import (
    Argument,
    Stream,
    inject_external_kernels,
    matmul_external_kernel_config_map,
    device_config_map,
    get_df_kernels,
    classify_aie_functions,
    codegen_external_kernels,
    simplify_matmul_accumulate,
    collect_lib_func_call,
    pack_int4,
    read_tensor_from_file,
    codegen_host,
    RuntimeArgs,
    is_inverse_transform_layout,
)
from .mapping import ComputationGraph


def is_available():
    return "MLIR_AIE_INSTALL_DIR" in os.environ


def _as_operation(op):
    """Return the generic MLIR operation for an operation or operation view."""
    return getattr(op, "operation", op)


def _is_nested_in_operation(op, ancestor):
    """Return whether ``op`` is ``ancestor`` or is nested under it."""
    op = _as_operation(op)
    ancestor = _as_operation(ancestor)
    while op is not None:
        if op == ancestor:
            return True
        op = getattr(op, "parent", None)
    return False


def _get_direct_child(op, ancestor):
    """Return the direct child of ``ancestor`` that contains ``op``."""
    op = _as_operation(op)
    ancestor = _as_operation(ancestor)
    while op is not None:
        parent = getattr(op, "parent", None)
        if parent == ancestor:
            return op
        op = parent
    return None


def _promote_loop_carried_layout_transformations(function):
    """Promote inverse layout-transform round trips out of affine loops."""

    def get_value_owner(value):
        owner = value.owner
        # Operation results are owned by operations; block arguments are owned
        # by blocks, whose owner is the enclosing operation.
        if getattr(owner, "name", None) is None:
            owner = owner.owner
        return _as_operation(owner)

    def match_tail_pairs(loop):
        if len(loop.regions) != 1 or len(loop.regions[0].blocks) != 1:
            return [], []
        operations = [_as_operation(op) for op in loop.regions[0].blocks[0]]
        if operations and operations[-1].name == "affine.yield":
            operations.pop()

        # Vectorization can leave trivially dead operations after the
        # accumulator copy. Canonicalization removes them later, so ignore
        # only these exact no-op forms when identifying the effective tail.
        trailing_noops = []
        while operations:
            trailing_op = operations[-1]
            is_noop_copy = (
                trailing_op.name == "memref.copy"
                and trailing_op.operands[0] == trailing_op.operands[1]
            )
            is_dead_alloc = trailing_op.name == "memref.alloc" and all(
                not list(result.uses) for result in trailing_op.results
            )
            if not (is_noop_copy or is_dead_alloc):
                break
            trailing_noops.append(operations.pop())
        operation_indices = {op: idx for idx, op in enumerate(operations)}
        pairs = []
        tail_idx = len(operations) - 1

        while tail_idx >= 1:
            copy_op = operations[tail_idx]
            from_op = operations[tail_idx - 1]
            if (
                copy_op.name != "memref.copy"
                or from_op.name != "allo.transform_layout"
                or copy_op.operands[0] != from_op.result
            ):
                break

            source = copy_op.operands[1]
            to_candidates = [
                op
                for op in operations[: tail_idx - 1]
                if op.name == "allo.transform_layout"
                and op.operands[0] == source
                and op.result == from_op.operands[0]
                and is_inverse_transform_layout(from_op, op)
            ]
            if len(to_candidates) != 1:
                break
            to_op = to_candidates[0]

            if _is_nested_in_operation(get_value_owner(source), loop):
                break

            source_uses_are_safe = True
            for use in source.uses:
                use_owner = _as_operation(use.owner)
                if not _is_nested_in_operation(use_owner, loop):
                    continue
                if (
                    use_owner.name == "memref.copy"
                    and use_owner.operands[0] == use_owner.operands[1]
                ):
                    continue
                if not (
                    (use_owner == to_op and use.operand_number == 0)
                    or (use_owner == copy_op and use.operand_number == 1)
                ):
                    source_uses_are_safe = False
                    break
            if not source_uses_are_safe:
                break

            from_uses = list(from_op.result.uses)
            if not (
                len(from_uses) == 1
                and _as_operation(from_uses[0].owner) == copy_op
                and from_uses[0].operand_number == 0
            ):
                break

            to_result_uses_are_safe = True
            for use in to_op.result.uses:
                use_owner = _as_operation(use.owner)
                direct_child = _get_direct_child(use_owner, loop)
                if (
                    direct_child is None
                    or direct_child not in operation_indices
                    or operation_indices[direct_child] > operation_indices[from_op]
                ):
                    to_result_uses_are_safe = False
                    break
            if not to_result_uses_are_safe:
                break

            pairs.append((to_op, from_op, copy_op))
            tail_idx -= 2

        return list(reversed(pairs)), trailing_noops

    def promote_from_loop(loop):
        pairs, trailing_noops = match_tail_pairs(loop)
        if not pairs:
            return

        for noop in trailing_noops:
            noop.erase()

        loop = _as_operation(loop)
        for to_op, _, _ in pairs:
            to_op.move_before(loop)

        insert_after = loop
        for _, from_op, copy_op in pairs:
            from_op.move_after(insert_after)
            copy_op.move_after(from_op)
            result_use = allo_d.get_last_use_in_function(copy_op.operands[1], function)
            if getattr(result_use, "name", None) == "allo.stream_put":
                result_use.operands[1] = from_op.result
                insert_after = from_op
                copy_op.erase()
            else:
                insert_after = copy_op

    def visit(op):
        op = _as_operation(op)
        for region in op.regions:
            for block in region.blocks:
                for inner_op in list(block.operations):
                    visit(inner_op)
        if op.name == "affine.for":
            promote_from_loop(op)

    visit(function)


def _promote_sliced_matmul_accumulators(module, function):
    """Keep zero-initialized, whole-plane matmul accumulators packed.

    Only private [planes, M, N] accumulators whose complete use graph is a
    zero fill, one matched slice update, and a full output view are eligible.
    A generated pointer-offset wrapper preserves the bare-pointer kernel ABI;
    passing a strided memref directly would lose its dynamic offset there.
    """

    def op_of(value):
        return _as_operation(value.owner)

    def users(value):
        return {_as_operation(use.owner) for use in value.uses}

    def ints(op, attr):
        return list(op.attributes[attr])

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    yield _as_operation(child)
                    yield from walk(child)

    changed = False
    for alloc in [
        op
        for op in walk(function)
        if op.name == "memref.alloc" and len(MemRefType(op.result.type).shape) == 3
    ]:
        typ = MemRefType(alloc.result.type)
        if len(typ.shape) != 3 or any(d <= 0 for d in typ.shape):
            continue
        if typ.layout.value != allo_ir.ir.AffineMap.get_identity(
            3, context=function.context
        ):
            continue
        planes, rows, cols = typ.shape
        if planes * rows * cols >= 2**31:
            continue
        acc_users = users(alloc.result)
        fills = [op for op in acc_users if op.name == "linalg.fill"]
        slices = [op for op in acc_users if op.name == "memref.subview"]
        views = [op for op in acc_users if op.name == "memref.reinterpret_cast"]
        if len(fills) != 1 or len(slices) not in (1, 2) or len(views) != 1:
            continue
        if acc_users != set(fills + slices + views):
            continue
        fill, view = fills[0], views[0]
        zero = op_of(fill.operands[0])
        if zero.name != "arith.constant" or str(zero.attributes["value"]) not in (
            "0 : i16",
            "0 : i32",
            "0 : i8",
            "0.000000e+00 : f32",
            "0.000000e+00 : bf16",
        ):
            continue
        if fill.parent != alloc.parent or view.parent != alloc.parent:
            continue
        output_type = MemRefType(view.result.type)
        if (
            list(output_type.shape) != [planes * rows, cols]
            or ints(view, "static_offsets") != [0]
            or ints(view, "static_sizes") != [planes * rows, cols]
            or ints(view, "static_strides") != [cols, 1]
        ):
            continue
        puts = list(users(view.result))
        if len(puts) != 1 or puts[0].name != "allo.stream_put":
            continue
        if puts[0].parent != view.parent:
            continue
        first = slices[0]
        if any(
            ints(s, "static_sizes") != [1, rows, cols]
            or ints(s, "static_strides") != [1, 1, 1]
            or ints(s, "static_offsets")[1:] != [0, 0]
            or len(s.operands) != 2
            or list(s.operands) != list(first.operands)
            or s.attributes["static_offsets"] != first.attributes["static_offsets"]
            or s.parent != first.parent
            for s in slices
        ):
            continue
        slice_users = set().union(*(users(s.result) for s in slices))
        reads = [
            op
            for op in slice_users
            if op.name == "memref.copy" and op.operands[0] in [s.result for s in slices]
        ]
        writes = [
            op
            for op in slice_users
            if op.name == "memref.copy" and op.operands[1] in [s.result for s in slices]
        ]
        if len(reads) != 1 or len(writes) != 1 or slice_users != set(reads + writes):
            continue
        read, write = reads[0], writes[0]
        scratch = read.operands[1]
        if op_of(scratch).name != "memref.alloc":
            continue
        to_ops = [op for op in users(scratch) if op.name == "allo.transform_layout"]
        if len(to_ops) != 1 or users(scratch) != {read, to_ops[0]}:
            continue
        to = to_ops[0]
        back = op_of(write.operands[0])
        if back.name != "allo.transform_layout" or not is_inverse_transform_layout(
            back, to
        ):
            continue
        if back.operands[0] != to.result or users(back.result) != {write}:
            continue
        calls = [op for op in users(to.result) if op.name == "func.call"]
        if len(calls) != 1 or users(to.result) != {back, calls[0]}:
            continue
        call = calls[0]
        name = call.attributes["callee"].value
        if (
            name not in module.injected_external_kernels
            or not _is_builtin_matmul(name)
            or len(call.operands) != 3
            or call.operands[2] != to.result
            or to.result in call.operands[:2]
        ):
            continue
        if any(op.parent != first.parent for op in (read, to, call, back, write)):
            continue
        # Require the exact blocked-matrix permutation, not merely a hint.
        sizes = ints(to, "sizes")
        if len(sizes) != 4:
            continue
        rm, cn, micro_m, micro_n = sizes
        if (
            rm * micro_m != rows
            or cn * micro_n != cols
            or ints(to, "offsets") != [0] * 4
            or ints(to, "strides") != [micro_m * cols, micro_n, cols, 1]
            or ints(back, "offsets") != [0] * 4
            or ints(back, "sizes") != [rm, micro_m, cn, micro_n]
            or ints(back, "strides") != [micro_m * cols, micro_n, micro_m * micro_n, 1]
        ):
            continue
        # Prove ordering and reject branches/escaping scopes. Only counted
        # loops may surround the update; initialization precedes all updates.
        scope = first
        while scope.parent != alloc.parent and scope.parent.name in (
            "affine.for",
            "scf.for",
        ):
            scope = scope.parent
        if scope.parent != alloc.parent:
            continue
        siblings = list(alloc.parent.regions[0].blocks[0].operations)
        siblings = [_as_operation(op) for op in siblings]
        if not (
            siblings.index(fill)
            < siblings.index(scope)
            < siblings.index(view)
            < siblings.index(puts[0])
        ):
            continue
        body = [
            _as_operation(op) for op in first.parent.regions[0].blocks[0].operations
        ]
        if not (
            body.index(read)
            < body.index(to)
            < body.index(call)
            < body.index(back)
            < body.index(write)
        ):
            continue
        with function.context, allo_ir.ir.Location.unknown():
            from ..._mlir.dialects import arith as arith_d
            from ..._mlir.ir import IntegerType
            from .utils import external_kernel_aie2c_type

            offset_name = f"{name}_slice_{planes}x{rows}x{cols}"
            zero_name = f"fill_zeros_{str(typ.element_type)}_{planes * rows}_{cols}_vector_planes_{planes}"
            ctype_a = external_kernel_aie2c_type[
                str(call.operands[0].type.element_type)
            ]
            ctype_b = external_kernel_aie2c_type[
                str(call.operands[1].type.element_type)
            ]
            ctype_c = external_kernel_aie2c_type[str(typ.element_type)]
            index_type = IntegerType.get_signless(32)
            node = module.virtual_computation_graph.nodes[function.name.value]
            for symbol, types, code, include, inputs, outputs in (
                (
                    offset_name,
                    [call.operands[0].type, call.operands[1].type, typ, index_type],
                    f"void {offset_name}({ctype_a} *a, {ctype_b} *b, {ctype_c} *c, int32_t slice) {{\n"
                    f"  {name}(a, b, c + slice * {rows * cols});\n}}\n",
                    module.include_src[name],
                    [0, 1, 3],
                    [2],
                ),
                (
                    zero_name,
                    [typ],
                    f"void {zero_name}({ctype_c} *c) {{\n"
                    f"  zero_vectorized<{ctype_c}, {planes * rows}, {cols}>(c);\n}}\n",
                    f'#include "{"aie2" if module.device == "npu1" else "aie2p"}/zero.cc"\n',
                    [],
                    [0],
                ),
            ):
                if symbol not in module.injected_external_kernels:
                    module.injected_external_kernels[symbol] = ExternalModuleBase(
                        symbol, inputs, outputs, code
                    )
                    module.include_src[symbol] = include
                    decl = allo_func_d.FuncOp(
                        symbol,
                        allo_func_d.FunctionType.get(types, []),
                        ip=InsertionPoint(function),
                    )
                    decl.attributes["sym_visibility"] = StringAttr.get("private")
                node.meta_data.used_external_kernel.add(symbol)
            slice_index = arith_d.IndexCastOp(
                index_type, first.operands[1], ip=InsertionPoint(call)
            )
            allo_func_d.CallOp(
                [],
                FlatSymbolRefAttr.get(offset_name),
                [call.operands[0], call.operands[1], alloc.result, slice_index.result],
                ip=InsertionPoint(call),
            )
            allo_func_d.CallOp(
                [],
                FlatSymbolRefAttr.get(zero_name),
                [alloc.result],
                ip=InsertionPoint(fill),
            )
            # Unpack exactly once, at the output boundary. The DMA pass can
            # fold this permutation into the output FIFO's stream layout.
            unpack = allo_d.transform_layout(
                view.result.type,
                view.result,
                [0] * 4,
                [planes * rm, micro_m, cn, micro_n],
                [micro_m * cols, micro_n, micro_m * micro_n, 1],
                ip=InsertionPoint(puts[0]),
            )
            puts[0].operands[1] = unpack
            for op in (write, back, call, to, read, fill):
                op.erase()
            for s in slices:
                s.erase()
            op_of(scratch).erase()
            changed = True
    return changed


class AIE_MLIRModule:
    # ############################################################
    # Construction
    # ############################################################
    def __init__(
        self,
        module: allo_ir.ir.Module,
        top_func_name: str,
        func_args: dict,
        project_dir: str,
        stream_info: dict,
        stream_types_dict: dict[str, Type],
        ext_libs: list = None,
        func_instances: dict = None,
        extra_stream_info: dict = None,
    ):
        """
        Note: the module is data-driven,
            we need to carefully manage data transfer between 'functions' to avoid deadlocks.
            For example, launching the kernels in topological order.
        """
        # module metadata
        self.trace_size = 0
        self.project_dir: str = project_dir
        self.allo_module: allo_ir.ir.Module = module
        self.top_func_name: str = top_func_name
        self.func_instances = func_instances

        self.external_kernel_lib: dict[str, ExternalModule] = {}
        for ext_kernel in ext_libs:
            if isinstance(ext_kernel, ExternalModule):
                self.external_kernel_lib[ext_kernel.top] = ext_kernel

        self.func_args: dict[str, list[Argument | list[Argument]]] = {}
        self.streams: dict[str, Stream] = {}
        self.stream_info: dict[str, dict[str, bool]] = {}
        self._init_func_args(func_args)
        self.computation_is_dag = self._init_streams(
            stream_info, stream_types_dict, extra_stream_info
        )

        # index in top function argument list -> DTensor
        self.global_tensors: dict[int, DTensor] = None
        self.module_runtime_args: list[RuntimeArgs] = None
        # function name -> (argument index -> (argument, is_input))
        self.core_func_args: dict[
            str, dict[int, tuple[Argument | list[Argument], bool]]
        ] = None

        self.aie_module: aie_ir.Module = None

    def _init_func_args(self, func_args: dict):
        tmp_map: dict = {}
        for func_name, args in func_args.items():
            self.func_args[func_name] = []
            for arg in args:
                if isinstance(arg, list):
                    stream_args = []
                    for stream_ in arg:
                        assert isinstance(stream_, str)
                        if stream_ in tmp_map:
                            stream_args.append(tmp_map[stream_])
                        else:
                            stream = Stream(stream_)
                            self.streams[stream_] = stream
                            argument = Argument(None, stream)
                            stream_args.append(argument)
                            tmp_map[stream_] = argument
                    self.func_args[func_name].append(stream_args)
                elif arg in tmp_map:
                    self.func_args[func_name].append(tmp_map[arg])
                elif isinstance(arg, DTensor):
                    stream = Stream(arg.name)
                    arg.type_as_param = arg.get_local_shape()
                    stream.is_tensor = len(arg.type_as_param) > 0
                    with self.allo_module.context, allo_ir.ir.Location.unknown():
                        stream.allo_element_type = MemRefType.get(
                            arg.type_as_param, arg.dtype.build()
                        )
                    argument = Argument(arg, stream)
                    self.func_args[func_name].append(argument)
                    tmp_map[arg] = argument
                elif isinstance(arg, str):
                    stream = Stream(arg)
                    self.streams[arg] = stream
                    argument = Argument(None, stream)
                    self.func_args[func_name].append(argument)
                    tmp_map[arg] = argument
                else:
                    raise ValueError(f"Unresolved function argument {arg}")

    def _init_streams(
        self,
        stream_info: dict,
        stream_types_dict: dict[str, Type],
        extra_stream_info: dict,
    ):
        """
        Collect allo.stream information for each function.
        """
        for func_name, info_list in stream_info.items():
            self.stream_info[func_name] = {}
            for name, io in info_list:
                self.streams[name].set_element_type(
                    str(stream_types_dict[name]), self.allo_module.context
                )
                if io == "in":
                    self.streams[name].dst = func_name
                    self.stream_info[func_name][name] = True
                    if (
                        extra_stream_info is not None
                        and name in extra_stream_info[func_name]
                    ):
                        self.streams[name].dst_related_iter_info = extra_stream_info[
                            func_name
                        ][name]
                else:
                    self.streams[name].src = func_name
                    self.stream_info[func_name][name] = False
                    if (
                        extra_stream_info is not None
                        and name in extra_stream_info[func_name]
                    ):
                        self.streams[name].src_related_iter_info = extra_stream_info[
                            func_name
                        ][name]
        edge_map = {src: set() for src in stream_info.keys()}
        for stream in self.streams.values():
            edge_map[stream.src].add(stream.dst)
        visited = set()
        rec_stack = set()

        def dfs(node):
            if node in rec_stack:
                return False  # found a cycle
            if node in visited:
                return True

            visited.add(node)
            rec_stack.add(node)

            for neighbor in edge_map.get(node, set()):
                if not dfs(neighbor):
                    return False

            rec_stack.remove(node)
            return True

        return all(dfs(node) for node in edge_map.keys())

    # ############################################################
    # Build
    # ############################################################
    def init_virtual_graph(self, used_external_kernels: dict[str, set[str]]):
        assert (
            self.core_func_args is not None and self.global_tensors is not None
        ), "Analysis of kernel parameters should be done before initializing virtual graph"
        self.virtual_computation_graph: ComputationGraph = ComputationGraph(
            self.allo_module,
            self.top_func_name,
            self.streams,
            self.core_func_args,
            used_external_kernels,
            self.func_instances,
        )

    def analyze_kernel_parameters(
        self,
        df_kernels: list[allo_func_d.FuncOp],
    ):
        """
        Analyze the parameters of each df.kernel.

        Collected information:
            - self.core_func_args: function name -> (argument index -> (argument, is_input))
            - self.global_tensors: global argument index -> DTensor
        """
        top_func_args = [arg.dtensor.name for arg in self.func_args[self.top_func_name]]
        tag_to_read_write_pattern: dict[str, tuple[list, list]] = {}
        # init
        self.core_func_args = {}
        self.global_tensors = {}
        # analyze
        tag_to_func: dict[str, allo_func_d.FuncOp] = {}
        for func in df_kernels:
            tag = func.attributes["tag"].value
            if tag not in tag_to_func:
                tag_to_func[tag] = func
                in_idx_list, out_idx_list = [], []
                for arg in func.arguments:
                    is_put = None
                    for use in arg.uses:
                        use_op = getattr(use.owner, "name", None)
                        if use_op == "allo.stream_put":
                            assert is_put is None or is_put
                            is_put = True
                        else:
                            assert use_op == "allo.stream_get"
                            assert is_put is None or not is_put
                            is_put = False
                    if is_put is None:
                        continue
                    if is_put:
                        out_idx_list.append(arg.arg_number)
                    else:
                        in_idx_list.append(arg.arg_number)
                tag_to_read_write_pattern[tag] = (in_idx_list, out_idx_list)

        for orig_name, kernel_instance_info in self.func_instances.items():
            for dim, predicate_tag in kernel_instance_info.items():
                kernel_name = construct_kernel_name(orig_name, dim)
                kernel = tag_to_func[predicate_tag]
                self.core_func_args[kernel_name] = {}
                in_idx_list, out_idx_list = tag_to_read_write_pattern[predicate_tag]
                for io_idx_list, io_type in (
                    (in_idx_list, "in"),
                    (out_idx_list, "out"),
                ):
                    for io_idx in io_idx_list:
                        argument = self.func_args[kernel_name][io_idx]
                        self.core_func_args[kernel_name][io_idx] = (
                            argument,
                            io_type == "in",
                        )
                        if (
                            isinstance(argument, Argument)
                            and not argument.dtensor is None
                        ):
                            argument.dtensor.set_access_pattern()
                            global_idx = top_func_args.index(argument.dtensor.top_name)
                            argument.dtensor.set_global_info(
                                global_idx, io_type == "in"
                            )
                            self.global_tensors[global_idx] = argument.dtensor
                # streams
                for i, _ in enumerate(kernel.arguments):
                    func_arg = self.func_args[kernel_name][i]
                    if i in self.core_func_args[kernel_name]:
                        continue
                    # unused Dtensor
                    if isinstance(func_arg, Argument) and func_arg.stream is None:
                        continue
                    if isinstance(func_arg, Argument):
                        sample_stream = func_arg.stream
                    else:
                        assert len(func_arg) > 0 and func_arg[0].stream is not None
                        sample_stream = func_arg[0].stream
                    if sample_stream.name in self.stream_info[kernel_name]:
                        self.core_func_args[kernel_name][i] = (
                            func_arg,
                            self.stream_info[kernel_name][sample_stream.name],
                        )

    def allo_opt(self):
        """
        run optimized passes on allo mlir
        """

        def vectorize_matmul(function: allo_func_d.FuncOp):
            """
            Using vectorized matmul to replace scalar version.
            Layout transform is required before and after using vectorized matmul.

            * All external kernels are from https://github.com/Xilinx/mlir-aie/tree/v1.0/aie_kernels.
            """
            matmul_ops: list[allo_func_d.CallOp] = collect_lib_func_call(
                function, "matmul_scalar"
            )
            for call_matmul_op in matmul_ops:
                input_a = call_matmul_op.operands[0]
                input_b = call_matmul_op.operands[1]
                output = call_matmul_op.operands[-1]
                M, K = MemRefType(input_a.type).shape
                _, N = MemRefType(input_b.type).shape
                dtype_a = str(input_a.type.element_type)
                dtype_b = str(input_b.type.element_type)
                out_dtype = str(output.type.element_type)
                if dtype_a == dtype_b:
                    assert (dtype_a, out_dtype) in matmul_external_kernel_config_map
                    matmul_configs = matmul_external_kernel_config_map[
                        (dtype_a, out_dtype)
                    ]
                    if self.device == "npu1":
                        m, k, n = matmul_configs["aie2"]
                    else:
                        m, k, n = matmul_configs["aie2p"]
                elif self.device == "npu1":
                    if dtype_a == dtype_b:
                        m, k, n = matmul_external_kernel_config_map[
                            (dtype_a, out_dtype)
                        ]["aie2"]
                    else:
                        m, k, n = matmul_external_kernel_config_map[
                            (dtype_a, dtype_b, out_dtype)
                        ]["aie2"]
                else:
                    continue
                # validity test
                if self.device == "npu2" or dtype_a == "i16":
                    factor_m, factor_n = m * 2, n * 2
                elif dtype_a == "bf16":
                    factor_m, factor_n = m * 4, n * 4
                else:
                    factor_m, factor_n = m * 4, n * 2
                valid = K % k == 0 and M % factor_m == 0 and N % factor_n == 0
                if not valid:
                    warn = AlloWarning(
                        "Detected a vectorized matmul kernel, but it cannot be used because the tiling constraints are not met."
                        f"Ensure that tile_K % {k} == 0, tile_M % {factor_m} == 0, and tile_N % {factor_n} == 0 to enable this optimization."
                    )
                    if (
                        dtype_a == "i4"
                        or dtype_b == "i4"
                        or os.environ.get("ALLO_EXTERNAL_KERNEL_DIR") is None
                    ):
                        # - we do not provide scalar kernel for i4
                        # - mlir-aie external kernels instantiate both vector and scalar versions and fail compilation when tiling constraints are not satisfied. Our library is safe to use.
                        raise warn
                    warn.warn()
                    continue
                with function.context, allo_ir.ir.Location.unknown():
                    new_input_0 = allo_d.transform_layout(
                        input_a.type,
                        input_a,
                        [0, 0, 0, 0],
                        [M // m, K // k, m, k],
                        [m * K, k, K, 1],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    new_input_0.owner.attributes["layout_hint"] = StringAttr.get(
                        f"A_to_{M}x{N}x{K}_{m}x{n}x{k}"
                    )
                    new_input_1 = allo_d.transform_layout(
                        input_b.type,
                        input_b,
                        [0, 0, 0, 0],
                        [K // k, N // n, k, n],
                        [N * k, n, N, 1],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    new_input_1.owner.attributes["layout_hint"] = StringAttr.get(
                        f"B_to_{M}x{N}x{K}_{m}x{n}x{k}"
                    )
                    new_output = allo_d.transform_layout(
                        output.type,
                        output,
                        [0, 0, 0, 0],
                        [M // m, N // n, m, n],
                        [m * N, n, N, 1],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    new_output.owner.attributes["layout_hint"] = StringAttr.get(
                        f"C_to_{M}x{N}x{K}_{m}x{n}x{k}"
                    )
                    vectorized_kernel_name = call_matmul_op.attributes[
                        "lib"
                    ].value.replace("matmul_scalar_", "matmul_")
                    allo_func_d.CallOp(
                        [],
                        FlatSymbolRefAttr.get(vectorized_kernel_name),
                        [new_input_0, new_input_1, new_output],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    matmul_output = allo_d.transform_layout(
                        new_output.type,
                        new_output,
                        [0, 0, 0, 0],
                        [M // m, m, N // n, n],
                        [m * N, n, m * n, 1],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    matmul_output.owner.attributes["layout_hint"] = StringAttr.get(
                        f"C_from_{M}x{N}x{K}_{m}x{n}x{k}"
                    )
                    allo_memref_d.copy(
                        matmul_output, output, ip=InsertionPoint(call_matmul_op)
                    )
                    self.virtual_computation_graph.nodes[
                        function.attributes["sym_name"].value
                    ].meta_data.used_external_kernel.add(vectorized_kernel_name)

                    if vectorized_kernel_name not in self.injected_external_kernels:
                        scalar_kernel: ExternalModuleBase = (
                            self.injected_external_kernels[
                                call_matmul_op.attributes["lib"].value
                            ]
                        )
                        kernel_code = scalar_kernel.kernel_code.replace(
                            "matmul_scalar_", "matmul_vectorized_"
                        )
                        self.injected_external_kernels[vectorized_kernel_name] = (
                            ExternalModuleBase(
                                vectorized_kernel_name,
                                scalar_kernel.input_idx,
                                scalar_kernel.output_idx,
                                kernel_code,
                                scalar_kernel.kernel_header,
                            )
                        )
                        self.include_src[vectorized_kernel_name] = self.include_src[
                            scalar_kernel.top
                        ]
                        operand_types = [x.type for x in call_matmul_op.operands]
                        func_type = allo_func_d.FunctionType.get(
                            operand_types,
                            [],
                        )
                        vectorized_kernel = allo_func_d.FuncOp(
                            vectorized_kernel_name,
                            func_type,
                            ip=InsertionPoint(function),
                        )
                        vectorized_kernel.attributes["sym_visibility"] = StringAttr.get(
                            "private"
                        )
                    call_matmul_op.erase()

        def optimize_layout_transformation(function: allo_func_d.FuncOp):
            """
            Optimize layout transformation operations
                - some can be 'push out of the function' and done at transfer time (e.g. with dma)
                - some contiguous inverse transformation can be safely removed.
                - loop-carried inverse transformations can be promoted out of affine loops.
            """
            node = self.virtual_computation_graph.nodes[
                func.attributes["sym_name"].value
            ]
            dead_ops = []
            # no need to transform if the result is unchanged
            excuse_operands = set()

            def optimize_layout_transformation_recursive(op):
                # op.operands[0] is whole zero
                if (
                    "lib" in op.attributes
                    and "fill_zeros" in op.attributes["lib"].value
                ):
                    excuse_operands.add(op.operands[0])
                    return
                if getattr(op, "name", None) == "allo.transform_layout":
                    if op.operands[0] in excuse_operands:
                        op.result.replace_all_uses_with(op.operands[0])
                        excuse_operands.remove(op.operands[0])
                        dead_ops.append(op)
                        return
                    if "layout_hint" in op.attributes:
                        layout_hint = op.attributes["layout_hint"].value
                        parts = re.findall(r"[^_]+", layout_hint)
                        if len(parts) == 4 and parts[1] == "from":
                            next_use_op = allo_d.get_next_use_in_function(
                                op.result, op, function
                            )
                            if next_use_op is not None:
                                # transform_layout
                                if "layout_hint" in next_use_op.attributes:
                                    if is_inverse_transform_layout(op, next_use_op):
                                        next_use_op.result.replace_all_uses_with(
                                            op.operands[0]
                                        )
                                        dead_ops.append(op)
                                        dead_ops.append(next_use_op)
                                # memcpy to another memref
                                elif (
                                    getattr(next_use_op, "name", None) == "memref.copy"
                                    and op.result == next_use_op.operands[0]
                                ):
                                    memcpy_op = next_use_op
                                    next_use_op = allo_d.get_next_use_in_function(
                                        memcpy_op.operands[1], next_use_op, function
                                    )
                                    # cpy result used in transform_layout
                                    if (
                                        next_use_op is not None
                                        and "layout_hint" in next_use_op.attributes
                                    ):
                                        if is_inverse_transform_layout(op, next_use_op):
                                            if (
                                                getattr(
                                                    allo_d.get_next_use_in_function(
                                                        memcpy_op.operands[1],
                                                        next_use_op,
                                                        function,
                                                    ),
                                                    "name",
                                                    None,
                                                )
                                                == "memref.copy"
                                            ):
                                                next_use_op.result.replace_all_uses_with(
                                                    op.operands[0]
                                                )
                                                dead_ops.append(next_use_op)
                                                dead_ops.append(memcpy_op)
                                                dead_ops.append(op)
                                                return

                for region in op.regions:
                    for block in region.blocks:
                        # fixme: using 'stack' for nested blocks can be better
                        excuse_operands.clear()
                        for inner_op in block.operations:
                            optimize_layout_transformation_recursive(inner_op)

            def transform_with_dma():
                arg_info = self.core_func_args[function.name.value]
                for arg in func.arguments:
                    if arg.arg_number not in arg_info:
                        continue
                    # input: transform to the same layout before every 'use'
                    sample_stream = (
                        arg_info[arg.arg_number][0].stream
                        if isinstance(arg_info[arg.arg_number][0], Argument)
                        else arg_info[arg.arg_number][0][0].stream
                    )
                    if arg_info[arg.arg_number][1]:
                        var = arg
                        if (
                            sample_stream is not None
                            and len(list(arg.uses)) == 1  # allo.stream_get
                        ):
                            var = list(arg.uses)[0].owner.result
                        op = None
                        for use in var.uses:
                            if (
                                getattr(use.owner, "name", None)
                                == "allo.transform_layout"
                            ):
                                if op is not None:
                                    if (
                                        op.attributes["offsets"]
                                        != use.owner.attributes["offsets"]
                                        or op.attributes["sizes"]
                                        != use.owner.attributes["sizes"]
                                        or op.attributes["strides"]
                                        != use.owner.attributes["strides"]
                                    ):
                                        op = None
                                        break
                                op = use.owner
                            else:
                                op = None
                                break
                        if op is not None:
                            node.interface_layout[arg.arg_number] = (
                                list(op.attributes["offsets"]),
                                list(op.attributes["sizes"]),
                                list(op.attributes["strides"]),
                            )
                            if sample_stream is not None:
                                sample_stream.dst_layout_transform = (
                                    list(op.attributes["offsets"]),
                                    list(op.attributes["sizes"]),
                                    list(op.attributes["strides"]),
                                    (
                                        None
                                        if "layout_hint" not in op.attributes
                                        else op.attributes["layout_hint"].value
                                    ),
                                )
                            op.result.replace_all_uses_with(var)
                            op.erase()
                    # output
                    else:
                        op = allo_d.get_last_use_in_function(arg, function)
                        is_dtensor = sample_stream is None
                        operand_idx = 1
                        if (
                            getattr(op, "name", None) == "allo.stream_put"
                            and op.operands[operand_idx] not in func.arguments
                            and getattr(op.operands[operand_idx].owner, "name", None)
                            == "allo.transform_layout"
                        ):
                            transform_layout_op = op.operands[operand_idx].owner
                            # the layout transfer before copy can be forwarded to transfer after copy
                            forward_layout_transform_flag = (
                                transform_layout_op.operands[0] not in func.arguments
                                and allo_d.get_last_use_in_function(
                                    transform_layout_op.operands[0],
                                    function,
                                )
                                == transform_layout_op
                            )
                            opt_flag = False
                            if is_dtensor:
                                # typical transform before transfer pattern
                                if transform_layout_op.operands[0] == arg:
                                    opt_flag = True
                                else:
                                    opt_flag = forward_layout_transform_flag
                            else:
                                opt_flag = forward_layout_transform_flag
                            if opt_flag:
                                node.interface_layout[arg.arg_number] = (
                                    list(transform_layout_op.attributes["offsets"]),
                                    list(transform_layout_op.attributes["sizes"]),
                                    list(transform_layout_op.attributes["strides"]),
                                )
                                if not is_dtensor:
                                    sample_stream.src_layout_transform = (
                                        list(transform_layout_op.attributes["offsets"]),
                                        list(transform_layout_op.attributes["sizes"]),
                                        list(transform_layout_op.attributes["strides"]),
                                        (
                                            None
                                            if "layout_hint"
                                            not in transform_layout_op.attributes
                                            else transform_layout_op.attributes[
                                                "layout_hint"
                                            ].value
                                        ),
                                    )
                                    op.operands[1] = transform_layout_op.operands[0]
                                if not forward_layout_transform_flag:
                                    # if is forward, cannot erase the op
                                    op.erase()
                                transform_layout_op.result.replace_all_uses_with(
                                    transform_layout_op.operands[0]
                                )
                                transform_layout_op.erase()

            _promote_loop_carried_layout_transformations(function)
            optimize_layout_transformation_recursive(function)
            for op in dead_ops:
                op.erase()
            allo_d.copy_on_write_on_function(function)
            transform_with_dma()

        for func in self.allo_module.body.operations:
            if isinstance(func, allo_func_d.FuncOp) and "df.kernel" in func.attributes:
                simplify_matmul_accumulate(func)
                allo_d.copy_on_write_on_function(func)
                vectorize_matmul(func)
                optimize_layout_transformation(func)

        pipeline = "builtin.module(canonicalize)"
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)

        for func in list(self.allo_module.body.operations):
            if isinstance(func, allo_func_d.FuncOp) and "df.kernel" in func.attributes:
                if _promote_sliced_matmul_accumulators(self, func):
                    optimize_layout_transformation(func)
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)

        # record optimized allo mlir
        with open(
            os.path.join(self.project_dir, "original_opt.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))

    def build(
        self,
        device_type="npu1_4col",
        mapping_primitives: list[tuple[str, list]] = None,
        profile: bool = False,
        warmup: int = 20,
        num_iters: int = 100,
        trace: list[tuple[str, tuple[int, ...]]] = None,
        trace_size: int = 4096,
        output_fifo_depth: int = 2,
        fifo_lowering: str = "auto",
    ):
        if type(output_fifo_depth) is not int or output_fifo_depth not in (1, 2):
            raise ValueError("output_fifo_depth must be 1 or 2")
        if fifo_lowering not in ("auto", "static", "dynamic"):
            raise ValueError("fifo_lowering must be auto, static, or dynamic")
        # virtual mapping can only be applied to DAG
        if not self.computation_is_dag:
            if mapping_primitives is not None and len(mapping_primitives) > 0:
                raise ValueError(
                    "The input computation graph is not a DAG. Do not support virtual mapping now."
                )

        if "npu1" in device_type:
            self.device = "npu1"
        elif "npu2" in device_type:
            self.device = "npu2"
        else:
            raise ValueError("Unsupported device type.")
        self.profile = profile
        self.warmup = warmup
        self.num_iters = num_iters
        if trace is not None:
            self.trace_size = trace_size
        build_dir = os.path.join(self.project_dir, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "raw.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))
        # inject external kernels
        # (inject before virtual mapping since using external kernel may require layout transformation when transferring data)
        used_external_kernels, self.injected_external_kernels, self.include_src = (
            inject_external_kernels(
                self.allo_module,
                self.top_func_name,
                self.external_kernel_lib,
                "aie2" if self.device == "npu1" else "aie2p",
            )
        )
        self.analyze_kernel_parameters(get_df_kernels(self.allo_module))
        # ------------------------- virtual mapping -------------------------
        self.init_virtual_graph(used_external_kernels)
        if os.getenv("DEBUG") == "1":
            self.virtual_computation_graph.dump(self.project_dir)

        if mapping_primitives is not None:
            for mapping in mapping_primitives:
                primitive = mapping[0]
                arg_list = mapping[1]
                if primitive == "chain":
                    assert len(arg_list) == 2
                    self.virtual_computation_graph.chain(arg_list[0], arg_list[1])
                if primitive == "bundle":
                    if isinstance(arg_list[0], str):
                        arg_list = [(x,) for x in arg_list]
                    self.virtual_computation_graph.bundle_multi(arg_list)
            if os.getenv("DEBUG") == "1":
                self.virtual_computation_graph.dump(self.project_dir, "after_mapping")

        self.virtual_computation_graph.refactor()

        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "original.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))

        # mapping guard
        logical_node_num = len(self.virtual_computation_graph.nodes)
        assert device_type in device_config_map, "Unsupported device type"
        physical_node_num = np.prod(device_config_map[device_type]["mesh"])
        assert (
            logical_node_num <= physical_node_num
        ), f"Unresolvable mapping from {logical_node_num} logical nodes to {physical_node_num} physical nodes"

        # ------------------------- code optimization -------------------------
        self.allo_opt()

        passes = [
            "func.func(convert-linalg-to-affine-loops),lower-transform-layout-ops",
        ]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)
        # ------------------------- mlir-aie code generation -------------------------
        with allo_ir.ir.Context() as ctx, allo_ir.ir.Location.unknown():
            allo_d.register_dialect(ctx)
            pattern = re.compile(r"memref<([\dx]+)xi4>")
            module_str = str(self.allo_module)

            def repl(match):
                dims_str = match.group(1)
                dims = [int(d) for d in dims_str.split("x")]
                dims[-1] //= 2
                new_dims_str = "x".join(str(d) for d in dims)
                return f"memref<{new_dims_str}xi8>"

            module_str = pattern.sub(repl, module_str)
            self.allo_module = allo_ir.ir.Module.parse(module_str)

        top_func, core_funcs, external_funcs = classify_aie_functions(
            self.allo_module, self.top_func_name
        )

        external_cc_list: list[set[str]] = []
        linked_external_cc: dict[str, int] = {}

        for func in core_funcs:
            func_name = func.attributes["sym_name"].value
            used_external_kernel = self.virtual_computation_graph.nodes[
                func_name
            ].meta_data.used_external_kernel
            try:
                linked_external_cc[func_name] = external_cc_list.index(
                    used_external_kernel
                )
            except ValueError:
                linked_external_cc[func_name] = len(external_cc_list)
                external_cc_list.append(used_external_kernel)

        code_generator = CodeGenerator(
            device_type,
            self.global_tensors,
            top_func,
            self.core_func_args,
            self.streams,
            self.virtual_computation_graph,
            output_fifo_depth=output_fifo_depth,
            fifo_lowering=fifo_lowering,
        )
        (
            self.aie_module,
            self.module_runtime_args,
        ) = code_generator.aie_codegen(
            core_funcs,
            external_funcs,
            linked_external_cc,
            trace,
            trace_size,
        )

        # TODO: opt passes on aie-mlir
        passes = [
            "func.func(affine-loop-unroll), canonicalize",
        ]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.aie_module.context:
            aie_pass_manager.PassManager.parse(pipeline).run(self.aie_module.operation)

        # ------------------------- build project -------------------------
        self.post_codegen_build(external_cc_list)
        return self

    def post_codegen_build(self, external_cc_list: list[set[str]]):
        with open(
            os.path.join(self.project_dir, "top.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.aie_module))
        if len(self.injected_external_kernels) > 0:
            paths = set()
            # user defined external kernels
            for ext_module in self.external_kernel_lib.values():
                paths.add((ext_module.impl_path, ext_module.filename))
            for src_path, dst_file in paths:
                target_path = os.path.join(self.project_dir, dst_file)
                if os.path.exists(target_path) and os.path.samefile(
                    src_path, target_path
                ):
                    continue
                shutil.copy(src_path, target_path)
            for idx, kernel_info in enumerate(external_cc_list):
                if len(kernel_info) == 0:
                    continue
                injected_kernels_ = {
                    k: v
                    for k, v in self.injected_external_kernels.items()
                    if k in kernel_info
                }
                include_src_ = set()
                for kernel in kernel_info:
                    include_src_.add(self.include_src[kernel])
                kernel_code = codegen_external_kernels(
                    injected_kernels_,
                    include_src_,
                    "aie2" if self.device == "npu1" else "aie2p",
                )
                with open(
                    os.path.join(self.project_dir, f"external{idx}.cc"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(kernel_code)
                cmd = f"cd {self.project_dir} && $PEANO_INSTALL_DIR/bin/clang++ -O2 -v -std=c++20 --target=aie2{"p" if self.device == "npu2" else ""}-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -DNDEBUG -I $MLIR_AIE_INSTALL_DIR/include -I $MLIR_AIE_EXTERNAL_KERNEL_DIR/ -I. -c external{idx}.cc -o external{idx}.o"
                with subprocess.Popen(cmd, shell=True) as process:
                    process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to compile external kernels.")
        # build mlir-aie
        cmd = f"cd {self.project_dir} && aiecc.py --alloc-scheme=basic-sequential --aie-generate-xclbin --no-compile-host --xclbin-name=build/final.xclbin --no-xchesscc --no-xbridge --peano ${{PEANO_INSTALL_DIR}} --aie-generate-npu-insts --npu-insts-name=insts.txt top.mlir"
        with subprocess.Popen(cmd, shell=True) as process:
            process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                "Failed to compile the MLIR-AIE code.\n"
                "Possible causes include:\n"
                "\n"
                "  \033[93m1. Program memory overflow:\033[0m\n"
                "     If you see errors like \033[96m[AIE ERROR] _XAie_LoadProgMemSection():231: Overflow of program memory\033[0m\n"
                "     it means the generated program is too large for the AIE program memory.\n"
                "     Consider reducing program size or adjusting virtual mapping primitives (e.g., reducing excessive chaining).\n"
                "\n"
                "  \033[93m2. Data memory allocation failure:\033[0m\n"
                "     If you see errors like \033[96merror: \"-\":13:17: 'aie.tile' op allocated buffers exceeded available memory\033[0m\n"
                "     it means the total buffer allocation requested in AIE data memory exceeds its capacity.\n"
                "     Consider adjusting the tiling strategy or using smaller tile sizes.\n"
            )
        # generate host code
        path = os.path.dirname(__file__)
        path = os.path.join(path, "../../harness/aie")
        os.system(f"cp -r {path}/* {self.project_dir}")
        if self.module_runtime_args is None:
            self.module_runtime_args = []
            total_tensors = len(self.global_tensors)
            for idx in range(total_tensors):
                dtensor = self.global_tensors[idx]
                runtime_arg = RuntimeArgs(dtensor.dtype, dtensor.is_input)
                runtime_arg.global_tensors.append(idx)
                runtime_arg.current_size += np.prod(dtensor.shape)
                self.module_runtime_args.append(runtime_arg)
        host_code = codegen_host(self.global_tensors, self.module_runtime_args)
        with open(
            os.path.join(self.project_dir, "test.cpp"), "w", encoding="utf-8"
        ) as f:
            f.write(host_code)
        cmd = f"cd {self.project_dir}/build && cmake .. -DTARGET_NAME=top -DMLIR_AIE_DIR=$RUNTIME_LIB_DIR/.. && cmake --build . --config Release"
        with subprocess.Popen(cmd, shell=True) as process:
            process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to build AIE project.")

    def __call__(self, *args):
        for idx, dtensor in self.global_tensors.items():
            if dtensor.is_input:
                arg = args[idx]
                if str(dtensor.dtype) == "i4":
                    arg = pack_int4(arg)
                with open(
                    os.path.join(self.project_dir, f"input{idx}.data"), "wb"
                ) as f:
                    f.write(arg.tobytes())

        cmd = f"cd {self.project_dir} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE --trace_sz {self.trace_size} {f'-p true --warmup {self.warmup} --test_iter {self.num_iters}' if self.profile else ''}"
        with subprocess.Popen(cmd, shell=True) as process:
            process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to execute AIE code.")
        for idx, dtensor in self.global_tensors.items():
            if not dtensor.is_input:
                result = read_tensor_from_file(
                    dtensor.dtype,
                    args[idx].shape,
                    f"{self.project_dir}/output{idx}.data",
                )
                args[idx][:] = result


def _call_prj(
    project: str,
    dtype_list: list,
    trace_size: int,
    input_idx: list[int],
    output_idx: list[int],
    *args,
):
    """
    This function allows you to manually adjust files under the project
    directory (e.g., top.prj) after building with AIE_MLIRModule, such as
    `top.mlir`, external kernel functions, or `test.cpp`, and then
    recompile and rerun.

    Note (Shihan): currently intended for internal debugging use only.
    """
    # generate insts.txt
    cmd = f"cd {project} && aiecc.py --alloc-scheme=basic-sequential --aie-generate-xclbin --no-compile-host --xclbin-name=build/final.xclbin --no-xchesscc --no-xbridge --peano ${{PEANO_INSTALL_DIR}} --aie-generate-npu-insts --npu-insts-name=insts.txt top.mlir"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to compile the MLIR-AIE code")
    cmd = f"cd {project}/build && cmake .. -DTARGET_NAME=top -DMLIR_AIE_DIR=$RUNTIME_LIB_DIR/.. && cmake --build . --config Release"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to build AIE project.")
    # suppose the last argument is output
    for idx in input_idx:
        arg = args[idx]
        if str(dtype_list[idx]) == "i4":
            arg = pack_int4(arg)
        with open(os.path.join(project, f"input{idx}.data"), "wb") as f:
            f.write(arg.tobytes())
    if trace_size > 0:
        cmd = f"cd {project} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE --trace_sz {trace_size}"
    else:
        cmd = f"cd {project} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE -p false --warmup 20 --test_iter 100"
    with subprocess.Popen(cmd, shell=True) as process:
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("Failed to execute AIE code.")
    for idx in output_idx:
        result = read_tensor_from_file(
            dtype_list[idx],
            args[idx].shape,
            f"{project}/output{idx}.data",
        )
        args[idx][:] = result
