<!--
Copyright Allo authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Unified Spatial and Temporal Sharding for AIE Dataflow

> **Status: Partially implemented.** Typed axes, tensor ports, ordered shard
> products, and default temporal traversal are implemented. The runnable
> [rho=2 ATB GEMM](test_temporal_atb_gemm.py) combines spatial M/N sharding with
> temporal K/rho sharding. The interface below also describes future features:
> axis-object `meta_for`, `get_axis_id`, traversal overrides, and static port
> consumption checks are not yet implemented.
>
> This document specifies the broader dataflow programming interface. The
> [host-driven temporal-reduction test](test_gemm_temporal_reduction.py) is the
> current baseline and prior motivation, not the final interface.

## Runnable ATB GEMM

In the AIE Docker environment, after `source activate allo` and installing the
checkout with `python3 -m pip install --no-build-isolation -e .`:

```bash
# Default: M=N=K=128, spatial mesh 4x4, K tile 32, rho=2.
python3 tests/dataflow/aie/test_temporal_atb_gemm.py

# Larger square and rectangular problems.
python3 tests/dataflow/aie/test_temporal_atb_gemm.py --m 256 --n 256 --k 256
python3 tests/dataflow/aie/test_temporal_atb_gemm.py --m 128 --n 256 --k 128 --pm 2

# 1024-square: automatically bundle a logical 16x16 mesh onto 16 cores.
python3 tests/dataflow/aie/test_temporal_atb_gemm.py --m 1024 --n 1024 --k 1024

# Optional 64-wide K tiles: use a finer M partition to fit local buffers.
python3 tests/dataflow/aie/test_temporal_atb_gemm.py --m 1024 --n 1024 --k 1024 --pm 32 --pn 16 --pk 16

# Compiler checks without device execution; full bitstream build without execution.
python3 tests/dataflow/aie/test_temporal_atb_gemm.py --check-only
python3 tests/dataflow/aie/test_temporal_atb_gemm.py --build-only
```

The example uses `int16` inputs and accumulation, checking small signed inputs
against an `int32` NumPy reference. Each hardware invocation checks two random
inputs and a zero-input case with the same compiled module. Large values can
overflow `int16`; this version does not provide wide accumulation.

The mapping is `[Sm, Sn, Tk, Tr]` and the layouts are:

```python
LyA = [S(0) * S(3), S(2)]
LyB = [S(2), S(1)]
LyC = [S(0), S(1)]
```

With `Mt=M/Pm`, `Nt=N/Pn`, `Kt=K/Pk`, and `At=Mt/2`:

| Storage/transfer | A | B | C |
| --- | --- | --- | --- |
| L1 | `At x Kt` | `Kt x Nt` | `Mt x Nt` retained across K |
| L2 row extent | `Mt` (two adjacent A transfers grouped) | `Kt` | `Mt` |

K is outer and rho is inner. Each PE fetches B once per K step, consumes A0
and A1 in order, and updates the corresponding halves of C. The rho steps
execute sequentially on the same PE; different M/N tiles execute spatially.
There are `Pm*Pn` logical kernels, named `gemm_m_n`. Temporal coordinates do not
create kernels and do not require the prototype's `chain` mapping primitives.
`gen_atb_mapping_primitives(Pm, Pn)` bundles independent output tiles onto a
4x4 physical mesh. Each bundle visits rows and columns with stride four, using
the same ordering as the library GEMM mapping. The kernels in one bundle reuse
the physical core's buffers and execute sequentially. Logical mesh extents
need not be multiples of four; singleton groups need no mapping primitive.

For `M=N=K=1024`, the defaults are `Pm=Pn=16` and `Pk=32`:

| Property | Default 1024-square configuration |
| --- | --- |
| Logical output tiles | 256 |
| Physical compute cores | 16 |
| Output tiles per core | 16 |
| L1 A / B / retained C | `32x32` / `32x64` / `64x64` |
| L2 A | `64x32`, split into two L1 transfers |

This configuration has been built and checked on the NPU against NumPy for two
random inputs and a zero-input case. It uses the existing `bundle` primitive;
no aggressive port-utilization environment setting is required.
The optional `Pm=32, Pn=16, Pk=16` configuration also passes the same hardware
checks. It uses `32x64` C tiles and `16x64` A tiles, bundling 512 logical output
tiles onto 16 cores.

The local accumulator is stored as `[rho, At, Nt]`; a zero-copy `allo.view`
presents it to `C_port.put` as `[Mt, Nt]`. Ordinary `range` loops express the
default temporal traversal. L2 aggregation is limited to provably consecutive
row transfers through one multicast interface; DMA packing for the matrix
intrinsics remains separate from logical sharding. Cores with multiple temporal
axes use dynamic object-FIFO lowering to avoid static loop expansion exceeding
instruction memory.

`--pm`, `--pn`, and `--pk` override logical partition counts; `--project` selects
the build directory. By default, `Pm=max(4, ceil(M/64))` and
`Pn=max(4, ceil(N/64))`, bounding the C tile while preserving the previous
4x4 defaults for 128/256-square matrices. Exact division is required. The
default K tile is 32; for
`int16` AIE vector kernels, choose `At` and `Nt` divisible by 8 and `Kt`
divisible by 4. Bundling permits a logical grid larger than the physical mesh,
but each tile's buffers must still fit in local memory. The reference
large-GEMM test uses `int8`, so its local buffer sizes are not interchangeable
with this implementation's `int16` sizes.

For the zero-initialized, full-plane update pattern above, the AIE compiler
keeps C packed across the entire K reduction and updates each rho plane using
a pointer-offset microkernel wrapper. It borrows B's FIFO buffer until the
nested rho loop finishes and allocates C directly in the output FIFO through
its full contiguous reshape. Output DMA performs the final unpacking. These
optimizations conservatively reject unknown aliases, partial views, escaping
uses, and unsupported control flow; those cases retain the existing lowering.

With `--m 1024 --n 1024 --k 1024 --pm 32 --pn 16 --pk 16`, this removes all
core copies and private scratch buffers. The double-buffered A/B/C payloads
occupy 4/16/8 KiB per core (28 KiB total, excluding stack and bookkeeping),
versus 46 KiB before these optimizations. Tile sizes and DMA port placement
are unchanged. Reduced memory use alone does not guarantee a speedup over
symmetric GEMM: at a fixed C tile size, ATB has the same input traffic and
more, smaller microkernel calls.

For the fixed 1024-square configuration above, enable the existing DMA port
spreading option to give each of the four B multicast groups its own shim and
memory-tile route:

```bash
ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH=1 \
python3 tests/dataflow/aie/test_temporal_atb_gemm.py \
    --m 1024 --n 1024 --k 1024 --pm 32 --pn 16 --pk 16
```

This keeps the 16 physical cores and L1 tile sizes unchanged. A still uses
32x64 L2 objects split into two 16x64 L1 objects; B uses 64x64 objects. The
compiler skips optional temporal row aggregation when its enlarged row extent
would exceed the shim DMA limit of 1023. B's K traversal is instead batched as
legal `[1,16,64,64]` descriptors, preserving element order and multicast.
Each shim uses two input channels; each memory tile uses three output channels
and six input channels, including the four C producers. The existing token,
BD allocation, and completion-wait scheduler remains responsible for reuse.
The option is opt-in; omit it to reproduce the unspread topology.

For controlled tile studies, `benchmark_atb_tiles.py` builds one int16 design
with 16 physical cores and port spreading, then records allocated FIFO bytes,
programmed DMA traffic, and scheduled microkernel calls in `metrics.json`:

```bash
python3 tests/dataflow/aie/benchmark_atb_tiles.py \
    --spec '{"kind":"atb","m":1024,"k":1024,"n":1024,"tmc":128,"tk":64,"tn":64,"rho":4}' \
    --project atb_tile_study.prj
```

`TmA=tmc/rho`; use `kind="symmetric"` and `rho=1` for the library GEMM control.
Adding `frontend="temporal"` to that symmetric specification uses the existing
temporal kernel with `rho=1`, to distinguish temporal-loop lowering from
asymmetric buffering without changing the library GEMM implementation.
This helper only builds and inspects. Supply binary inputs and verify outputs
when profiling the generated host with `--warmup 200 --test_iter 1000 -p true`.
The current FIFO policy double-buffers A, B, and C; payload is
`4*(TmA*Tk + Tk*Tn + TmC*Tn)` bytes/core, in addition to stack/control state.

Software regression tests:

```bash
python3 -m pytest tests/test_temporal_layout.py tests/test_memory.py \
    tests/test_aie_dma.py tests/test_aie_buffer_optimizations.py \
    tests/test_aie_mapping_dependencies.py \
    tests/test_linalg.py::test_static_view_dimensions \
    tests/test_linalg.py::test_library_higher_dimension_ops \
    tests/dataflow/aie/test_temporal_atb_gemm.py -q
```

## Motivation

An `@df.kernel` annotation should describe how a complete tensor in off-chip
memory is distributed without requiring the programmer to calculate or declare
local tile sizes. The interface needs a clear boundary between global storage
and values that are local to a processing element (PE):

- Kernel tensor annotations always retain the complete global tensor shape.
- Every `@df.kernel` tensor argument is a port handle, not a local tensor.
- `get()` and `put(value)` cross the port boundary using compiler-derived
  transfer-tile types. Scratch and accumulation buffers are ordinary local Allo
  tensors with programmer-written shapes.
- Spatial ownership and temporal transfer order are logical sharding concepts.
  Packing and vector width remain independent backend optimizations.

This separation lets the compiler infer port transfer types, generate DMA
offsets, and check transfer counts while local computation storage remains
explicit and uses existing Allo tensor declarations.

### Alternatives considered

| Design | Advantages | Limitations |
| --- | --- | --- |
| Separate `df.T(...)` axes and `temporal=[...]` | Makes temporal axes explicit and keeps `mapping` spatial-only. | Duplicates declarations, creates separate axis namespaces, and requires layouts and traversal metadata to refer to the same concept in different places. |
| Treat subtiles as packing size | Closely resembles a physical transfer width and can guide DMA optimization. | Does not express logical ownership, tile count, or multidimensional traversal order; it also couples the public model to a backend optimization. |
| Add an explicit `step` or subtile index | Can express arbitrary access orders. | Exposes mixed-radix scheduling arithmetic to the programmer and makes port synchronization error-prone. |
| Put typed spatial and temporal axes in `mapping` | Provides one source of truth for axis kind, extent, identity, placement, and default traversal. The same `S(i)` syntax composes both kinds of sharding. | Broadens the meaning of `mapping`, makes numeric references sensitive to axis reordering, and requires every compiler stage to distinguish logical axes from physical PE axes. |

The proposed interface adopts the last design. Its additional costs are
intentional: `get_pid()` cannot correspond one-to-one with every entry in
`mapping`, static port-access restrictions require new analysis, and data that
is invariant across a temporal axis must be explicitly saved and reused.

## Proposed public interface

Axes are immutable, typed objects declared directly in the kernel mapping:

```python
Sm = df.Axis.Spatial(Pm, name="m")
Sn = df.Axis.Spatial(Pn, name="n")
Tk = df.Axis.Temporal(Pk, name="k")

@df.kernel(mapping=[Sm, Sn, Tk], args=[A, B, C])
def kernel(...):
    ...
```

The interface follows these rules:

- An axis extent is a positive compile-time integer.
- An axis name is optional and used only for diagnostics. Layouts refer to the
  axis's logical index in `mapping`.
- A legacy integer mapping entry is shorthand for a spatial axis, preserving
  the meaning of existing mappings such as `mapping=[2, 4]`.
- All spatial axes precede all temporal axes in `mapping`. Spatial entries, in
  order, define the physical PE mesh. Temporal entries, in order, define the
  default outer-to-inner traversal; the last temporal axis is fastest.
- `S(i)` shards by logical mapping axis `i`, regardless of whether that axis is
  spatial or temporal.
- Raw string forms such as `S("T")` and a separate `temporal=` kernel argument
  are not part of the final interface.
- `df.get_pid()` continues to return only spatial coordinates. Its tuple length
  is the number of spatial axes, not `len(mapping)`.
- `df.get_axis_id(i)` returns the coordinate of spatial axis `i`, or the current
  coordinate of temporal axis `i` while its explicit `allo.meta_for` is active.
- Ordinary Allo functions and inter-kernel `Stream` objects retain their current
  semantics.

### Ordered shard products

Several axes can shard one tensor dimension through an ordered product:

```python
S(0) * S(2)  # spatial axis 0, then temporal axis 2
S(2) * S(0)  # temporal axis 2, then spatial axis 0
S(0) * S(1)  # two spatial axes shard the same dimension
```

For a global dimension of extent `G` and the ordered product
`X0 * X1 * ... * Xn`, let `Ei` be the extent of `Xi` and `Ci` its coordinate.
The compiler derives:

```text
local_extent = G / product(Ei)
block_id     = (((C0 * E1 + C1) * E2 + C2) ... * En + Cn)
origin       = block_id * local_extent
```

For example, if axes 0 and 2 have extents `P0` and `Pk`:

```text
S(0) * S(2): block_id = pid0 * Pk + tk
S(2) * S(0): block_id = tk * P0 + pid0
```

Both products infer the same leaf extent, but they deliberately select
different global tiles. If a temporal factor precedes a spatial factor, the
tiles owned by one PE can form a non-contiguous set even though every individual
transfer tile remains rectangular.

The compiler requires exact divisibility, valid mapping indices, and at most one
occurrence of any axis in a tensor layout. `Replicate` cannot participate in a
shard product.

## Global, spatial, and local views

The compiler maintains three distinct views of every port tensor:

1. **Global tensor:** the complete shape written in the kernel annotation.
2. **Spatial ownership:** the set of tiles selected after fixing the PE's
   spatial coordinates. This set is not necessarily contiguous for every shard
   product order.
3. **Transfer tile:** the rectangular leaf tensor returned by one `get()` or
   consumed by one `put(value)` after applying every spatial and temporal shard
   factor.

The kernel argument itself always has the first view and acts as a port handle.
The compiler derives the transfer-tile type at each port operation. A local
scratch or accumulation buffer is not another port view: it is declared as an
ordinary Allo tensor with an explicit local shape, and `put()` validates that
shape against the inferred transfer tile.

## Port semantics

- `port.get()` returns one inferred transfer tile and advances that input
  port's cursor.
- `port.put(value)` requires `value` to have the inferred transfer-tile type and
  advances that output port's cursor.
- Ports do not allocate computation buffers. Local buffers use ordinary Allo
  declarations such as `acc: TyO[M // Pm, N // Pn] = 0`.
- Every port has an independent cursor. The first accesses to two different
  ports therefore select temporal coordinate zero on both ports.
- The required transfer count for a port is the product of the extents of the
  temporal axes appearing in that port's layout. A layout with no temporal axis
  implies one transfer.
- Calling `get()` again requests the next tile. Reuse is expressed by saving the
  previously returned local tensor rather than calling `get()` again.

The compiler statically rejects cursor under-consumption and over-consumption,
data-dependent `get()` or `put()` control flow, and a `put()` value whose type
does not match the inferred transfer tile.

### Traversal forms

Temporal traversal has two static source forms.

**Manually unrolled access.** If a port is accessed without an explicit
temporal loop, its cursor follows the temporal axes in `mapping` order,
outermost to innermost. The last applicable temporal axis varies fastest.

**Explicit access.** `allo.meta_for` accepts a temporal axis object. A complete
nest can override the default order, with lexical loop nesting defining the
outer-to-inner traversal:

```python
with allo.meta_for(Touter) as _outer:
    with allo.meta_for(Tinner) as _inner:
        tile = port.get()
```

Every temporal axis participating in that port access must occur exactly once
in the active nest. Partial nests, duplicated axes, and mixing loop-driven and
cursor-driven traversal for one port are invalid. A port may be accessed only
under the temporal axes used by its layout. If a local tile is invariant to an
enclosing temporal axis, it must be fetched outside that loop and explicitly
reused inside it; the compiler does not add implicit caching or duplicate
transfers.

## Example 1: split-K GEMM

The global shapes remain `M x K`, `K x N`, and `M x N`. Spatial axes distribute
the output rows and columns, while one temporal axis streams successive pieces
of the reduction dimension:

```python
Sm = df.Axis.Spatial(Pm, name="m")   # axis 0
Sn = df.Axis.Spatial(Pn, name="n")   # axis 1
Tk = df.Axis.Temporal(Pk, name="k")  # axis 2

LyA = [S(0), S(2)]
LyB = [S(2), S(1)]
LyC = [S(0), S(1)]

@df.region()
def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
    @df.kernel(mapping=[Sm, Sn, Tk], args=[A, B, C])
    def gemm(
        A_port: TyI[M, K] @ LyA,
        B_port: TyI[K, N] @ LyB,
        C_port: TyO[M, N] @ LyC,
    ):
        acc: TyO[M // Pm, N // Pn] = 0

        with allo.meta_for(Tk) as _k:
            acc[:, :] = allo.add(
                allo.matmul(A_port.get(), B_port.get()),
                acc,
            )

        C_port.put(acc)
```

No local dimension aliases such as `Mt`, `Nt`, or `Kt` appear in the program.
The input transfer types are inferred, while the accumulator shape is written
explicitly and checked at the output port boundary:

```text
A_port.get()       -> inferred TyI[M / Pm, K / Pk]
B_port.get()       -> inferred TyI[K / Pk, N / Pn]
acc                -> explicitly declared TyO[M / Pm, N / Pn]
C_port.put(value)  expects TyO[M / Pm, N / Pn]
```

At spatial coordinates `(pm, pn)` and temporal coordinate `k`, the global
origins are:

```text
A: (pm * M / Pm, k  * K / Pk)
B: (k  * K / Pk, pn * N / Pn)
C: (pm * M / Pm, pn * N / Pn)
```

The A and B ports each consume `Pk` tiles. C has no temporal factor, so its local
accumulator is created once and one completed tile is written after the
reduction.

## Example 2: two-dimensional temporal traversal

Consider a global `M x N` tensor spatially divided into four horizontal device
bands. Within every device, temporal axes divide its band into two rows and four
columns of transfer tiles. The desired traversal visits the top and bottom tile
of one column before moving to the next column.

```python
Sd = df.Axis.Spatial(4, name="device")  # axis 0
Tc = df.Axis.Temporal(4, name="col")    # axis 1, outer
Tr = df.Axis.Temporal(2, name="row")    # axis 2, inner

LyX = [S(0) * S(2), S(1)]

@df.region()
def top(X: TyI[M, N], Y: TyO[M, N]):
    @df.kernel(mapping=[Sd, Tc, Tr], args=[X, Y])
    def copy(
        X_port: TyI[M, N] @ LyX,
        Y_port: TyO[M, N] @ LyX,
    ):
        with allo.meta_for(Tc) as _c:
            with allo.meta_for(Tr) as _r:
                Y_port.put(X_port.get())
```

Assuming `M` is divisible by 8 and `N` by 4, the compiler derives:

```text
spatial view per device = [M / 4, N]
transfer tile           = [M / 8, N / 4]

row_origin = (device * 2 + row) * M / 8
col_origin = col * N / 4
```

Because `Tc` is outer and `Tr` is inner, each device's `(row, column)` sequence
is:

```text
(0, 0), (1, 0), (0, 1), (1, 1),
(0, 2), (1, 2), (0, 3), (1, 3)
```

The same default order can be expressed by eight manually unrolled accesses;
the syntax contains no explicit coordinates because each port has its own
cursor:

```python
Y_port.put(X_port.get())  # (row=0, col=0)
Y_port.put(X_port.get())  # (row=1, col=0)
Y_port.put(X_port.get())  # (row=0, col=1)
Y_port.put(X_port.get())  # (row=1, col=1)
Y_port.put(X_port.get())  # (row=0, col=2)
Y_port.put(X_port.get())  # (row=1, col=2)
Y_port.put(X_port.get())  # (row=0, col=3)
Y_port.put(X_port.get())  # (row=1, col=3)
```

Explicit loops may intentionally override traversal without changing placement.
Reversing the nest makes row outer and column inner:

```python
with allo.meta_for(Tr) as _r:
    with allo.meta_for(Tc) as _c:
        Y_port.put(X_port.get())
```

This visits `(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), ...`; the layout and global
origin formulas remain unchanged.

## Future compiler work

Implementation should proceed as one intentional dataflow API change rather
than preserving two meanings for kernel tensor arguments:

1. Add immutable spatial and temporal axis types. Normalize legacy integer
   mapping entries into spatial axes and derive the physical mesh by filtering
   spatial entries from the complete logical mapping.
2. Extend `Shard` with an ordered `ShardProduct`. Normalize layout expressions,
   infer leaf shapes, and represent mixed-radix origins symbolically until both
   spatial and temporal coordinates are known.
3. Preserve global shapes on kernel arguments and introduce an internal tensor
   port type whose `get()` and `put()` operations expose or validate inferred
   leaf types. Keep local allocation in ordinary typed Allo declarations.
4. Extend `allo.meta_for` to accept temporal axis objects. Add static traversal,
   cursor-count, access-scope, and invariant-reuse validation.
5. Instantiate functions and AIE cores only over the spatial mesh. Enumerate
   temporal DMA coordinates separately and lower mixed-radix origins into DMA
   or object-FIFO access patterns; temporal axes must not become fake mesh
   dimensions.
6. Migrate existing `@df.kernel` tensor arguments to explicit port operations.
   Keep ordinary functions and inter-kernel streams unchanged.

## Future test plan

Positive coverage should include:

- Split-K GEMM with an explicit local accumulator shape and no local dimension
  aliases.
- The four-device, two-dimensional temporal traversal above.
- Identical values and DMA order for explicit loops and manual unrolling.
- Different origins for `S(0) * S(2)` and `S(2) * S(0)`.
- One tensor dimension sharded over two spatial axes.
- An explicit temporal-loop order override.
- Explicit reuse of a port invariant to an enclosing temporal axis.
- Legacy integer mappings and spatial-only `df.get_pid()` behavior.

Static failure tests should cover:

- Non-positive or non-constant axis extents.
- Global extents that are not exactly divisible by their shard products.
- Invalid mapping indices, undeclared axes, or repeated axes in one layout.
- A shard product containing `Replicate`.
- Partial, duplicated, or otherwise invalid explicit temporal-loop nests.
- Data-dependent port accesses or mixed traversal modes.
- Cursor under-consumption or over-consumption.
- A `put()` value with an incompatible leaf shape.

The first implementation remains limited to static rectangular exact-division
sharding. Tails, halos, overlapping windows, arbitrary indexed traversal, and
backend packing policy are deferred.

### Single-C and kernel-scheduling experiments

AIE builds accept `output_fifo_depth=1` to single-buffer global outputs on
compute tiles, retaining depth 2 at memory-tile endpoints. The default is 2.
This applies to library GEMM and temporal GEMM, including rho=1 controls.
`fifo_lowering="auto"` preserves the existing heuristic; `"static"` and
`"dynamic"` allow explicit, shared FIFO scheduling experiments. Static mode
can increase code size, so validate program-memory feasibility for each tile.

`benchmark_atb_tiles.py` accepts these options in its JSON spec. The additional
`"kernel_variant":"full_unroll_k"` is an experimental int16 Peano scheduling
option: it explicitly unrolls the 2x2 MMUL K loop before compiling the generated
external kernel. The helper retains exact generated C++ and validates the
expected source-loop pattern. It is available to both symmetric and ATB cases;
ordinary builds and benchmark specs retain the original kernel by default.

For example, inside the existing AIE container with `conda activate allo`:

```bash
python tests/dataflow/aie/benchmark_atb_tiles.py \
  --spec '{"kind":"atb","m":2048,"k":4096,"n":2048,"tmc":64,"tk":128,"tn":64,"rho":2,"output_fifo_depth":1,"kernel_variant":"full_unroll_k"}' \
  --project atb_single_c_example.prj
```

This builds and inspects the project; it does not profile the device. Use the
established host profiling protocol and exact output verification. Temporal
symmetric controls use `"kind":"symmetric","frontend":"temporal","rho":1`.
A performance claim must compare against the best tested symmetric tiling with
equivalent applicable kernel and FIFO optimizations, rather than only the
symmetric counterpart of an ATB tile that exceeds local-memory capacity.
