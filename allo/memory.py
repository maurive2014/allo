# pylint: disable=too-many-instance-attributes, redundant-returns-doc, unsupported-binary-operation
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product
from collections import defaultdict
from dataclasses import dataclass
from math import prod


class Axis:
    @dataclass(frozen=True)
    class Spatial:
        size: int
        name: str

    @dataclass(frozen=True)
    class Temporal:
        size: int
        name: str


class Layout:
    """
      Example:

      mesh_dim = [2, 2, 2]
        +-----+
     2 /|    /|
      +-+---+ +
    1 |/    |/
      +-----+
         0
      2D tensor: tensor_dim = [32, 32]
      +-----------+
      | 0,0 | 0,1 |
      +-----------+
      | 1,0 | 1,1 |
      +-----------+

      partitions = [Shard(0), Shard(2)] ->  (tensor_dim[0], shard on mesh_dim[0]),
                                            (tensor_dim[1], shard on mesh_dim[2])

      PE tile (a, ?, b) gets tensor tile (a, b)
    """

    @dataclass(frozen=True)
    class Shard:
        axis: int

        def __mul__(self, other):
            if isinstance(other, Layout.Shard):
                return Layout.ShardProduct((self.axis, other.axis))
            if isinstance(other, Layout.ShardProduct):
                return Layout.ShardProduct((self.axis,) + other.axes)
            return NotImplemented

    @dataclass(frozen=True)
    class ShardProduct:
        """Ordered mixed-radix sharding of one tensor dimension."""

        axes: tuple[int, ...]

        def __mul__(self, other):
            if isinstance(other, Layout.Shard):
                return Layout.ShardProduct(self.axes + (other.axis,))
            if isinstance(other, Layout.ShardProduct):
                return Layout.ShardProduct(self.axes + other.axes)
            return NotImplemented

    @dataclass(frozen=True)
    class Replicate:
        pass

    def __init__(self, partitions: list[Replicate | Shard | ShardProduct]):
        self.partitions = partitions

    @staticmethod
    def shard_axes(partition):
        if isinstance(partition, Layout.Shard):
            return (partition.axis,)
        if isinstance(partition, Layout.ShardProduct):
            return partition.axes
        if partition is Layout.Replicate or isinstance(partition, Layout.Replicate):
            return ()
        raise ValueError(f"Invalid layout partition: {partition}")

    def partition_sizes(self, mesh_dims):
        """Validate axes and return the number of blocks per tensor dimension."""
        seen = set()
        sizes = []
        for partition in self.partitions:
            axes = self.shard_axes(partition)
            for axis in axes:
                if (
                    not isinstance(axis, int)
                    or isinstance(axis, bool)
                    or not 0 <= axis < len(mesh_dims)
                ):
                    raise ValueError(f"Invalid mapping axis {axis}")
                if axis in seen:
                    raise ValueError(f"Repeated mapping axis {axis} in layout")
                seen.add(axis)
            sizes.append(prod(mesh_dims[axis] for axis in axes))
        return sizes

    def get_placement(self, mesh_dims: list[int]):
        """
        Calculate mapping from tensor tile IDs to PE tile IDs based on the partition scheme.

        Args:
            mesh_dims (list): Dimensions of the device mesh (e.g., [4] for 1D, [2,2] for 2D)

        Returns:
            dict: A mapping from tensor tile IDs to corresponding PE tile coordinates
        """
        self.partition_sizes(mesh_dims)
        # Generate all possible PE coordinates
        pe_coords = list(product(*[range(dim) for dim in mesh_dims]))
        mapping = defaultdict(list)
        # For each PE coordinate, determine its tensor tile ID
        for pe_coord in pe_coords:
            tensor_id_parts = []
            for partition in self.partitions:
                block_id = 0
                for axis in self.shard_axes(partition):
                    block_id = block_id * mesh_dims[axis] + pe_coord[axis]
                tensor_id_parts.append(block_id)

            tensor_id = tuple(tensor_id_parts)
            # Add this PE coordinate to the mapping for this tensor ID
            mapping[tensor_id].append(pe_coord)

        # Post-process the mapping to combine PE coordinates for replicated dimensions
        result: dict[tuple[int | str, ...], list[tuple[int, ...]]] = {}
        for tensor_id, coords in mapping.items():
            # Convert to tuples for final output
            result[tensor_id] = [tuple(coord) for coord in coords]
        return result


class Memory:
    """
    Memory interface specification for HLS synthesis.

    This class allows users to specify memory implementation details
    for array/tensor arguments, similar to Vitis HLS pragma interface.

    Supported options:
    - resource: Memory resource type
        - "BRAM": Block RAM
        - "URAM": Ultra RAM
        - "LUTRAM": LUT-based RAM
        - "SRL": Shift Register LUT
        - "AUTO": Let the tool decide (default)
    - storage_type: RAM access pattern
        - "RAM_1P": Single-port RAM
        - "RAM_2P": Simple dual-port RAM (one read, one write port)
        - "RAM_T2P": True dual-port RAM (two read/write ports)
        - "RAM_1WNR": Single write, N read ports
        - "RAM_S2P": Simple dual-port (alias for RAM_2P)
        - "ROM_1P": Single-port ROM
        - "ROM_2P": Dual-port ROM
        - "ROM_NP": N-port ROM
    - latency: Memory access latency in cycles
    - depth: Depth of the memory (for streams/FIFOs)

    Example:
        def kernel(a: int32[32] @ Memory(resource="URAM")):
            ...

        def kernel(b: float32[64, 64] @ Memory(resource="BRAM", storage_type="RAM_2P")):
            ...
    """

    # Valid resource types
    VALID_RESOURCE = {"BRAM", "URAM", "LUTRAM", "SRL", "AUTO"}

    # Memory space encoding for MLIR MemRefType
    # Format: resource_code * 16 + storage_type_code
    # This allows encoding both resource and storage_type in a single integer
    RESOURCE_TO_SPACE = {
        "AUTO": 0,
        "BRAM": 1,
        "URAM": 2,
        "LUTRAM": 3,
        "SRL": 4,
    }

    STORAGE_TYPE_TO_CODE = {
        None: 0,
        "RAM_1P": 1,
        "RAM_2P": 2,
        "RAM_T2P": 3,
        "RAM_1WNR": 4,
        "RAM_S2P": 5,
        "ROM_1P": 6,
        "ROM_2P": 7,
        "ROM_NP": 8,
    }

    # Reverse mappings for decoding in C++ emitter
    SPACE_TO_RESOURCE = {v: k for k, v in RESOURCE_TO_SPACE.items()}
    CODE_TO_STORAGE_TYPE = {v: k for k, v in STORAGE_TYPE_TO_CODE.items()}

    # Valid storage types
    VALID_STORAGE_TYPE = {
        "RAM_1P",
        "RAM_2P",
        "RAM_T2P",
        "RAM_1WNR",
        "RAM_S2P",
        "ROM_1P",
        "ROM_2P",
        "ROM_NP",
    }

    def __init__(
        self,
        resource: str = "AUTO",
        storage_type: str = None,
        latency: int = None,
        depth: int = None,
    ):
        """
        Create a Memory specification.

        Parameters
        ----------
        resource : str
            Memory resource type. One of: BRAM, URAM, LUTRAM, SRL, AUTO.
            Default is AUTO (let the tool decide).
        storage_type : str
            RAM access pattern. One of: RAM_1P, RAM_2P, RAM_T2P, RAM_1WNR,
            RAM_S2P, ROM_1P, ROM_2P, ROM_NP. Default is None (tool decides).
        latency : int
            Memory access latency in cycles. Default is None (tool decides).
        depth : int
            Depth of the memory. Useful for streams/FIFOs. Default is None.
        """
        if resource.upper() not in self.VALID_RESOURCE:
            raise ValueError(
                f"Invalid resource '{resource}'. Must be one of: {', '.join(self.VALID_RESOURCE)}"
            )
        self.resource = resource.upper()

        if storage_type is not None:
            if storage_type.upper() not in self.VALID_STORAGE_TYPE:
                raise ValueError(
                    f"Invalid storage_type '{storage_type}'. "
                    f"Must be one of: {', '.join(self.VALID_STORAGE_TYPE)}"
                )
            self.storage_type = storage_type.upper()
        else:
            self.storage_type = None

        self.latency = latency
        self.depth = depth

    def __repr__(self):
        parts = [f'resource="{self.resource}"']
        if self.storage_type is not None:
            parts.append(f'storage_type="{self.storage_type}"')
        if self.latency is not None:
            parts.append(f"latency={self.latency}")
        if self.depth is not None:
            parts.append(f"depth={self.depth}")
        return f"Memory({', '.join(parts)})"

    def __eq__(self, other):
        if not isinstance(other, Memory):
            return False
        return (
            self.resource == other.resource
            and self.storage_type == other.storage_type
            and self.latency == other.latency
            and self.depth == other.depth
        )

    def __hash__(self):
        return hash((self.resource, self.storage_type, self.latency, self.depth))

    def get_memory_space(self) -> int:
        """
        Get the memory space encoding for MLIR MemRefType.

        The encoding combines resource and storage_type into a single integer:
            memory_space = resource_code * 16 + storage_type_code

        This allows the C++ emitter to decode both values and emit
        appropriate HLS pragmas.

        Returns
        -------
        int
            Memory space encoding. 0 means AUTO (default, no pragma needed).
        """
        resource_code = self.RESOURCE_TO_SPACE.get(self.resource, 0)
        storage_code = self.STORAGE_TYPE_TO_CODE.get(self.storage_type, 0)
        return resource_code * 16 + storage_code

    @staticmethod
    def decode_memory_space(memory_space: int) -> tuple:
        """
        Decode memory space integer back to (resource, storage_type).

        Parameters
        ----------
        memory_space : int
            The memory space encoding.

        Returns
        -------
        tuple
            (resource: str, storage_type: str or None)
        """
        resource_code = memory_space // 16
        storage_code = memory_space % 16
        resource = Memory.SPACE_TO_RESOURCE.get(resource_code, "AUTO")
        storage_type = Memory.CODE_TO_STORAGE_TYPE.get(storage_code, None)
        return resource, storage_type


class DTensor:
    """
    Distributed tensor.
    """

    def __init__(self, rank, mapping, shape, dtype, spec, name=None, top_name=None):
        self.rank = rank
        self.axis: list[int | Axis.Temporal] = mapping
        if mapping is not None:
            self.mapping = [
                axis if isinstance(axis, int) else axis.size for axis in mapping
            ]
            if any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for size in self.mapping
            ):
                raise ValueError("Mapping extents must be positive integers")
        else:
            self.mapping = mapping
        self.shape = shape  # tensor shape
        self.dtype = dtype
        self.name = name
        self.top_name = top_name

        # Handle spec: can be Layout, Memory, or None
        self.layout: Layout = None
        self.memory: Memory = None

        if isinstance(spec, Layout):
            self.layout = spec
        elif isinstance(spec, Memory):
            self.memory = spec
        elif spec is not None:
            # For backward compatibility, treat as layout if it has placement attr
            if hasattr(spec, "placement"):
                self.layout = spec
            elif hasattr(spec, "resource"):
                self.memory = spec

        if self.layout is not None and mapping is not None:
            self.get_local_shape()
            # tensor tile ID -> PE tile IDs
            self.global_placement: dict[
                tuple[int | str, ...], list[tuple[int, ...]]
            ] = self.layout.get_placement(self.mapping)
            # Placement is fixed at construction. Rebuilding this reverse
            # index for every PE makes large spatial task graphs quadratic.
            self._pe_to_tensor_tile = {
                pe_id: tensor_tile_id
                for tensor_tile_id, pe_tile_ids in self.global_placement.items()
                for pe_id in pe_tile_ids
            }
        self.access_pattern_set = False
        self.global_id: int = None
        self.is_input: bool = None
        self.type_as_param: list = None

    def get_local_shape(self):
        """
        Get the local shape of the tensor.
        """
        if self.layout is None or self.mapping is None:
            return self.shape
        if len(self.shape) != len(self.layout.partitions):
            raise ValueError("Layout rank must match tensor rank")
        local_shape = []
        for shape, size in zip(self.shape, self.layout.partition_sizes(self.mapping)):
            if shape % size:
                raise ValueError(
                    f"Tensor extent {shape} must be divisible by shard product {size}"
                )
            local_shape.append(shape // size)
        return tuple(local_shape)

    def set_global_info(self, global_id: int, is_input: bool):
        self.global_id = global_id
        self.is_input = is_input

    def contiguous_temporal_rows(self):
        """Number of adjacent row transfers that can share one L2 input tile.

        Only trailing temporal factors of the row shard product qualify, and
        they must be the innermost axes in this port's traversal. For ATB this
        groups the rho half-rows while preserving the smaller L1 transfer type.
        """
        if self.layout is None or len(self.shape) != 2:
            return 1
        row_axes = self.layout.shard_axes(self.layout.partitions[0])
        trailing = []
        for axis in reversed(row_axes):
            if not isinstance(self.axis[axis], Axis.Temporal):
                break
            trailing.insert(0, axis)
        used_axes = {
            axis
            for partition in self.layout.partitions
            for axis in self.layout.shard_axes(partition)
        }
        temporal_axes = [
            i
            for i, axis in enumerate(self.axis)
            if i in used_axes and isinstance(axis, Axis.Temporal)
        ]
        if not trailing or temporal_axes[-len(trailing) :] != trailing:
            return 1
        return prod(self.mapping[axis] for axis in trailing)

    def set_access_pattern(self):
        """
        Specify how to access the dtensor (local tensor) from the global tensor
            (tensor has at most 4 dimensions: DMA support 4-dimension address generation)
        Set offset map for each tensor tile.

        Returns:
            - device_dims (list): Indexes of tensor dimensions sharded across devices.
            - size (list): 4D tensor dimensions used for access.
            - stride (list): Stride along each dimension in the global tensor.
        """
        if self.access_pattern_set:
            return
        self.access_pattern_set = True
        # tensor tile ID -> address offset
        self.offset_map: dict[tuple[int | str, ...], Offset4D] = {}
        partition_str = "".join(
            ["S" if self.layout.shard_axes(p) else "R" for p in self.layout.partitions]
        )
        partition_sizes = self.layout.partition_sizes(self.mapping)
        if len(self.shape) == 1:
            if partition_str == "S":
                parts = partition_sizes[0]
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(0, 0, i, 0)
                shard_size = self.shape[0] // parts
                device_dims = [2]  # partition idx = 2
                size = [1, 1, parts, shard_size]
                stride = [0, 0, shard_size, 1]
            elif partition_str == "R":
                for key in self.global_placement.keys():
                    self.offset_map[key] = Offset4D(0, 0, 0, 0)
                device_dims = []  # no partition
                size = [1, 1, 1, self.shape[0]]
                stride = [0, 0, 0, 1]
            else:
                raise ValueError("Unsupported access pattern for 1D tensor.")
        elif len(self.shape) == 2:
            tensor_m, tensor_n = self.shape  # [tensor_m x tensor_n]
            if partition_str == "SS":
                device_a, device_b = partition_sizes
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(i // device_b, i % device_b, 0, 0)
                device_dims = [0, 1]
                size = [device_a, device_b, tensor_m // device_a, tensor_n // device_b]
                stride = [
                    (tensor_m // device_a) * tensor_n,
                    tensor_n // device_b,
                    tensor_n,
                    1,
                ]
            elif partition_str == "SR":
                device_a = partition_sizes[0]
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(i // device_a, i % device_a, 0, 0)
                # First dim sharded across all devices, second replicated
                device_dims = [1]
                size = [1, device_a, tensor_m // device_a, tensor_n]
                stride = [0, (tensor_m // device_a) * tensor_n, tensor_n, 1]
            elif partition_str == "RS":
                device_b = partition_sizes[1]
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(i // device_b, i % device_b, 0, 0)
                # First dim replicated, second sharded across second dim of mesh
                device_dims = [1]
                size = [1, device_b, tensor_m, tensor_n // device_b]
                stride = [0, tensor_n // device_b, tensor_n, 1]
            elif partition_str == "RR":
                for key in self.global_placement.keys():
                    self.offset_map[key] = Offset4D(0, 0, 0, 0)
                # Both dimensions replicated
                device_dims = []
                size = [1, 1, tensor_m, tensor_n]
                stride = [0, 0, tensor_n, 1]
            else:
                raise ValueError("Unsupported access pattern for 2D tensor.")
        else:
            raise ValueError("Unsupported access pattern.")
        self.shared_dims, self.size, self.stride = device_dims, size, stride

    def PE_tile_id_to_tensor_tile_id(
        self, pe_tile_id: tuple[int, ...]
    ) -> list[tuple[int | str, ...]]:
        """Return tensor tiles accessed by a physical PE in temporal order.

        ``pe_tile_id`` contains coordinates for spatial axes only. Temporal
        coordinates are enumerated from ``self.axis`` in outer-to-inner order,
        with the last applicable temporal axis varying fastest. Axes absent
        from this port's layout do not advance its cursor: their tile is reused.
        """
        pe_to_tensor_tile = self._pe_to_tensor_tile
        used_axes = {
            axis
            for partition in self.layout.partitions
            for axis in self.layout.shard_axes(partition)
        }
        temporal_ranges = [
            range(axis.size if i in used_axes else 1)
            for i, axis in enumerate(self.axis)
            if isinstance(axis, Axis.Temporal)
        ]
        tensor_tile_ids = []
        for temporal_tile_id in product(*temporal_ranges):
            spatial_idx = 0
            temporal_idx = 0
            logical_pe_coords = []
            for axis in self.axis:
                if isinstance(axis, Axis.Temporal):
                    logical_pe_coords.append(temporal_tile_id[temporal_idx])
                    temporal_idx += 1
                else:
                    logical_pe_coords.append(pe_tile_id[spatial_idx])
                    spatial_idx += 1
            logical_pe_tile_id = tuple(logical_pe_coords)
            if logical_pe_tile_id not in pe_to_tensor_tile:
                raise ValueError(
                    f"PE tile ID {pe_tile_id} with temporal coordinates "
                    f"{temporal_tile_id} not found in {self.global_placement}"
                )
            tensor_tile_ids.append(pe_to_tensor_tile[logical_pe_tile_id])
        return tensor_tile_ids

    def __str__(self):
        parts = [
            f"name={self.name}",
            f"shape={self.shape}",
            f"dtype={self.dtype}",
        ]
        if self.layout is not None:
            parts.append(f"layout={self.layout}")
        if self.memory is not None:
            parts.append(f"memory={self.memory}")
        parts.extend(
            [
                f"mapping={self.mapping}",
                f"rank={self.rank}",
                f"local_shape={self.get_local_shape()}",
            ]
        )
        return f"DTensor({', '.join(parts)})"

    def __repr__(self):
        return f"{self.name}"


# ############################################################
# 4D Addressing
# ############################################################
@dataclass(frozen=True)
class Offset4D:
    """
    4D offset.
    indexed from left to right.
        offset_a: offset along mesh_dim[0]
        offset_b: offset along mesh_dim[1]
        offset_c: offset along mesh_dim[2]
        offset_d: offset along mesh_dim[3]
    """

    offset_a: int
    offset_b: int
    offset_c: int
    offset_d: int

    def get_offset(self, dim: int) -> int:
        if dim == 0:
            return self.offset_a
        if dim == 1:
            return self.offset_b
        if dim == 2:
            return self.offset_c
        if dim == 3:
            return self.offset_d
        raise ValueError(f"Invalid dimension: {dim}")

    def get_next_offset(self, dim: int) -> "Offset4D":
        if dim == 0:
            return Offset4D(
                self.offset_a + 1, self.offset_b, self.offset_c, self.offset_d
            )
        if dim == 1:
            return Offset4D(
                self.offset_a, self.offset_b + 1, self.offset_c, self.offset_d
            )
        if dim == 2:
            return Offset4D(
                self.offset_a, self.offset_b, self.offset_c + 1, self.offset_d
            )
        if dim == 3:
            return Offset4D(
                self.offset_a, self.offset_b, self.offset_c, self.offset_d + 1
            )
        raise ValueError(f"Invalid dimension: {dim}")

    def check_next_offset(self, next_: "Offset4D") -> bool:
        """
        Check whether next_ is the next offset of self
        """
        diffs = [
            next_.offset_a - self.offset_a,
            next_.offset_b - self.offset_b,
            next_.offset_c - self.offset_c,
            next_.offset_d - self.offset_d,
        ]
        if diffs.count(0) == 3 and diffs.count(1) == 1:
            indices = [i for i, diff in enumerate(diffs) if diff == 1]
            return indices[0]
        return -1

    def to_list(self) -> list[int]:
        return [self.offset_a, self.offset_b, self.offset_c, self.offset_d]

    def __eq__(self, other) -> bool:
        return (
            self.offset_a == other.offset_a
            and self.offset_b == other.offset_b
            and self.offset_c == other.offset_c
            and self.offset_d == other.offset_d
        )

    def __hash__(self) -> int:
        return hash((self.offset_a, self.offset_b, self.offset_c, self.offset_d))

    def __str__(self) -> str:
        return f"offset4D ({self.offset_a}, {self.offset_b}, {self.offset_c}, {self.offset_d})"

    def __repr__(self) -> str:
        return self.__str__()


class Size4D:
    """
    4D size.
    indexed from left to right.
        size_a: size along mesh_dim[0]
        size_b: size along mesh_dim[1]
        size_c: size along mesh_dim[2]
        size_d: size along mesh_dim[3]
    """

    def __init__(self, size_a: int, size_b: int, size_c: int, size_d: int):
        self.size_a = size_a
        self.size_b = size_b
        self.size_c = size_c
        self.size_d = size_d

    def copy(self) -> "Size4D":
        return Size4D(self.size_a, self.size_b, self.size_c, self.size_d)

    def get_k_slice(self, k: int) -> "Size4D":
        """
        get a slice of size k
        """
        size_list = [self.size_a, self.size_b, self.size_c, self.size_d]
        dim = 3
        while dim >= 0 and k >= size_list[dim]:
            assert k % size_list[dim] == 0, "Invalid slice size"
            k //= size_list[dim]
            dim -= 1
        size_list[dim] = k
        dim -= 1
        while dim >= 0:
            size_list[dim] = 1
            dim -= 1
        return Size4D.from_list(size_list)

    @classmethod
    def from_list(cls, size_list: list[int]) -> "Size4D":
        if len(size_list) > 4:
            raise ValueError(
                f"Size4D must have at most 4 dimensions, but got {len(size_list)}"
            )
        while len(size_list) < 4:
            size_list.insert(0, 1)
        return cls(*size_list)

    @staticmethod
    def coalesce(size_1: "Size4D", size_2: "Size4D") -> "Size4D":
        return Size4D(
            (size_1.size_a * size_2.size_a),
            (size_1.size_b * size_2.size_b),
            (size_1.size_c * size_2.size_c),
            (size_1.size_d * size_2.size_d),
        )

    @staticmethod
    def subtract(a: "Size4D", b: "Size4D") -> "Size4D":
        list_a, list_b = a.to_list(), b.to_list()
        sub = False
        for i in range(4):
            if list_a[i] != list_b[i]:
                if sub:
                    raise ValueError("Cannot subtract")
                sub = True
                list_a[i] = list_a[i] - list_b[i]
        if not sub:
            return Size4D.from_list([0, 0, 0, 0])
        return Size4D.from_list(list_a)

    @staticmethod
    def divide(a: "Size4D", b: "Size4D") -> "Size4D":
        list_a, list_b = a.to_list(), b.to_list()
        for i in range(4):
            assert list_a[i] % list_b[i] == 0, "invalid division"
            list_a[i] //= list_b[i]
        return Size4D.from_list(list_a)

    @staticmethod
    def multiply(a: "Size4D", b: "Size4D") -> "Size4D":
        list_a, list_b = a.to_list(), b.to_list()
        for i in range(4):
            list_a[i] *= list_b[i]
        return Size4D.from_list(list_a)

    def get_dim_size(self, dim: int) -> int:
        if dim == 0:
            return self.size_a
        if dim == 1:
            return self.size_b
        if dim == 2:
            return self.size_c
        if dim == 3:
            return self.size_d
        raise ValueError(f"Invalid dimension: {dim}")

    def set_dim_size(self, dim: int, size: int):
        if dim == 0:
            self.size_a = size
        elif dim == 1:
            self.size_b = size
        elif dim == 2:
            self.size_c = size
        elif dim == 3:
            self.size_d = size
        else:
            raise ValueError(f"Invalid dimension: {dim}")

    def inc_on_dim(self, dim: int):
        if dim == 0:
            self.size_a += 1
        elif dim == 1:
            self.size_b += 1
        elif dim == 2:
            self.size_c += 1
        elif dim == 3:
            self.size_d += 1
        else:
            raise ValueError(f"Invalid dimension: {dim}")

    def get_total_size(self) -> int:
        return self.size_a * self.size_b * self.size_c * self.size_d

    def to_list(self) -> list[int]:
        return [self.size_a, self.size_b, self.size_c, self.size_d]

    def __eq__(self, other) -> bool:
        return (
            self.size_a == other.size_a
            and self.size_b == other.size_b
            and self.size_c == other.size_c
            and self.size_d == other.size_d
        )

    def __hash__(self) -> int:
        return hash((self.size_a, self.size_b, self.size_c, self.size_d))

    def __str__(self) -> str:
        return f"size4D ({self.size_a}, {self.size_b}, {self.size_c}, {self.size_d})"

    def __repr__(self) -> str:
        return self.__str__()


def coalesce_memory_access(offset_map: dict[Offset4D, list]):
    """
    Coalesce memory tile access.
        The default way is sending each tiling separately.
        But we can try to coalesce some.
    """
    offsets = list(offset_map.keys())
    access: dict[Offset4D, Size4D] = {offset: Size4D(1, 1, 1, 1) for offset in offsets}
    coalesce_info: dict[Offset4D, list[Offset4D]] = {
        offset: [offset] for offset in offsets
    }
    connected_nodes: dict[Offset4D, list[list]] = {
        offset: [offset_map[offset]] for offset in offsets
    }
    coalesce_dim = 3
    while coalesce_dim >= 0:
        sorted_offsets = sorted(
            access.keys(),
            key=lambda x: (x.offset_a, x.offset_b, x.offset_c, x.offset_d),
        )
        coalesced = set()
        base_offset, inc_offset, base_size = None, None, None
        for offset in sorted_offsets:
            if offset in coalesced:
                continue
            if base_offset is None:
                base_offset, inc_offset, base_size = offset, offset, access[offset]
            else:
                inc_offset = inc_offset.get_next_offset(coalesce_dim)
                if inc_offset in access:
                    base_size.inc_on_dim(coalesce_dim)
                    coalesced.add(inc_offset)
                    coalesce_info[base_offset].extend(coalesce_info[inc_offset])
                    connected_nodes[base_offset].extend(connected_nodes[inc_offset])
                else:
                    base_offset, inc_offset, base_size = offset, offset, access[offset]
        for offset in coalesced:
            access.pop(offset)
            coalesce_info.pop(offset)
            connected_nodes.pop(offset)
        coalesce_dim -= 1
    return access, coalesce_info, connected_nodes
