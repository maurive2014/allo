# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered spatial/temporal layouts and their global DMA addresses."""

import numpy as np
import pytest

from allo.ir.types import int16
from allo.memory import Axis, DTensor, Layout

S = Layout.Shard
R = Layout.Replicate


@pytest.mark.parametrize("temporal_first", [False, True])
def test_shard_product_dma(temporal_first):
    mapping = [Axis.Spatial(2, "m"), Axis.Temporal(3, "k"), Axis.Temporal(2, "r")]
    rows = S(2) * S(0) if temporal_first else S(0) * S(2)
    tensor = DTensor(1, mapping, (32, 24), int16, Layout([rows, S(1)]))
    assert tensor.get_local_shape() == (8, 8)
    assert tensor.contiguous_temporal_rows() == (1 if temporal_first else 2)
    tensor.set_access_pattern()
    source = np.arange(32 * 24).reshape(32, 24)
    for m in range(2):
        ids = tensor.PE_tile_id_to_tensor_tile_id((m,))
        assert ids == [
            (r * 2 + m if temporal_first else m * 2 + r, k)
            for k in range(3)
            for r in range(2)
        ]
        for row, col in ids:
            offset = tensor.offset_map[(row, col)].to_list()
            base = sum(o * s for o, s in zip(offset, tensor.stride))
            gathered = np.array(
                [
                    source.flat[base + i * tensor.stride[2] + j]
                    for i in range(8)
                    for j in range(8)
                ]
            ).reshape(8, 8)
            np.testing.assert_array_equal(
                gathered, source[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8]
            )


@pytest.mark.parametrize(
    "layout,expected",
    [
        ([S(1), R], [(k, 0) for k in range(3)]),
        ([R, S(2)], [(0, r) for r in range(2)]),
        ([R, R], [(0, 0)]),
    ],
)
def test_invariant_temporal_axes(layout, expected):
    mapping = [2, Axis.Temporal(3, "k"), Axis.Temporal(2, "r")]
    tensor = DTensor(1, mapping, (24, 16), int16, Layout(layout))
    for m in range(2):
        assert tensor.PE_tile_id_to_tensor_tile_id((m,)) == expected


def test_three_factor_product():
    assert S(0) * S(1) * S(2) == S(0) * (S(1) * S(2))
    tensor = DTensor(
        2, [2, 3, Axis.Temporal(4, "t")], (96,), int16, Layout([S(0) * S(2) * S(1)])
    )
    assert tensor.get_local_shape() == (4,)
    assert tensor.PE_tile_id_to_tensor_tile_id((1, 2)) == [(14,), (17,), (20,), (23,)]
    tensor.set_access_pattern()
    assert tensor.size == [1, 1, 24, 4]
    assert tensor.offset_map[(17,)].to_list() == [0, 0, 17, 0]


def test_row_aggregation_respects_traversal():
    # If K varies fastest, consecutive transfers are not adjacent row slices.
    tensor = DTensor(
        1,
        [2, Axis.Temporal(2, "r"), Axis.Temporal(3, "k")],
        (32, 24),
        int16,
        Layout([S(0) * S(1), S(2)]),
    )
    assert tensor.contiguous_temporal_rows() == 1


@pytest.mark.parametrize(
    "mapping,shape,layout,match",
    [
        ([2, 3], (10,), [S(0) * S(1)], "divisible"),
        ([2], (8,), [S(1)], "Invalid mapping axis"),
        ([2], (8,), [S(-1)], "Invalid mapping axis"),
        ([2], (8,), [S(0) * S(0)], "Repeated mapping axis"),
        ([2], (8, 8), [S(0), S(0)], "Repeated mapping axis"),
        ([0], (8,), [S(0)], "positive integers"),
        ([True], (8,), [S(0)], "positive integers"),
        ([2], (8, 8), [S(0)], "Layout rank"),
    ],
)
def test_invalid_layout(mapping, shape, layout, match):
    with pytest.raises(ValueError, match=match):
        DTensor(1, mapping, shape, int16, Layout(layout))


def test_replicate_product_rejected():
    with pytest.raises(TypeError):
        _ = S(0) * R


def test_repeated_tile_lookup_reuses_placement_index():
    tensor = DTensor(
        2,
        [8, 8, Axis.Temporal(4, "k"), Axis.Temporal(2, "rho")],
        (128, 64),
        int16,
        Layout([S(0) * S(3), S(2)]),
    )
    expected = tensor.PE_tile_id_to_tensor_tile_id((3, 5))

    class NoRescan(dict):
        def items(self):
            raise AssertionError("Tile lookup rescanned the full placement")

    tensor.global_placement = NoRescan(tensor.global_placement)
    for _ in range(3):
        assert tensor.PE_tile_id_to_tensor_tile_id((3, 5)) == expected
    assert expected == [(6 + r, k) for k in range(4) for r in range(2)]
