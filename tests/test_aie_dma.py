# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMA aggregation must preserve FIFO traversal and encodable row sizes."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("aie")
from allo.backend.aie.mlir_codegen import CodeGenerator


@pytest.mark.parametrize(
    "rows,factor,variant,aggregate",
    [
        (16, 2, "valid", True),
        (341, 3, "valid", True),  # 1023 is representable.
        (512, 2, "valid", False),  # 1024 is not.
        (64, 16, "valid", False),  # ATB B after port spreading.
        (16, 2, "token", False),
        (16, 2, "order", False),
    ],
)
def test_temporal_row_aggregation(rows, factor, variant, aggregate):
    tensor = SimpleNamespace(contiguous_temporal_rows=lambda: factor)
    port = SimpleNamespace(
        is_input=True,
        connect_interface=[object()],
        size=[1, 1, rows, 8],
        fifo=SimpleNamespace(data_shape=[1, 1, rows, 8]),
    )
    tasks = [
        CodeGenerator.GlobalIODMATask(
            token="other" if variant == "token" and i == 1 else "token",
            start_time=i,
            end_time=i + 1,
            io_port=port,
            dtensor=tensor,
            size=[1, 1, rows, 8],
            offset=[i if variant != "order" else factor - i - 1, 0, 0, 0],
            stride=[rows * 64, 8, 64, 1],
        )
        for i in range(factor)
    ]
    generator = SimpleNamespace(global_dma_trough_port=list(tasks))
    CodeGenerator.coalesce_temporal_input_rows(generator, [port])
    result = generator.global_dma_trough_port
    assert len(result) == (1 if aggregate else factor)
    assert (
        port.size
        == port.fifo.data_shape
        == [1, 1, rows * factor if aggregate else rows, 8]
    )
    if not aggregate:
        assert all(a is b for a, b in zip(tasks, result))
    else:
        assert result[0].start_time == 0
        assert result[0].end_time == factor

    def addresses(sequence):
        arrays = []
        for task in sequence:
            base = sum(o * s for o, s in zip(task.offset, task.stride))
            indices = np.indices(task.size)
            arrays.append(
                (base + sum(indices[i] * task.stride[i] for i in range(4))).ravel()
            )
        return np.concatenate(arrays)

    np.testing.assert_array_equal(addresses(result), addresses(tasks))
