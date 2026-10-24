# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency accounting when chaining large virtual task graphs."""
from collections import defaultdict

import pytest

pytest.importorskip("aie")
from allo.backend.aie.mapping import ComputationGraph


def test_chained_dependency_users():
    graph = ComputationGraph.__new__(ComputationGraph)
    graph.dependencies = {
        name: defaultdict(int, deps)
        for name, deps in {
            "a": {"input": 2},
            "b": {"a": 1, "input": 3},
            "c": {"b": 4},
            "observer": {"a": 7, "b": 8},
        }.items()
    }
    graph._dependency_users = None
    graph._chain_dependencies("a", "b", "ab")
    assert dict(graph.dependencies["ab"]) == {"input": 5, "ab": 1}
    assert dict(graph.dependencies["c"]) == {"ab": 4}
    # Preserve the existing behavior when both replaced keys occur.
    assert dict(graph.dependencies["observer"]) == {"ab": 8}
    graph._chain_dependencies("ab", "c", "abc")
    assert dict(graph.dependencies["abc"]) == {"input": 5, "abc": 5}
    assert dict(graph.dependencies["observer"]) == {"abc": 8}
    assert graph._dependency_users["input"] == {"abc"}
    assert graph._dependency_users["abc"] == {"abc", "observer"}


def test_chaining_uses_cached_dependency_index():
    class NoFullScan(dict):
        def items(self):
            raise AssertionError("repeated full-graph scan")

    graph = ComputationGraph.__new__(ComputationGraph)
    graph.dependencies = {
        name: defaultdict(int, deps)
        for name, deps in {"a": {}, "b": {"a": 1}, "c": {"b": 1}}.items()
    }
    graph._dependency_users = None
    graph._chain_dependencies("a", "b", "ab")
    graph.dependencies = NoFullScan(graph.dependencies)
    graph._chain_dependencies("ab", "c", "abc")
    assert dict(graph.dependencies["abc"]) == {"abc": 2}
