"""
GpuAnalyticalPlacer

Section-0 lock-in scaffold:
- Keeps evaluator compatibility by exposing place(self, benchmark).
- Adds __call__ delegating to place for convenience.
- Encodes the repository's working PLC net-member extraction pattern:
    for driver, sinks in plc.nets.items():
        for pin_name in [driver] + sinks:
            parent = pin_name.split("/")[0]
- Encodes official proxy-cost call pattern:
    compute_proxy_cost(placement, benchmark, plc)

Later sections (objective, legalization, SA, etc.) are implemented incrementally.
"""

from __future__ import annotations

import os
from typing import Iterable, List

import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def _iter_plc_member_names(plc) -> Iterable[tuple[str, List[str]]]:
    """
    Repository-locked net iteration pattern used by loader/objective pipeline.
    """
    for driver, sinks in plc.nets.items():
        yield driver, sinks


def _net_macro_member_indices(driver: str, sinks: List[str], name_to_idx: dict[str, int]) -> List[int]:
    """
    Repository-locked member extraction pattern:
    - Iterate [driver] + sinks
    - Parse parent with pin_name.split('/')[0]
    - Keep only names that map to benchmark macro tensor indices.
    """
    members: List[int] = []
    for pin_name in [driver] + sinks:
        parent = pin_name.split("/")[0]
        if parent in name_to_idx:
            members.append(name_to_idx[parent])
    return members


def _official_proxy_cost(placement: torch.Tensor, benchmark: Benchmark, plc) -> dict:
    """
    Official evaluator call pattern lock-in from macro_place/objective.py.
    """
    return compute_proxy_cost(placement, benchmark, plc)


class GpuAnalyticalPlacer:
    """
    Analytical placer skeleton.

    Notes:
    - Evaluator requires place(self, benchmark) -> Tensor.
    - __call__ delegates to place.
    - Full optimization pipeline is added in later implementation steps.
    """

    def __init__(self):
        self._cache = {}

    def __call__(self, benchmark: Benchmark) -> torch.Tensor:
        return self.place(benchmark)

    def _load_plc(self, benchmark: Benchmark):
        """
        Keep path search order fixed for later full implementation.
        """
        candidates = [
            os.path.join("external", "MacroPlacement", "Testcases", "ICCAD04", benchmark.name),
            os.path.join("external", "MacroPlacement", "Flows", "NanGate45", benchmark.name),
            os.path.join(
                "external",
                "MacroPlacement",
                "Flows",
                "NanGate45",
                benchmark.name.split("_ng45")[0],
            ),
            os.path.join(
                "external",
                "MacroPlacement",
                "Flows",
                "NanGate45",
                benchmark.name.split("_")[0],
            ),
        ]
        attempted = []
        for path in candidates:
            attempted.append(path)
            try:
                _, plc = load_benchmark_from_dir(path)
                return plc
            except Exception:
                continue
        raise RuntimeError(f"Failed to load plc for {benchmark.name}. Attempted: {attempted}")

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Section-0 compatibility baseline.
        Returns a clone of benchmark positions while full pipeline is pending.
        """
        return benchmark.macro_positions.clone()


placer = GpuAnalyticalPlacer()
