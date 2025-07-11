"""Computational profiling utilities."""

import functools
import json
import logging
import time
import tracemalloc
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ComputationReport:
    """Accumulates timing and memory measurements across experiments."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add(self, name: str, wall_time: float, peak_memory_mb: float, **kwargs):
        entry = {"name": name, "wall_time_s": round(wall_time, 2),
                 "peak_memory_mb": round(peak_memory_mb, 1)}
        entry.update(kwargs)
        self.entries.append(entry)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.entries)

    def summary(self) -> str:
        total_time = sum(e["wall_time_s"] for e in self.entries)
        peak_mem = max(e["peak_memory_mb"] for e in self.entries) if self.entries else 0
        return (f"ComputationReport: {len(self.entries)} experiments, "
                f"total={format_time(total_time)}, peak_mem={format_memory(peak_mem * 1e6)}")


# Global report instance
_global_report = ComputationReport()


def timed_experiment(func):
    """Decorator that measures wall time and peak memory."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            wall_time = time.time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 1e6
            logger.info("%s: %s, peak_mem=%s",
                        func.__name__, format_time(wall_time), format_memory(peak))
            _global_report.add(func.__name__, wall_time, peak_mb)
        return result

    return wrapper


def get_global_report() -> ComputationReport:
    return _global_report


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def format_memory(bytes_val: float) -> str:
    if bytes_val < 1e6:
        return f"{bytes_val/1e3:.0f}KB"
    elif bytes_val < 1e9:
        return f"{bytes_val/1e6:.1f}MB"
    else:
        return f"{bytes_val/1e9:.2f}GB"
