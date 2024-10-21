import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch


class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.benchmarks = {}

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        try:
            start_time = time()
            yield
        finally:
            end_time = time()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    def store(self, tag, x):
        if tag not in self.benchmarks.keys():
            self.benchmarks[tag] = []
        self.benchmarks[tag].append(x)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)
    
    def dump_stats(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        new_benchmarks = self.benchmarks.copy()
        for tag in self.benchmarks.keys():
            new_benchmarks[tag+'_avg'] = np.mean(self.benchmarks[tag])
        with path.open("w") as f:
            json.dump(new_benchmarks, f)
        self.benchmarks = new_benchmarks

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
