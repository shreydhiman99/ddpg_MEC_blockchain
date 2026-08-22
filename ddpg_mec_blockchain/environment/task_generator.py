"""Utility helpers for generating task and server parameter sets."""

import numpy as np
from config import (
    DATA_SIZE_RANGE, MAX_DATA_SIZE_SERVER, UNIT_PRICE_RANGE,
    TIME_CONSTRAINT_RANGE, CPU_FREQ_RANGE, TRANSMISSION_POWER_RANGE,
    CHANNEL_GAIN_RANGE, BANDWIDTH_RANGE, CPU_ARCH_PARAM, CPU_CYCLES_PER_UNIT
)


def generate_servers(num_servers, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return {
        "max_cpu_cycles": rng.uniform(*MAX_DATA_SIZE_SERVER, size=num_servers),
        "cpu_freq": rng.uniform(*CPU_FREQ_RANGE, size=num_servers),
        "bandwidth": rng.uniform(*BANDWIDTH_RANGE, size=num_servers),
        "tx_power": rng.uniform(*TRANSMISSION_POWER_RANGE, size=num_servers),
        "channel_gain": rng.uniform(*CHANNEL_GAIN_RANGE, size=num_servers),
        "cpu_arch_param": np.full(num_servers, CPU_ARCH_PARAM),
        "cpu_cycles_per_unit": np.full(num_servers, CPU_CYCLES_PER_UNIT),
        "unit_price": rng.uniform(*UNIT_PRICE_RANGE, size=num_servers),
    }


def generate_tasks(num_tasks, num_servers, servers, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    tasks = {
        "unit_price": rng.uniform(*UNIT_PRICE_RANGE, size=num_tasks),
        "data_size": rng.uniform(*DATA_SIZE_RANGE, size=num_tasks),
        "time_constraint": rng.uniform(*TIME_CONSTRAINT_RANGE, size=num_tasks),
        "submitting_server": rng.integers(0, num_servers, size=num_tasks),
    }
    tasks["cpu_cycles_needed"] = (
        tasks["data_size"].reshape(1, -1) *
        servers["cpu_cycles_per_unit"].reshape(-1, 1)
    )
    return tasks
