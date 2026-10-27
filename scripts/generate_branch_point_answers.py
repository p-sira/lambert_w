# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Regenerate tests/branch_point.csv with uv run scripts/generate_branch_point_answers.py.

Evaluate exact binary inputs with mpmath at 100 and 180 decimal digits and
require identical rounded references. The rounded -1/e endpoints are excluded:
they lie below the real domain and the Rust API deliberately returns -1 there.
"""

import csv
import struct
from pathlib import Path

import mpmath as mp


def f32(x):
    return struct.unpack("!f", struct.pack("!f", x))[0]


def bits(x, single):
    return struct.unpack("!I" if single else "!Q", struct.pack("!f" if single else "!d", x))[0]


def from_bits(x, single):
    return struct.unpack("!f" if single else "!d", struct.pack("!I" if single else "!Q", x))[0]


with (Path(__file__).resolve().parents[1] / "tests/branch_point.csv").open("w", newline="") as output:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["type", "z", "w0", "wm1"])
    for single in [False, True]:
        rounded = f32 if single else float
        with mp.workdps(100):
            endpoint = rounded(-1 / mp.e)
        endpoint_bits = bits(endpoint, single)
        # Consecutive neighbors expose the amplified endpoint rounding error.
        values = {from_bits(endpoint_bits - i, single) for i in range(1, 33)}
        values.update(from_bits(endpoint_bits - 2**i, single) for i in range(5, 19 if single else 43))
        values.update(rounded(endpoint + 10.0**i) for i in range(-16, -1))
        # Include the sci-special series cutoff and the first rational interval boundaries.
        for boundary in [endpoint + 1e-6, -0.3542913309442164, -0.20729377764038414]:
            boundary = rounded(boundary)
            values.update(from_bits(bits(boundary, single) + i, single) for i in [-1, 0, 1])
        values.update(rounded(x) for x in [-0.36, -0.3578794411714423, -0.35, -0.3, -0.1])
        for x in sorted(values):
            if x <= endpoint:
                continue
            with mp.workdps(100):
                expected = [rounded(mp.lambertw(mp.mpf(x), k)) for k in [0, -1]]
            with mp.workdps(180):
                assert expected == [rounded(mp.lambertw(mp.mpf(x), k)) for k in [0, -1]]
            writer.writerow(["f32" if single else "f64", repr(x), *map(repr, expected)])
