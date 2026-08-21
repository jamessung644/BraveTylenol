"""Compatibility import for the speed-first baseline."""

from app.fast_harness import FastL2Harness, L2ResponseError, L2TimeoutError

L2Harness = FastL2Harness

__all__ = ["L2Harness", "L2ResponseError", "L2TimeoutError"]
