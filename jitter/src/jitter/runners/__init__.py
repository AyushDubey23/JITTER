"""Execution runners package for Jitter."""

from jitter.runners.base import BaseRunner
from jitter.runners.desktop_onnx import DesktopOnnxRunner
from jitter.runners.synthetic import SyntheticRunner

__all__ = [
    "BaseRunner",
    "SyntheticRunner",
    "DesktopOnnxRunner",
]
