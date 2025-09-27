"""Compatibility shim exposing LiveDetector under the expected module name."""

from .live import LiveDetector

__all__ = ["LiveDetector"]
