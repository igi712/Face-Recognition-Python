#!/usr/bin/env python3
"""Compatibility shim for the face recognition CLI entry point."""

from app.cli import FaceRecognitionSystem, main

__all__ = ["FaceRecognitionSystem", "main"]


if __name__ == "__main__":
    main()