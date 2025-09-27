"""Shared detector data structures mirroring Jetson Nano classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FaceObject:
    """Lightweight representation of a detected face."""

    rect: Optional[Tuple[int, int, int, int]] = None
    landmark: List[Tuple[int, int]] = field(default_factory=list)
    face_prob: float = 0.0
    name_prob: float = 0.0
    live_prob: float = 0.0
    angle: float = 0.0
    name_index: int = -1
    color: int = 0
    feature: Optional[Tuple[float, ...]] = None
