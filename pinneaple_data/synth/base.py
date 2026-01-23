from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, List


@dataclass
class SynthConfig:
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class SynthOutput:
    samples: List[Any]
    extras: Dict[str, Any]


class SynthGenerator(Protocol):
    def generate(self, **kwargs) -> SynthOutput: ...
