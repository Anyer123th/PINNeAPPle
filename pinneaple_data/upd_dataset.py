from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import os

from .physical_sample import PhysicalSample
from .validators import validate_physical_sample, assert_valid_physical_sample
from .serialization import (
    save_pt, load_pt,
    save_zarr, load_zarr,
    save_hdf5, load_hdf5,
    save_manifest, load_manifest
)


@dataclass
class UPDDataset:
    """
    Hardened dataset wrapper.

    Features:
      - schema version in manifest
      - validation pass
      - save/load (pt, zarr, hdf5)
    """
    samples: List[PhysicalSample]
    manifest: Dict[str, Any] | None = None

    def __post_init__(self):
        self.manifest = dict(self.manifest or {})
        self.manifest.setdefault("upd_version", "0.1")
        self.manifest.setdefault("count", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PhysicalSample:
        return self.samples[idx]

    def validate(
        self,
        *,
        units_policy: str = "warn",
        required_units: Optional[Sequence[str]] = None,
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        non_negative: Optional[Sequence[str]] = None,
        monotonic_dims: Optional[Sequence[str]] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, Any]:
        total_errors = 0
        total_warnings = 0
        by_sample: List[Dict[str, Any]] = []

        for s in self.samples:
            issues = validate_physical_sample(
                s,
                units_policy=units_policy,
                required_units=required_units,
                ranges=ranges,
                non_negative=non_negative,
                monotonic_dims=monotonic_dims,
            )
            e = sum(1 for i in issues if i.level == "error")
            w = sum(1 for i in issues if i.level == "warning")
            total_errors += e
            total_warnings += w
            if issues:
                by_sample.append(
                    {
                        "sample_id": s.sample_id,
                        "errors": [i.message for i in issues if i.level == "error"],
                        "warnings": [i.message for i in issues if i.level == "warning"],
                    }
                )

        report = {
            "count": len(self.samples),
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "samples": by_sample[:50],  # cap
        }

        if raise_on_error and total_errors > 0:
            raise ValueError(f"UPDDataset validation failed: {total_errors} errors. (See report['samples'].)")

        return report

    # -------------------------
    # Persistence
    # -------------------------
    def save(self, path: str, *, format: str = "pt") -> None:
        fmt = format.lower()
        os.makedirs(path if fmt in ("zarr",) else os.path.dirname(path) or ".", exist_ok=True)

        if fmt == "pt":
            save_pt(self.samples, path)
            save_manifest(path + ".manifest.json", self.manifest)
            return

        if fmt == "zarr":
            save_zarr(self.samples, path)
            save_manifest(os.path.join(path, "manifest.json"), self.manifest)
            return

        if fmt in ("hdf5", "h5"):
            save_hdf5(self.samples, path)
            save_manifest(path + ".manifest.json", self.manifest)
            return

        raise ValueError(f"Unknown format '{format}'. Use: pt|zarr|hdf5")

    @staticmethod
    def load(path: str, *, format: str = "pt") -> "UPDDataset":
        fmt = format.lower()
        if fmt == "pt":
            samples = load_pt(path)
            man_path = path + ".manifest.json"
            manifest = load_manifest(man_path) if os.path.exists(man_path) else {"upd_version": "0.1", "count": len(samples)}
            return UPDDataset(samples=samples, manifest=manifest)

        if fmt == "zarr":
            samples = load_zarr(path)
            man_path = os.path.join(path, "manifest.json")
            manifest = load_manifest(man_path) if os.path.exists(man_path) else {"upd_version": "0.1", "count": len(samples)}
            return UPDDataset(samples=samples, manifest=manifest)

        if fmt in ("hdf5", "h5"):
            samples = load_hdf5(path)
            man_path = path + ".manifest.json"
            manifest = load_manifest(man_path) if os.path.exists(man_path) else {"upd_version": "0.1", "count": len(samples)}
            return UPDDataset(samples=samples, manifest=manifest)

        raise ValueError(f"Unknown format '{format}'. Use: pt|zarr|hdf5")
