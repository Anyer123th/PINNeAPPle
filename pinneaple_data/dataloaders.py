from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from pinneaple_pinn.io import UPDItem, UPDDataset, SamplingSpec, ConditionSpec, PINNMapping

from .collate import collate_pinn_batches
from .physical_sample import PhysicalSample


@dataclass
class DataLoaderSpec:
    """
    Torch DataLoader specification.
    """
    batch_size: int = 1                 # number of shard-items per batch (we often keep 1)
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    drop_last: bool = False


class _UPDShardTorchDataset(Dataset):
    """
    Wraps UPDDataset.sample(spec) into a torch.utils.data.Dataset item generator.

    Each __getitem__ returns a dict in the format expected by collate_pinn_batches.
    """

    def __init__(
        self,
        item: UPDItem,
        mapping: PINNMapping,
        sampling: SamplingSpec,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        length: int = 10_000,
    ):
        self.item = item
        self.mapping = mapping
        self.sampling = sampling
        self.device = device
        self.dtype = dtype
        self.length = int(length)

        self._upd = UPDDataset(item=item, mapping=mapping, device=device, dtype=dtype)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # vary seed per idx for fresh random points
        spec = SamplingSpec(
            n_collocation=self.sampling.n_collocation,
            conditions=self.sampling.conditions,
            n_data=self.sampling.n_data,
            replace=self.sampling.replace,
            seed=int(self.sampling.seed) + int(idx),
        )
        b = self._upd.sample(spec)

        out: Dict[str, Any] = {
            "collocation": b.collocation,
            "conditions": b.conditions,
            "data": b.data,
            "meta": {
                "idx": idx,
                "zarr_path": self.item.zarr_path,
                "meta_path": self.item.meta_path,
            },
        }
        return out


def build_upd_dataloader(
    *,
    zarr_path: str,
    meta_path: str,
    mapping: PINNMapping,
    sampling: SamplingSpec,
    loader: Optional[DataLoaderSpec] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    length: int = 10_000,
) -> DataLoader:
    """
    Build a DataLoader from a single UPD shard (Zarr+JSON).
    """
    loader = loader or DataLoaderSpec()
    item = UPDItem(zarr_path=zarr_path, meta_path=meta_path)
    ds = _UPDShardTorchDataset(
        item=item,
        mapping=mapping,
        sampling=sampling,
        device=device,
        dtype=dtype,
        length=length,
    )
    return DataLoader(
        ds,
        batch_size=loader.batch_size,
        shuffle=loader.shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=collate_pinn_batches,
    )


def build_physical_sample_dataloader(
    sample: PhysicalSample,
    *,
    mapping: PINNMapping,
    sampling: SamplingSpec,
    loader: Optional[DataLoaderSpec] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    length: int = 10_000,
) -> DataLoader:
    """
    Build a DataLoader from a PhysicalSample.

    MVP behavior:
      - If sample is grid (xarray.Dataset), we create a temporary UPDItem-like wrapper
        using in-memory ds/meta form supported by UPDDataset.
      - Mesh samples will be supported later by MeshPhysicalDataset.
    """
    loader = loader or DataLoaderSpec()

    if not sample.is_grid():
        raise NotImplementedError("Mesh dataloader is MVP-2. For now, only grid PhysicalSample is supported.")

    # Create an in-memory UPDInput dict supported by UPDDataset
    upd_input = {"ds": sample.state, "meta": {"schema": sample.schema, "domain": sample.domain, "provenance": sample.provenance}}

    class _MemUPDItem:
        def __init__(self, ds, meta):
            self._ds = ds
            self._meta = meta
            self.zarr_path = "<in-memory>"
            self.meta_path = "<in-memory>"
        def open_dataset(self):
            return self._ds
        def load_meta(self):
            return self._meta

    mem_item = _MemUPDItem(upd_input["ds"], upd_input["meta"])

    # Use the same _UPDShardTorchDataset with the mem item
    ds = _UPDShardTorchDataset(
        item=mem_item,  # type: ignore
        mapping=mapping,
        sampling=sampling,
        device=device,
        dtype=dtype,
        length=length,
    )
    return DataLoader(
        ds,
        batch_size=loader.batch_size,
        shuffle=loader.shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=collate_pinn_batches,
    )
