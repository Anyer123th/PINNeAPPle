from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

Tensor = torch.Tensor


def _stack_or_cat(xs: List[Tensor], dim: int = 0) -> Tensor:
    """
    Tries to stack if shapes match perfectly; otherwise concatenates on dim.
    """
    if not xs:
        raise ValueError("Empty tensor list")
    shape0 = tuple(xs[0].shape)
    if all(tuple(x.shape) == shape0 for x in xs):
        return torch.stack(xs, dim=dim)
    return torch.cat(xs, dim=dim)


def _collate_tuple_of_tensors(items: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
    """
    Collate a list of tuples: [(t,x), (t,x), ...] -> (T, X) as concatenated tensors.
    Assumes each element tensor is (N_i, 1) or (N_i, d).
    We concat along batch dimension (dim=0).
    """
    if not items:
        raise ValueError("No items to collate")
    k = len(items[0])
    if any(len(it) != k for it in items):
        raise ValueError("Inconsistent tuple lengths in collate")

    outs: List[Tensor] = []
    for j in range(k):
        outs.append(torch.cat([it[j] for it in items], dim=0))
    return tuple(outs)


def collate_pinn_batches(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function compatible with torch DataLoader for PINNFactory batches.

    Each batch dict may contain:
      - "collocation": Tuple[Tensor,...]
      - "conditions": List[Tuple[Tensor,...]]
      - "data": (Tuple[Tensor,...], Tensor)
      - "meta": dict

    Collation rules:
      - collocation: concat along dim=0 for each input tensor
      - conditions: for each condition index, concat along dim=0
      - data: concat inputs along dim=0, and y_true along dim=0
      - meta: keep a list of metas (do not merge)
    """
    out: Dict[str, Any] = {}
    if not batches:
        return out

    # collocation
    colls = [b.get("collocation") for b in batches if b.get("collocation") is not None]
    if colls:
        out["collocation"] = _collate_tuple_of_tensors(colls)

    # conditions
    conds = [b.get("conditions") for b in batches if b.get("conditions") is not None]
    if conds:
        # conds: list of lists
        n_cond = len(conds[0])
        if any(len(c) != n_cond for c in conds):
            raise ValueError("Inconsistent number of conditions across batches")

        merged: List[Tuple[Tensor, ...]] = []
        for i in range(n_cond):
            ith = [c[i] for c in conds]  # list of tuples
            merged.append(_collate_tuple_of_tensors(ith))
        out["conditions"] = merged

    # data
    datas = [b.get("data") for b in batches if b.get("data") is not None]
    if datas:
        xs = [d[0] for d in datas]
        ys = [d[1] for d in datas]
        out_x = _collate_tuple_of_tensors(xs)
        out_y = torch.cat(ys, dim=0)
        out["data"] = (out_x, out_y)

    # meta
    metas = [b.get("meta") for b in batches if b.get("meta") is not None]
    if metas:
        out["meta"] = metas

    return out


def move_batch_to_device(batch: Dict[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    """
    Moves all tensors in a collated batch to device.
    Leaves meta unchanged.
    """
    dev = torch.device(device)

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to(dev)
        return x

    out: Dict[str, Any] = dict(batch)

    if "collocation" in out and out["collocation"] is not None:
        out["collocation"] = tuple(to_dev(t) for t in out["collocation"])

    if "conditions" in out and out["conditions"] is not None:
        out["conditions"] = [tuple(to_dev(t) for t in cond) for cond in out["conditions"]]

    if "data" in out and out["data"] is not None:
        x, y = out["data"]
        out["data"] = (tuple(to_dev(t) for t in x), to_dev(y))

    return out

def collate_upd_supervised(samples: List[Any]) -> Dict[str, Any]:
    """
    Collate for UPD PhysicalSample supervision batches.

    Input: list[PhysicalSample] (or objects with .state dict)
      each sample.state must have:
        - "x": Tensor (T,D) or (D,) or (T,D,...) (we keep as-is and stack)
        - optional "y": Tensor (T,Do) or (Do,)

    Output dict:
      - "x": Tensor stacked on batch dim -> (B, ...)
      - optional "y": Tensor stacked on batch dim
      - "meta": list[dict] with provenance/domain/schema summary (best-effort)
      - "samples": original samples (optional debug)
    """
    if not samples:
        return {}

    xs: List[Tensor] = []
    ys: List[Tensor] = []
    metas: List[Dict[str, Any]] = []

    for s in samples:
        # PhysicalSample has .state dict or xr.Dataset; MVP expects dict
        st = getattr(s, "state", None)
        if st is None or not isinstance(st, dict):
            raise TypeError("collate_upd_supervised expects PhysicalSample.state to be a dict with 'x'/'y' tensors.")

        if "x" not in st:
            raise KeyError("PhysicalSample.state missing key 'x'")

        x = st["x"]
        if not isinstance(x, torch.Tensor):
            raise TypeError("PhysicalSample.state['x'] must be a torch.Tensor")
        xs.append(x)

        if "y" in st and st["y"] is not None:
            y = st["y"]
            if not isinstance(y, torch.Tensor):
                raise TypeError("PhysicalSample.state['y'] must be a torch.Tensor")
            ys.append(y)

        metas.append(
            {
                "provenance": getattr(s, "provenance", {}) or {},
                "domain": getattr(s, "domain", {}) or {},
                "schema": getattr(s, "schema", {}) or {},
            }
        )

    out: Dict[str, Any] = {
        "x": _stack_or_cat(xs, dim=0),  # (B,T,D) if shapes match
        "meta": metas,
    }
    if ys:
        out["y"] = _stack_or_cat(ys, dim=0)

    return out
