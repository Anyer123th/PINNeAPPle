import os
import torch
from torch.utils.data import DataLoader

from pinneaple_data.physical_sample import PhysicalSample
from pinneaple_data.zarr_store import UPDZarrStore
from pinneaple_data.zarr_iterable import ZarrUPDIterable
from pinneaple_data.collate import collate_upd_supervised

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import default_metrics
from pinneaple_train.preprocess import PreprocessPipeline, SolverFeatureStep
from pinneaple_solvers.fft import FFTSolver

# ----------------------------
# 1) Build a tiny UPD dataset and write to Zarr
# ----------------------------
out_dir = "examples/_out/end_to_end_01"
os.makedirs(out_dir, exist_ok=True)
zarr_path = os.path.join(out_dir, "toy_ts.zarr")

if not os.path.isdir(zarr_path):
    samples = []
    for i in range(256):
        x = torch.randn(64, 8)   # (T,D)
        y = torch.randn(64, 2)   # (T,2)
        samples.append(
            PhysicalSample(
                state={"x": x, "y": y},
                domain={"type": "grid"},
                provenance={"i": i, "source": "toy"},
                schema={"units": {"x": "arb", "y": "arb"}},
            )
        )
    UPDZarrStore.write(zarr_path, samples, manifest={"name": "toy_ts"})

# ----------------------------
# 2) Stream from Zarr with workers
# ----------------------------
ds = ZarrUPDIterable(zarr_path, fields=["x", "y"], coords=[])

dl = DataLoader(
    ds,
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    collate_fn=collate_upd_supervised,  # <-- FIX
)

train_loader = dl
val_loader = dl

# ----------------------------
# 3) Model
# ----------------------------
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8 + 1, 64),  # +1 FFT feature
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (B,T,D)
        return self.net(x)

model = M()

# ----------------------------
# 4) Preprocess: add solver-derived FFT feature to x
# ----------------------------
fft_solver = FFTSolver()  # precisa retornar objeto com .result
preprocess = PreprocessPipeline(
    steps=[
        SolverFeatureStep(
            solver=fft_solver,
            mode="append",
            select_var_dim=None,
            reduce_fft_to="magnitude",
        )
    ]
)

# ----------------------------
# 5) Loss and Trainer
# ----------------------------
combined = CombinedLoss(
    supervised=SupervisedLoss("mse"),
    physics=None,
    w_supervised=1.0,
    w_physics=0.0,
)

def loss_fn(model, y_hat, batch):
    return combined(model, y_hat, batch)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    metrics=default_metrics(),
    preprocess=preprocess,
)

cfg = TrainConfig(
    epochs=2,
    lr=1e-3,
    device="cpu",
    log_dir=os.path.join(out_dir, "_runs"),
    run_name="demo_ts_fft",
    seed=123,
    deterministic=False,
    save_best=True,
)

out = trainer.fit(train_loader, val_loader, cfg)
print("best_val:", out["best_val"])
print("best_path:", out["best_path"])
