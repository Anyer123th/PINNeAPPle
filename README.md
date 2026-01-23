# Pinneaple 🍍
**Unified Physical Data, Geometry, Models and Training for Physics AI**

Pinneaple is an open-source Python platform designed to **bridge real physical data, geometry, numerical solvers, and machine learning models** into a single coherent ecosystem for **Physics-Informed AI**.

It is built to serve both **research** and **industrial workflows**, with strong emphasis on:
- Physical consistency
- Scalability
- Auditability
- Interoperability with CFD / CAD / scientific data formats

---

## ✨ Key Features

### 📦 Unified Physical Dataset (UPD)
A standardized abstraction to represent *physical samples*, including:
- Physical state (grids, meshes, graphs)
- Geometry (CAD / mesh)
- Governing equations, ICs, BCs, forcings
- Units, regimes, metadata and provenance

Used consistently across **data loading, training, validation, and inference**.

---

### 🌍 Data & IO (`pinneaple_data`)
- NASA / scientific-ready data pipelines
- Zarr-backed datasets with:
  - Lazy loading
  - Sharding
  - Adaptive prefetch
  - Byte-based LRU caching
- Deterministic shard-aware iterators
- Physical validation and schema enforcement

---

### 📐 Geometry & Mesh (`pinneaple_geom`)
- CAD generation (CadQuery)
- STL / mesh IO (trimesh, meshio, OpenFOAM MVP)
- Mesh repair, remeshing and simplification
- Sampling (points, grids, barycentric)
- Geometry-aware feature extraction

---

### 🧠 Model Zoo (`pinneaple_models`)
A curated catalog of architectures commonly used in Physics AI:

- PINNs (Vanilla, XPINN, VPINN, XTFC, Inverse PINN, PIELM)
- Neural Operators (FNO, DeepONet, PINO, GNO, UNO)
- Graph Neural Networks (GraphCast-style, GNN-ODE, equivariant GNNs)
- Transformers (Informer, FEDformer, Autoformer, TFT)
- Reduced Order Models (POD, DMD, HAVOK, Operator Inference)
- Classical & hybrid models (Kalman, ARIMA, Koopman, ESN)
- Physics-aware & structure-preserving networks

All models are discoverable via a **central registry**.

---

### 🧮 Physics Loss Factory (`pinneaple_pinn`)
- Symbolic PDE definitions (SymPy-based)
- Automatic differentiation graph construction
- PINN-ready residuals and constraints
- Works directly with UPD samples

---

### ⚙️ Solvers (`pinneaple_solvers`)
Numerical solvers and mathematical tools used for:
- Data generation
- Feature extraction
- Validation

Includes:
- FEM / FVM (MVP)
- FFT
- Hilbert–Huang Transform
- Adapters to/from UPD

---

### 🏗️ Synthetic Data Generation (`pinneaple_data.synth`)
Generate datasets from:
- Symbolic PDEs
- Parametric distributions
- Curve fitting from real data
- Images and signals
- Geometry perturbations and CAD parameter sweeps

---

### 🚂 Training & Evaluation (`pinneaple_train`)
- Deterministic, auditable training
- Dataset splitting (train/val/test)
- Preprocessing pipelines & normalizers
- Metrics & visualization
- Physics-aware loss integration
- Reproducible runs (seeds, env fingerprinting)
- Checkpointing & inference utilities

---

## 🚀 Installation

Pinneaple is currently distributed as an open-source research & industry framework directly from GitHub.

1. Clone the repository
```bash
git clone https://github.com/barrosyan/pinneaple.git
cd pinneaple
```

2. Create a virtual environment (strongly recommended)

Python ≥ 3.10 is recommended (3.11 works well; 3.13 may require extra care on Windows).

```bash
python -m venv .venv
```

Activate it:

Linux / macOS

```bash
source .venv/bin/activate
```

Windows (PowerShell)

```bash
.venv\Scripts\Activate.ps1
```

3. Install core dependencies

Install Pinneaple in editable (development) mode:

```bash
pip install -e .
```

This installs:

pinneaple_data

pinneaple_geom

pinneaple_models

pinneaple_pinn

pinneaple_pdb

pinneaple_solvers

pinneaple_train

4. Optional dependencies (recommended)

Pinneaple is modular. Install only what you need:

🔹 Geometry / CAD / Mesh
```bash
pip install trimesh meshio
pip install cadquery  # requires OCC stack
```

⚠️ On Windows, CadQuery is best installed via Conda:

```bash
conda create -n pinneaple-cq python=3.10 cadquery -c conda-forge
conda activate pinneaple-cq
pip install -e .
```

🔹 Scientific & ML stack
```bash
pip install torch numpy scipy sympy
```

Optional (recommended for performance & operators):

```bash
pip install zarr numcodecs
pip install open3d fast-simplification

```
5️. Development & testing tools

For contributors:

```bash
pip install -e ".[dev]"
```

6. Verify installation

Quick smoke test:

```python
from pinneaple_models.register_all import register_all
from pinneaple_models.registry import ModelRegistry

register_all()
print("Registered models:", len(ModelRegistry.list()))
```

🧠 Notes

Pinneaple is not yet released on PyPI — cloning the repo is required.

Some features (CFD, CAD, large-scale Zarr) rely on optional native backends.

All examples in examples/ are runnable after installation.

## 🧪 Quick Example

```python
from pinneaple_data.physical_sample import PhysicalSample
import xarray as xr
import numpy as np

ds = xr.Dataset(
    data_vars=dict(T2M=(("t","x"), np.random.randn(24,16))),
    coords=dict(t=np.arange(24), x=np.arange(16))
)

sample = PhysicalSample(
    state=ds,
    domain={"type": "grid"},
    schema={"governing": "toy"},
)

print(sample.summary())
```

---

## 🧑‍🔬 Who Is This For?

- Physics AI researchers
- CFD / FEA / climate ML teams
- Industrial R&D groups
- Scientific ML practitioners
- Anyone building **surrogates, inverse models, or hybrid solvers**

---

## 🤝 Contributing

We welcome contributions in:
- New datasets & adapters
- Models and solvers
- Benchmarks
- Documentation

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License
Apache 2.0 — see [LICENSE](LICENSE).

---

## 📚 Citation

If you use Pinneaple in research, please cite via `CITATION.cff`.

---

## 🌱 Project Philosophy

Pinneaple is **not** a single model or method.

It is a **platform** — designed to let physical data, geometry, equations and learning systems interact cleanly, reproducibly, and at scale.

> *From raw physics to deployable intelligence.*

---

**Status:** Early but ambitious.  
**Feedback & collaboration welcome.**
