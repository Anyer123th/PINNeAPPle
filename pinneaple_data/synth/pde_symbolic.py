from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import math
import torch

from .base import SynthConfig, SynthOutput
from .sample_adapter import to_physical_sample


def _safe_import_sympy():
    import sympy  # type: ignore
    return sympy


def _fd_u_x(u: torch.Tensor, dx: float, bc: str) -> torch.Tensor:
    if bc == "periodic":
        return (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * dx)
    # dirichlet0: clamp at boundaries
    up = u.clone()
    up[..., 0] = 0.0
    up[..., -1] = 0.0
    ux = torch.zeros_like(up)
    ux[..., 1:-1] = (up[..., 2:] - up[..., :-2]) / (2.0 * dx)
    return ux


def _fd_u_xx(u: torch.Tensor, dx: float, bc: str) -> torch.Tensor:
    if bc == "periodic":
        return (torch.roll(u, -1, dims=-1) - 2.0 * u + torch.roll(u, 1, dims=-1)) / (dx * dx)
    up = u.clone()
    up[..., 0] = 0.0
    up[..., -1] = 0.0
    uxx = torch.zeros_like(up)
    uxx[..., 1:-1] = (up[..., 2:] - 2.0 * up[..., 1:-1] + up[..., :-2]) / (dx * dx)
    return uxx


def _build_rhs_from_equation(
    equation: str,
    *,
    parameters: Optional[Dict[str, float]] = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
    """
    Returns rhs(u, t, x, derivs) = u_t as tensor.
    Supported symbolic tokens:
      - u, u_t, u_x, u_xx
      - t, x
      - named params (alpha, c, nu, etc.) from `parameters`
      - basic funcs: sin, cos, exp, pi
    """
    sympy = _safe_import_sympy()

    # symbols
    t, x = sympy.symbols("t x")
    u = sympy.Symbol("u")
    u_t = sympy.Symbol("u_t")
    u_x = sympy.Symbol("u_x")
    u_xx = sympy.Symbol("u_xx")

    params = parameters or {}
    p_syms = {k: sympy.Symbol(k) for k in params.keys()}

    namespace = {
        "t": t,
        "x": x,
        "u": u,
        "u_t": u_t,
        "u_x": u_x,
        "u_xx": u_xx,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "exp": sympy.exp,
        "pi": sympy.pi,
        **p_syms,
    }

    expr = sympy.sympify(equation, locals=namespace)

    # Solve for u_t (must be present)
    sol = sympy.solve(sympy.Eq(expr, 0), u_t, dict=True)
    if not sol:
        # maybe equation already written as u_t - RHS
        # try isolate by move terms if u_t appears linearly
        try:
            rhs_expr = sympy.solve(sympy.Eq(u_t, sympy.simplify(u_t - expr)), u_t)
        except Exception:
            rhs_expr = None
        if rhs_expr is None:
            raise ValueError("Could not solve equation for u_t. Ensure equation contains 'u_t' and is solvable.")
    else:
        rhs_expr = sol[0][u_t]

    # Lambdify: rhs(t,x,u,u_x,u_xx, params...)
    arg_list = [t, x, u, u_x, u_xx] + [p_syms[k] for k in params.keys()]
    rhs_fn = sympy.lambdify(arg_list, rhs_expr, "torch")

    def rhs(u_tens: torch.Tensor, t_tens: torch.Tensor, x_tens: torch.Tensor, derivs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # broadcast: u is (B,nx) or (nx,)
        ux = derivs["u_x"]
        uxx = derivs["u_xx"]
        p_vals = [torch.tensor(float(params[k]), device=u_tens.device, dtype=u_tens.dtype) for k in params.keys()]
        return rhs_fn(t_tens, x_tens, u_tens, ux, uxx, *p_vals)

    return rhs


def make_fd_residual_fn(
    equation: str,
    *,
    parameters: Optional[Dict[str, float]] = None,
    bc: str = "periodic",
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float, float], torch.Tensor]:
    """
    Returns residual(u, t, x, dt, dx) -> tensor residual over (T,X)

    u: (T, X) or (B, T, X)
    t: (T,)
    x: (X,)
    """
    sympy = _safe_import_sympy()

    # residual = equation expr with u_t,u_x,u_xx substituted by FD approximations
    # We will compute u_t with forward diff (time), u_x/u_xx with central diff (space)
    # and evaluate original expression (not isolated form).

    # symbols
    t_s, x_s = sympy.symbols("t x")
    u_s = sympy.Symbol("u")
    u_t_s = sympy.Symbol("u_t")
    u_x_s = sympy.Symbol("u_x")
    u_xx_s = sympy.Symbol("u_xx")

    params = parameters or {}
    p_syms = {k: sympy.Symbol(k) for k in params.keys()}
    namespace = {
        "t": t_s,
        "x": x_s,
        "u": u_s,
        "u_t": u_t_s,
        "u_x": u_x_s,
        "u_xx": u_xx_s,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "exp": sympy.exp,
        "pi": sympy.pi,
        **p_syms,
    }
    expr = sympy.sympify(equation, locals=namespace)

    arg_list = [t_s, x_s, u_s, u_t_s, u_x_s, u_xx_s] + [p_syms[k] for k in params.keys()]
    expr_fn = sympy.lambdify(arg_list, expr, "torch")

    def residual(u: torch.Tensor, t: torch.Tensor, x: torch.Tensor, dt: float, dx: float) -> torch.Tensor:
        # u shape: (T,X) or (B,T,X)
        if u.ndim == 2:
            u_ = u[None, ...]
        else:
            u_ = u
        B, T, X = u_.shape

        # FD time derivative (forward diff) on t dimension
        u_t = torch.zeros_like(u_)
        u_t[:, :-1, :] = (u_[:, 1:, :] - u_[:, :-1, :]) / float(dt)
        u_t[:, -1, :] = u_t[:, -2, :]  # copy last

        # FD space derivatives
        u_x = _fd_u_x(u_, float(dx), bc)
        u_xx = _fd_u_xx(u_, float(dx), bc)

        # broadcast t,x to (B,T,X)
        tt = t.view(1, T, 1).expand(B, T, X)
        xx = x.view(1, 1, X).expand(B, T, X)

        p_vals = [torch.tensor(float(params[k]), device=u.device, dtype=u.dtype) for k in params.keys()]
        out = expr_fn(tt, xx, u_, u_t, u_x, u_xx, *p_vals)

        return out[0] if u.ndim == 2 else out

    return residual


class SymbolicFDSynthGenerator:
    """
    Symbolic equation -> automatic finite difference operator -> synthetic dataset.

    Inputs:
      equation: string residual form, e.g. "u_t - alpha*u_xx"
      parameters: dict with alpha,c,nu,... numeric
      ic_fn: function(x, rng)->u0(x)

    Output:
      PhysicalSample (if available) or fallback
      fields:
        - "u": (T+1, X)
      coords:
        - "t": (T+1,)
        - "x": (X,)
      meta:
        - equation, parameters, dt, dx, bc
      extras:
        - rhs_fn (callable)
        - residual_fn (callable)
    """
    def __init__(self, cfg: Optional[SynthConfig] = None):
        self.cfg = cfg or SynthConfig()

    def _rng(self):
        return torch.Generator(device="cpu").manual_seed(int(self.cfg.seed))

    def generate(
        self,
        *,
        equation: str,
        parameters: Optional[Dict[str, float]] = None,
        n_samples: int = 8,
        dt: float = 1e-3,
        steps: int = 200,
        x_min: float = 0.0,
        x_max: float = 1.0,
        nx: int = 128,
        bc: str = "periodic",   # "periodic" | "dirichlet0"
        ic_fn: Optional[Callable[[torch.Tensor, torch.Generator], torch.Tensor]] = None,
        store_residual: bool = False,
    ) -> SynthOutput:
        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)
        rng = self._rng()

        params = parameters or {}

        x = torch.linspace(x_min, x_max, int(nx), device=device, dtype=dtype)
        dx = (x_max - x_min) / max(nx - 1, 1)

        # default IC: random Fourier series
        def default_ic(x_: torch.Tensor, g: torch.Generator) -> torch.Tensor:
            amps = torch.randn((4,), generator=g, device=x_.device, dtype=x_.dtype) * 0.3
            freqs = torch.randint(1, 6, (4,), generator=g, device=x_.device)
            u0 = torch.zeros_like(x_)
            for a, f in zip(amps, freqs):
                u0 = u0 + a * torch.sin(2 * math.pi * f * (x_ - x_min) / (x_max - x_min))
            return u0

        ic = ic_fn or default_ic

        rhs_fn = _build_rhs_from_equation(equation, parameters=params)
        residual_fn = make_fd_residual_fn(equation, parameters=params, bc=bc)

        samples = []
        residuals = []

        for _ in range(int(n_samples)):
            u = ic(x, rng)  # (X,)
            traj = [u]

            # time stepping
            for s in range(int(steps)):
                # derivs for RHS
                ux = _fd_u_x(u[None, :], float(dx), bc)[0]
                uxx = _fd_u_xx(u[None, :], float(dx), bc)[0]

                t_now = torch.tensor(float(s) * float(dt), device=device, dtype=dtype)
                # x as vector; rhs expects tensors broadcasting ok
                du = rhs_fn(u, t_now, x, {"u_x": ux, "u_xx": uxx})
                u = u + float(dt) * du
                traj.append(u)

            U = torch.stack(traj, dim=0)  # (T+1, X)
            t = torch.linspace(0.0, float(dt) * float(steps), steps + 1, device=device, dtype=dtype)

            if store_residual:
                r = residual_fn(U, t, x, float(dt), float(dx))
                residuals.append(r.detach())

            ps_like = {
                "fields": {"u": U},
                "coords": {"t": t, "x": x},
                "meta": {
                    "equation": equation,
                    "parameters": dict(params),
                    "dt": float(dt),
                    "dx": float(dx),
                    "bc": bc,
                    "generator": "SymbolicFDSynthGenerator",
                },
            }
            samples.append(to_physical_sample(ps_like, device=device, dtype=dtype))

        extras: Dict[str, Any] = {
            "rhs_fn": rhs_fn,
            "residual_fn": residual_fn,
            "equation": equation,
            "parameters": dict(params),
        }
        if store_residual and residuals:
            extras["residuals"] = residuals

        return SynthOutput(samples=samples, extras=extras)
