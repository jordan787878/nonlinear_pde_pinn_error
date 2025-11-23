#%%
# main_Burgers1d_error.py
#
# Goal:
#   1) Train primary Burgers solver -> u_hat (fixed afterward)
#   2) Compute PDE residual r of u_hat
#   3) Train nonlinear error PDE for e ≈ u_true - u_hat *without correcting u_hat*
#      using (optionally) adaptive residual-based collocation for e_hat only
#   4) Plot/compare true error vs learned error
#
# Assumes you already added:
#   - src/PDEs.py : class BurgersErrorNonlinear
#   - src/Gram_matrice.py : helper construct_Theta_test_Burgers(..., which=...)
#
# Run e.g.:
#   python main_Burgers1d_error.py --show_figure True

import argparse
import os

os.makedirs("figs", exist_ok=True)

import jax
import jax.numpy as jnp
from jax import vmap, jit
jax.config.update("jax_enable_x64", True)

import numpy as onp
from numpy import random

import matplotlib.pyplot as plt

from src.solver import solver_GP
from src.PDEs import BurgersErrorNonlinear
from src.Gram_matrice import construct_Theta_test_Burgers  # new helper


# ------------------ argparse ------------------
def get_parser():
    parser = argparse.ArgumentParser(
        description="Burgers GP solver + nonlinear error PDE solver (learn e_hat only)"
    )

    # --- pretrained model I/O (npz) ---
    parser.add_argument("--load_uhat", type=str, default=None,
                        help="Path to .npz holding pretrained u_hat GP solution.")
    parser.add_argument("--save_uhat", type=str, default=None,
                        help="Where to save u_hat GP solution as .npz.")

    parser.add_argument("--load_ehat", type=str, default=None,
                        help="Path to .npz holding pretrained e_hat GP solution.")
    parser.add_argument("--save_ehat", type=str, default=None,
                        help="Where to save e_hat GP solution as .npz.")


    # equation parameters
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.02)

    # kernel setting
    parser.add_argument("--kernel", type=str, default="anisotropic_Gaussian")
    parser.add_argument("--kernel_parameter", type=float, nargs="+", default=[0.3, 0.05])
    parser.add_argument("--nugget", type=float, default=1e-5)
    parser.add_argument(
        "--nugget_type",
        type=str,
        default="adaptive",
        choices=["adaptive", "identity", "none"],
    )

    # sampling points for primary u_hat solve
    parser.add_argument("--sampled_type", type=str, default="random", choices=["random", "grid"])
    parser.add_argument("--N_domain", type=int, default=1000)
    parser.add_argument("--N_boundary", type=int, default=200)

    # GN iterations for primary Burgers
    parser.add_argument("--GNsteps", type=int, default=8)
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--initial_sol", type=str, default="rdm")

    # GN iterations for error PDE
    parser.add_argument("--GNsteps_err", type=int, default=6)
    parser.add_argument("--step_size_err", type=float, default=1.0)
    parser.add_argument("--initial_sol_err", type=str, default="rdm")

    # adaptive collocation for error PDE only
    parser.add_argument("--n_adapt", type=int, default=3)
    parser.add_argument("--K_init_resid", type=int, default=400)  # initial |r| enrichment
    parser.add_argument("--K_add", type=int, default=300)         # per-round enrichment
    parser.add_argument("--Ncand_t", type=int, default=250)
    parser.add_argument("--Ncand_x", type=int, default=250)

    # logs and visualization
    parser.add_argument("--print_hist", type=bool, default=True)
    parser.add_argument("--show_figure", type=bool, default=True)

    parser.add_argument("--randomseed", type=int, default=0)
    return parser.parse_args()


def set_random_seeds(args):
    random.seed(args.randomseed)


def save_gp_solution_npz(path, eqn, cfg):
    """Save everything needed to reproduce GP posterior."""
    onp.savez(
        path,
        X_domain=onp.array(eqn.X_domain),
        X_boundary=onp.array(eqn.X_boundary),
        sol_vec=onp.array(eqn.sol_vec),
        kernel=str(cfg.kernel),
        kernel_parameter=onp.array(cfg.kernel_parameter),
        nugget=float(cfg.nugget),
        nugget_type=str(cfg.nugget_type),
    )
    print(f"[Saved] GP solution -> {path}")


def load_gp_solution_npz(path):
    d = dict(onp.load(path, allow_pickle=True))
    # unwrap 0-d arrays / bytes
    for k in ["kernel", "nugget_type"]:
        if k in d:
            d[k] = str(d[k])
    return d


def build_temp_from_loaded(eqn, d):
    """Given eqn + loaded dict, assemble Θ, factorize, and compute temp."""
    eqn.get_sampled_points(d["X_domain"], d["X_boundary"])
    eqn.Gram_matrix(
        kernel=d["kernel"],
        kernel_parameter=d["kernel_parameter"].tolist()
            if hasattr(d["kernel_parameter"], "tolist") else d["kernel_parameter"],
        nugget=float(d["nugget"]),
        nugget_type=d["nugget_type"],
    )
    eqn.Gram_Cholesky()
    eqn.sol_vec = jnp.asarray(d["sol_vec"])
    L = eqn.L
    temp = jnp.linalg.solve(L.T, jnp.linalg.solve(L, eqn.sol_vec))
    return temp



# ------------------ main ------------------
cfg = get_parser()
set_random_seeds(cfg)
print(f"[Seeds] random seeds: {cfg.randomseed}")

alpha = float(cfg.alpha)
nu = float(cfg.nu)

domain_box = onp.array([[0.0, 1.0], [-1.0, 1.0]])  # (t,x) domain


# ============================================================
# 0) PRIMARY Burgers solve -> u_hat (fixed afterward)
# ============================================================
solver = solver_GP(cfg, PDE_type="Burgers")

@jit
def u_bdy(t, x):
    return -jnp.sin(jnp.pi * x) * (t == 0) + 0.0 * (x == 0)

@jit
def f_rhs(t, x):
    return 0.0

solver.set_equation(bdy=u_bdy, rhs=f_rhs, domain=domain_box)

if cfg.load_uhat is not None:
    dU = load_gp_solution_npz(cfg.load_uhat)
    # IMPORTANT: use loaded collocation points + sol_vec
    solver.eqn = solver.eqn  # already Burgers eqn
    temp = build_temp_from_loaded(solver.eqn, dU)
    X_dom = solver.eqn.X_domain
    X_bdy = solver.eqn.X_boundary
    print(f"[Loaded u_hat] from {cfg.load_uhat}  N_dom={X_dom.shape[0]} N_bdy={X_bdy.shape[0]}")
else:
    solver.auto_sample(cfg.N_domain, cfg.N_boundary, sampled_type=cfg.sampled_type)
    if cfg.show_figure:
        solver.show_sample()

    solver.solve()
    if cfg.show_figure:
        solver.show_loss_hist()

    # GP posterior coefficients for u_hat
    L = solver.eqn.L
    sol_vec = solver.eqn.sol_vec
    temp = jnp.linalg.solve(L.T, jnp.linalg.solve(L, sol_vec))

    X_dom = solver.eqn.X_domain
    X_bdy = solver.eqn.X_boundary

    if cfg.save_uhat is not None:
        save_gp_solution_npz(cfg.save_uhat, solver.eqn, cfg)


# ============================================================
# 1) Boundary mismatch of u_hat -> Dirichlet data for error
# ============================================================
Theta_u_bdy = construct_Theta_test_Burgers(
    X_bdy, X_dom, X_bdy,
    kernel=cfg.kernel,
    kernel_parameter=cfg.kernel_parameter,
    which="u",
)
uhat_bdy = Theta_u_bdy @ temp

g_bdy = vmap(u_bdy)(X_bdy[:, 0], X_bdy[:, 1])  # prescribed BC/IC
e_bdy_vals = g_bdy - uhat_bdy                 # mismatch is true boundary error

print("[u_hat boundary mismatch max]:", float(jnp.max(jnp.abs(e_bdy_vals))))


# ============================================================
# 2) Residual r of u_hat on interior points X_dom
# ============================================================
Theta_u   = construct_Theta_test_Burgers(X_dom, X_dom, X_bdy,
                                         kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="u")
Theta_ut  = construct_Theta_test_Burgers(X_dom, X_dom, X_bdy,
                                         kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="ut")
Theta_ux  = construct_Theta_test_Burgers(X_dom, X_dom, X_bdy,
                                         kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="ux")
Theta_uxx = construct_Theta_test_Burgers(X_dom, X_dom, X_bdy,
                                         kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="uxx")

uhat    = Theta_u   @ temp
uhat_t  = Theta_ut  @ temp
uhat_x  = Theta_ux  @ temp
uhat_xx = Theta_uxx @ temp

r_dom = uhat_t + alpha * uhat * uhat_x - nu * uhat_xx  # rhs=0
s_dom = -r_dom


# ============================================================
# 3) Initial residual-based enrichment for error PDE ONLY
#    (optional but usually helpful)
# ============================================================
Ncand_t, Ncand_x = int(cfg.Ncand_t), int(cfg.Ncand_x)
tt_c = jnp.linspace(0.0, 1.0, Ncand_t)
xx_c = jnp.linspace(-1.0, 1.0, Ncand_x)
TTc, XXc = jnp.meshgrid(tt_c, xx_c)
X_cand = jnp.stack([TTc.ravel(), XXc.ravel()], axis=1)

Theta_u_c   = construct_Theta_test_Burgers(X_cand, X_dom, X_bdy,
                                           kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="u")
Theta_ut_c  = construct_Theta_test_Burgers(X_cand, X_dom, X_bdy,
                                           kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="ut")
Theta_ux_c  = construct_Theta_test_Burgers(X_cand, X_dom, X_bdy,
                                           kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="ux")
Theta_uxx_c = construct_Theta_test_Burgers(X_cand, X_dom, X_bdy,
                                           kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="uxx")

uh_c   = Theta_u_c   @ temp
uht_c  = Theta_ut_c  @ temp
uhx_c  = Theta_ux_c  @ temp
uhxx_c = Theta_uxx_c @ temp

r_c = uht_c + alpha * uh_c * uhx_c - nu * uhxx_c
abs_r = jnp.abs(r_c)

K_init = int(cfg.K_init_resid)
idx_top = jnp.argsort(abs_r)[-K_init:]
X_highR = X_cand[idx_top]

# augmented interior set for ERROR PDE only
X_dom_err = jnp.concatenate([X_dom, X_highR], axis=0)


# ============================================================
# 4) Nonlinear error PDE solve with adaptive sampling rounds
# ============================================================
@jit
def e_bdy(t, x):
    d2 = (X_bdy[:, 0] - t) ** 2 + (X_bdy[:, 1] - x) ** 2
    j = jnp.argmin(d2)
    return e_bdy_vals[j]

err_cfg = argparse.Namespace(**vars(cfg))
err_cfg.GNsteps = cfg.GNsteps_err
err_cfg.step_size = cfg.step_size_err
err_cfg.initial_sol = cfg.initial_sol_err

err_solver = solver_GP(err_cfg, PDE_type="Burgers")

if cfg.load_ehat is not None:
    dE = load_gp_solution_npz(cfg.load_ehat)

    # Build a dummy error eqn just to reconstruct Θ + posterior
    err_eqn = BurgersErrorNonlinear(
        alpha=alpha, nu=nu,
        uhat=jnp.zeros((dE["X_domain"].shape[0],)),   # placeholders not used in posterior
        uhat_x=jnp.zeros((dE["X_domain"].shape[0],)),
        source_s=jnp.zeros((dE["X_domain"].shape[0],)),
        bdy=e_bdy, rhs=f_rhs,
        domain=domain_box
    )
    err_solver.eqn = err_eqn
    temp_e = build_temp_from_loaded(err_solver.eqn, dE)

    # overwrite X_dom_err to loaded interior points for any later use
    X_dom_err = err_solver.eqn.X_domain
    print(f"[Loaded e_hat] from {cfg.load_ehat}  N_dom_err={X_dom_err.shape[0]}")

else:
    # ---- your existing adaptive loop ----
    n_adapt = int(cfg.n_adapt)
    K_add   = int(cfg.K_add)

    for k in range(n_adapt):
        print(f"\n[Error adapt round {k+1}/{n_adapt}]  N_int = {X_dom_err.shape[0]}")

        # recompute uhat, uhat_x, s=-r on CURRENT interior set
        Theta_u_err   = construct_Theta_test_Burgers(X_dom_err, X_dom, X_bdy,
                                                     kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="u")
        Theta_ut_err  = construct_Theta_test_Burgers(X_dom_err, X_dom, X_bdy,
                                                     kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="ut")
        Theta_ux_err  = construct_Theta_test_Burgers(X_dom_err, X_dom, X_bdy,
                                                     kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="ux")
        Theta_uxx_err = construct_Theta_test_Burgers(X_dom_err, X_dom, X_bdy,
                                                     kernel=cfg.kernel, kernel_parameter=cfg.kernel_parameter, which="uxx")

        uhat_err   = Theta_u_err   @ temp
        uhat_t_err = Theta_ut_err  @ temp
        uhat_x_err = Theta_ux_err  @ temp
        uhat_xx_err= Theta_uxx_err @ temp

        r_err = uhat_t_err + alpha * uhat_err * uhat_x_err - nu * uhat_xx_err
        s_err = -r_err

        err_eqn = BurgersErrorNonlinear(
            alpha=alpha, nu=nu,
            uhat=uhat_err, uhat_x=uhat_x_err, source_s=s_err,
            bdy=e_bdy, rhs=f_rhs, domain=domain_box
        )
        err_eqn.get_sampled_points(X_dom_err, X_bdy)
        err_solver.eqn = err_eqn

        err_solver.solve()

        # GP posterior coefficients for e_hat on current set
        L_e = err_solver.eqn.L
        sol_vec_e = err_solver.eqn.sol_vec
        temp_e = jnp.linalg.solve(L_e.T, jnp.linalg.solve(L_e, sol_vec_e))

        # evaluate residual indicator on candidates and add points
        Theta_e   = construct_Theta_test_Burgers(X_cand, X_dom_err, X_bdy,
                                                 kernel=err_cfg.kernel, kernel_parameter=err_cfg.kernel_parameter, which="u")
        Theta_et  = construct_Theta_test_Burgers(X_cand, X_dom_err, X_bdy,
                                                 kernel=err_cfg.kernel, kernel_parameter=err_cfg.kernel_parameter, which="ut")
        Theta_ex  = construct_Theta_test_Burgers(X_cand, X_dom_err, X_bdy,
                                                 kernel=err_cfg.kernel, kernel_parameter=err_cfg.kernel_parameter, which="ux")
        Theta_exx = construct_Theta_test_Burgers(X_cand, X_dom_err, X_bdy,
                                                 kernel=err_cfg.kernel, kernel_parameter=err_cfg.kernel_parameter, which="uxx")

        e_c   = Theta_e   @ temp_e
        e_t_c = Theta_et  @ temp_e
        e_x_c = Theta_ex  @ temp_e
        e_xx_c= Theta_exx @ temp_e

        r_c = uht_c + alpha * uh_c * uhx_c - nu * uhxx_c

        R_e = (e_t_c
               + alpha * uh_c  * e_x_c
               + alpha * uhx_c * e_c
               + alpha * e_c   * e_x_c
               - nu * e_xx_c
               + r_c)

        abs_R_e = jnp.abs(R_e)
        idx_top = jnp.argsort(abs_R_e)[-K_add:]
        X_new = X_cand[idx_top]
        X_dom_err = jnp.concatenate([X_dom_err, X_new], axis=0)

    if cfg.save_ehat is not None:
        save_gp_solution_npz(cfg.save_ehat, err_solver.eqn, err_cfg)


# ============================================================
# 5) Test vs truth: compare e_true and e_hat
# ============================================================
[Gauss_pts, weights] = onp.polynomial.hermite.hermgauss(80)

def u_truth(t, x):
    temp2 = x - jnp.sqrt(4 * nu * t) * Gauss_pts
    val1 = weights * jnp.sin(jnp.pi * temp2) * jnp.exp(-jnp.cos(jnp.pi * temp2) / (2 * jnp.pi * nu))
    val2 = weights * jnp.exp(-jnp.cos(jnp.pi * temp2) / (2 * jnp.pi * nu))
    return -jnp.sum(val1) / jnp.sum(val2)

N_pts = 60
tt = jnp.linspace(0.0, 1.0, N_pts)
xx = jnp.linspace(-1.0, 1.0, N_pts)
TT, XX = jnp.meshgrid(tt, xx)
X_test = jnp.concatenate((TT.reshape(-1, 1), XX.reshape(-1, 1)), axis=1)

truth_test = vmap(u_truth)(X_test[:, 0], X_test[:, 1])

solver.test(X_test)
uhat_test = solver.eqn.extended_sol

err_solver.test(X_test)
e_hat_test = err_solver.eqn.extended_sol

e_true_test = truth_test - uhat_test
err_gap = e_true_test - e_hat_test

print("\n[Error PDE quality]")
print("  max |e_true - e_hat|:", float(jnp.max(jnp.abs(err_gap))))
print("  L2  |e_true - e_hat|:", float(jnp.sqrt(jnp.mean(err_gap ** 2))))

if cfg.show_figure:
    # ============================================================
    # (A) Compare u_true vs u_hat with shared colorbar
    # ============================================================
    u_true_grid = truth_test.reshape(TT.shape)
    u_hat_grid  = uhat_test.reshape(TT.shape)

    vmin_u = float(jnp.min(jnp.stack([truth_test, uhat_test])))
    vmax_u = float(jnp.max(jnp.stack([truth_test, uhat_test])))
    levels_u = 50

    fig_u, axes_u = plt.subplots(
        1, 2, figsize=(12, 6),
        sharex=True, sharey=True,
        constrained_layout=True
    )

    cu0 = axes_u[0].contourf(
        TT, XX, u_true_grid,
        levels=levels_u, vmin=vmin_u, vmax=vmax_u
    )
    axes_u[0].set_title(r"True solution  $u_{\mathrm{true}}$")
    axes_u[0].set_xlabel("t")
    axes_u[0].set_ylabel("x")

    cu1 = axes_u[1].contourf(
        TT, XX, u_hat_grid,
        levels=levels_u, vmin=vmin_u, vmax=vmax_u
    )
    axes_u[1].set_title(r"Learned solution  $\hat u$")
    axes_u[1].set_xlabel("t")

    fig_u.colorbar(cu1, ax=axes_u, shrink=0.95, pad=0.02)
    fig_u.savefig("figs/u_true_vs_uhat.png", dpi=300, bbox_inches="tight")

    # ============================================================
    # (B) Compare e_true vs e_hat with shared colorbar (your code)
    # ============================================================
    e_true_grid = e_true_test.reshape(TT.shape)
    e_hat_grid  = e_hat_test.reshape(TT.shape)
    vmin = float(jnp.min(jnp.stack([e_true_test, e_hat_test])))
    vmax = float(jnp.max(jnp.stack([e_true_test, e_hat_test])))
    levels = 50

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    c0 = axes[0].contourf(TT, XX, e_true_grid, levels=levels, vmin=vmin, vmax=vmax)
    axes[0].set_title(r"True error  $e_{\mathrm{true}}=u_{\mathrm{true}}-\hat u$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")

    c1 = axes[1].contourf(TT, XX, e_hat_grid, levels=levels, vmin=vmin, vmax=vmax)
    axes[1].set_title(r"Learned error  $\hat e$")
    axes[1].set_xlabel("t")

    fig.colorbar(c1, ax=axes, shrink=0.95, pad=0.02)
    fig.savefig("figs/e_true_vs_e_hat.png", dpi=300, bbox_inches="tight")

    # ============================================================
    # (C) Parity plot
    # ============================================================
    plt.figure()
    plt.title("Parity plot: e_hat vs e_true")
    plt.scatter(onp.array(e_true_test), onp.array(e_hat_test), s=5, alpha=0.4)
    mn = float(jnp.min(e_true_test))
    mx = float(jnp.max(e_true_test))
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xlabel("e_true")
    plt.ylabel("e_hat")
    plt.axis("equal")

    # ============================================================
    # (D) Time slices
    # ============================================================
    n_times = 6

    t_unique = jnp.unique(X_test[:, 0])
    t_unique = jnp.sort(t_unique)

    if t_unique.size >= n_times:
        idx_t = jnp.linspace(0, t_unique.size - 1, n_times)
        idx_t = jnp.round(idx_t).astype(int)
        times_to_plot = t_unique[idx_t]
    else:
        times_to_plot = jnp.linspace(t_unique[0], t_unique[-1], n_times)

    # --- 3 by 2 layout ---
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6), constrained_layout=True)

    for k, t_sel in enumerate(times_to_plot):
        r, c = divmod(k, ncols)   # k -> (row, col)
        ax = axes[r, c]

        mask = jnp.isclose(X_test[:, 0], t_sel)
        x_sel = X_test[mask, 1]

        idx = jnp.argsort(x_sel)
        x_sel = x_sel[idx]

        e_true_sel = e_true_test[mask][idx]
        e_hat_sel  = e_hat_test[mask][idx]

        ax.plot(x_sel, e_true_sel, label="e_true")
        ax.plot(x_sel, e_hat_sel, "--", label="e_hat")
        ax.set_title(f"t = {float(t_sel):.3f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Error slices vs x at selected times", y=1.02, fontsize=14)
    fig.supxlabel("x")
    fig.supylabel("error")

    # one legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    plt.savefig("figs/error_time_slices.png", dpi=300, bbox_inches="tight")

    plt.show()
