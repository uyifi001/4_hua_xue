# -*- coding: utf-8 -*-
"""
模型二：化学-力学耦合扩散模型 (论文 4.2, Eq.11-23) —— FEniCSx 实现

未知场：氧浓度 c (mol/m^3)、TGO 相场/体积分数 n (0=BC, 1=TGO)、位移 u (µm)

    ∂n/∂t = ξ (1-n) c                                     (Eq.11)  反应控制的相场演化
    S = κ ∂n/∂t,  κ = ρ/M · η_O2                          (Eq.12-13) 氧消耗汇项
    ∂c/∂t + ∇·J = -S                                      (Eq.14)  质量守恒
    J = -(D c / RT) ∇μ,  μ = μ0 + RT ln c - σ_m Ω         (Eq.15-16)
      ⇒ J = -D ∇c + (D c Ω / RT) ∇σ_m                     (应力辅助扩散)
    D = D0 exp(-Q/RT)                                     (Eq.17)  Q 取 Sec.3.4 拟合值
    ε = sym(∇u) = ε_el + ε_g,  ε_g = n γ I                (Eq.18-20)
    σ = K : (ε - ε_g),  K = (1-n) K_bc + n K_tgo          (Eq.21-22) 混合律
    ∇·σ = 0                                               (Eq.23)

数值算法 (论文 4.2.5 / Fig.27 的分离式隐式算法)：每个时间步内
    (1) 氧传输模块：以上一次迭代的 σ_m、n_old 求解 c (后向 Euler + Newton)；
        汇项采用与 (2) 完全一致的后向 Euler 形式  S = κ ξ (1-n_old) c / (1 + Δt ξ c)，保证质量守恒；
    (2) TGO 动力学模块：逐节点解析更新 n = (n_old + Δt ξ c) / (1 + Δt ξ c)；
    (3) 固体力学模块：以新 n 更新生长应变和混合刚度，求解平衡方程，得到新 σ_m 反馈到 (1)；
    重复 (1)-(3) n_stagger 次直至收敛。

D0 的确定：对"扩散控制 + 快速反应"的移动前沿，TGO 增厚满足 h^2 = 2 D c_s t / κ，
因此取 D_TGO = k_p κ / (2 c_s)，k_p 由 Arrhenius 式 (论文 Sec.3.4 实验拟合) 给出，
这等价于 D0 = K0 κ /(2 c_s)，Q 与实验活化能相同 —— 与论文"Q 由 3.4 节动力学模型确定"一致。

带/不带 TBC 开关：
    --tbc on : TC(300 µm, 多孔, 快扩散) / 初始 TGO(1 µm) / BC(200 µm)，TC 顶面 c = c_s
    --tbc off: 初始 Al2O3 氧化膜(1 µm) / Ni 基单晶基体(200 µm)，氧化膜外表面直接暴露 c = c_s，
               动力学参数取无涂层合金 (K0=6.78e5, Q=146 kJ/mol)，基体材料为 Ni 基单晶

用法：
    python chemo_mechanical.py --tbc on
    python chemo_mechanical.py --tbc off
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, geometry
from dolfinx.fem import petsc as fpetsc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (Params, R_GAS, add_common_args, params_from_args, build_mesh, stress_with_eigen,
                    von_mises, hydrostatic, symmetry_bcs, interp_expr, PlaneSampler, ensure_dir)

HERE = os.path.dirname(os.path.abspath(__file__))


def run(p: Params, outdir: str, n_stagger: int = 2, dt0: float = 0.01, dt_max: float = 0.5,
        stress_coupling: bool = True):
    comm = MPI.COMM_WORLD
    ensure_dir(outdir)
    t_start = time.time()
    dim = 2
    kp = p.kp()
    depth = p.h_tgo0 + 1.3 * np.sqrt(kp * p.t_end) + 3.0
    domain, eta_fn, cell_eta, info = build_mesh(p, depth, dim=dim)
    tc, tgo, base = p.materials()
    T = p.T_ox
    D_ox = p.D_ox * p.D_calib
    beta = p.Omega * 1e6 / (R_GAS * T) if stress_coupling else 0.0   # 1/MPa
    if comm.rank == 0:
        print(f"[CM] tbc={p.tbc} T={p.T_ox_C}°C kp={kp:.4f} µm²/h κ={p.kappa:.0f} mol/m³ "
              f"D_TGO={D_ox:.1f} µm²/h ξ={p.xi_h:.3f} m³/(mol·h) "
              f"ℓ_react={np.sqrt(D_ox / (p.kappa * p.xi_h)):.3f} µm ΩβRT={beta:.2e} 1/MPa "
              f"cells={info['n_cells']}")

    # ---------------- 分区 ----------------
    W0 = fem.functionspace(domain, ("DG", 0))
    chi_tc = fem.Function(W0)
    chi_tc.x.array[:] = (cell_eta > 0.0).astype(float)
    chi_r = 1.0 - chi_tc                    # 可反应区域 (BC/TGO 或 基体/氧化膜)

    V1 = fem.functionspace(domain, ("Lagrange", 1))
    c = fem.Function(V1, name="c_O2")
    c_old = fem.Function(V1)
    n = fem.Function(V1, name="n_TGO")
    n_old = fem.Function(V1)
    n0 = fem.Function(V1)
    eta_nodes_ = eta_fn.x.array
    react_node = eta_nodes_ <= 1e-9
    n0.x.array[:] = np.where(eta_nodes_ >= -p.h_tgo0 - 1e-9, 1.0, 0.0)   # 初始 TGO: n=1, BC: n=0
    n.x.array[:] = n0.x.array
    n_old.x.array[:] = n0.x.array
    c.x.array[:] = np.where(eta_nodes_ >= -1e-9, p.c_surf, 0.0)
    c_old.x.array[:] = c.x.array

    # ---------------- 氧扩散 (Eq.14-17) ----------------
    D = chi_tc * D_ox * p.D_tc_factor + chi_r * D_ox * (n + (1 - n) * p.D_base_factor)
    sm_cg = fem.Function(V1, name="sigma_m_cg")   # 连续化的静水应力 (用于 ∇σ_m)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt0))
    w = ufl.TestFunction(V1)
    xi, kap = p.xi_h, p.kappa
    S = chi_r * kap * xi * (1 - n_old) * c / (1 + dt_c * xi * c)
    Fc = ((c - c_old) / dt_c * w + ufl.dot(D * (ufl.grad(c) - beta * c * ufl.grad(sm_cg)), ufl.grad(w))
          + S * w) * ufl.dx
    # 顶面/底面的识别用自由度的 η 值 (几何已扭曲，无 TBC 时顶面为正弦形)
    top_dofs = np.where(np.isclose(eta_nodes_, info["top"]))[0].astype(np.int32)
    bot_dofs = np.where(np.isclose(eta_nodes_, info["bot"]))[0].astype(np.int32)
    bcs_c = [fem.dirichletbc(PETSc.ScalarType(p.c_surf), top_dofs, V1),
             fem.dirichletbc(PETSc.ScalarType(0.0), bot_dofs, V1)]
    prob_c = fpetsc.NonlinearProblem(
        Fc, c, bcs=bcs_c, petsc_options_prefix="oxy_",
        petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "none",
                       "snes_rtol": 1e-9, "snes_atol": 1e-10, "snes_max_it": 30,
                       "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

    # ---------------- 力学 (Eq.18-23) ----------------
    l_tc, m_tc = tc.lame() if p.tbc else (0.0, 0.0)
    l_bc, m_bc = base.lame()
    l_tg, m_tg = tgo.lame()
    lmb = chi_tc * l_tc + chi_r * ((1 - n) * l_bc + n * l_tg)     # Eq.22 混合律
    mu = chi_tc * m_tc + chi_r * ((1 - n) * m_bc + n * m_tg)
    gamma = p.gamma_g
    eg_s = chi_r * gamma * (n - n0)                                 # Eq.20 (以初始状态为参考)
    I = ufl.Identity(dim)
    V = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
    uT, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    sig_T, _ = stress_with_eigen(uT, lmb, mu, eg_s * I, eg_s, dim)
    Fu = ufl.inner(sig_T, ufl.sym(ufl.grad(v))) * ufl.dx
    bcs_u = symmetry_bcs(V, domain, info, dim)
    uh = fem.Function(V, name="u")
    prob_u = fpetsc.LinearProblem(ufl.lhs(Fu), ufl.rhs(Fu), bcs=bcs_u, u=uh, petsc_options_prefix="mech_",
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                 "pc_factor_mat_solver_type": "mumps"})
    sig, szz = stress_with_eigen(uh, lmb, mu, eg_s * I, eg_s, dim)
    sm_expr = hydrostatic(sig, szz, dim)
    # σ_m 的 L2 投影到 CG1
    q, r = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    prob_sm = fpetsc.LinearProblem(q * r * ufl.dx, sm_expr * r * ufl.dx, u=sm_cg, petsc_options_prefix="proj_",
                                   petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-10})

    vm_f = fem.Function(W0, name="von_Mises_MPa")
    syy_f = fem.Function(W0, name="sigma_yy_MPa")
    sm_f = fem.Function(W0, name="sigma_m_MPa")

    # ---------------- TGO 厚度统计 ----------------
    k = info["k"]
    xs_arc = np.linspace(0, info["Lx"], 4001)
    arc = np.trapezoid(np.sqrt(1 + (p.amplitude * k * np.sin(k * xs_arc)) ** 2), xs_arc)
    area_form = fem.form(chi_r * n * ufl.dx)
    sig_tgo_form = fem.form(chi_r * n * 0.5 * (sig[0, 0] + szz) * ufl.dx)   # TGO 面内平均应力 (x, z)

    def tgo_thickness():
        """法向平均厚度 = ∫ n dA / 界面弧长；另给出波峰(x=0)与波谷(x=λ/2)处竖直厚度"""
        a = comm.allreduce(fem.assemble_scalar(area_form), op=MPI.SUM)
        s_in = comm.allreduce(fem.assemble_scalar(sig_tgo_form), op=MPI.SUM) / max(a, 1e-12)
        return a / arc, s_in

    def column_thickness(x0):
        ys = np.linspace(-depth - p.amplitude, 0.5 + p.amplitude, 3000)
        pts = np.zeros((len(ys), 3))
        pts[:, 0] = min(max(x0, 1e-6), info["Lx"] - 1e-6)
        pts[:, 1] = ys
        tree = geometry.bb_tree(domain, domain.topology.dim)
        coll = geometry.compute_colliding_cells(domain, geometry.compute_collisions_points(tree, pts), pts)
        vals = np.zeros(len(ys))
        etav = np.zeros(len(ys))
        for i in range(len(ys)):
            cl = coll.links(i)
            if len(cl):
                vals[i] = n.eval(pts[i:i + 1], cl[:1])[0]
                etav[i] = eta_fn.eval(pts[i:i + 1], cl[:1])[0]
        mask = (etav <= 0) & (vals > 0.5)
        return (ys[1] - ys[0]) * mask.sum()

    # ---------------- 绘图 ----------------
    A_ = p.amplitude
    ylo, yhi = -A_ - depth + 2.0, A_ + (10.0 if p.tbc else 1.0)
    sampler = PlaneSampler(domain, dim, (0.0, info["Lx"]), (ylo, yhi), nx=300, ny=300)
    snaps = {}
    eta_s = sampler(eta_fn)
    in_tc = eta_s > 1e-6            # 绘图时 n 只在 BC/TGO 区显示

    def snapshot(t):
        interp_expr(von_mises(sig, szz, dim), W0, vm_f)
        interp_expr(sig[1, 1], W0, syy_f)
        interp_expr(sm_expr, W0, sm_f)
        nn = sampler(n)
        nn[in_tc] = np.nan
        snaps[t] = dict(n=nn, c=sampler(c), syy=sampler(syy_f), vm=sampler(vm_f))

    xdmf = io.XDMFFile(comm, os.path.join(outdir, "fields.xdmf"), "w")
    xdmf.write_mesh(domain)

    h_init, _ = tgo_thickness()     # 初始 TGO 的法向平均厚度

    # ---------------- 时间推进 ----------------
    t, dt = 0.0, dt0
    hist = []
    out_times = list(p.out_times)
    step = 0
    while t < p.t_end - 1e-9:
        nxt = min([to for to in out_times if to > t + 1e-9], default=p.t_end)
        dt = min(dt, nxt - t)
        dt_c.value = dt
        c_old.x.array[:] = c.x.array
        n_old.x.array[:] = n.x.array
        for it in range(n_stagger):
            prob_c.solve()
            reason = prob_c.solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(f"氧扩散 Newton 未收敛 (reason={reason}) t={t}")
            np.maximum(c.x.array, 0.0, out=c.x.array)
            # (2) 相场解析更新 (后向 Euler)
            a_ = dt * xi * c.x.array
            n_new = (n_old.x.array + a_) / (1 + a_)
            n.x.array[:] = np.where(react_node, np.maximum(n_new, n0.x.array), n0.x.array)
            n.x.scatter_forward()
            # (3) 力学
            prob_u.solve()
            if beta != 0.0:
                prob_sm.solve()
        t += dt
        step += 1
        h_avg, s_in = tgo_thickness()
        rec = dict(t_h=t, h_tgo_avg_um=h_avg, dh_tgo_avg_um=h_avg - h_init, sigma_inplane_tgo_MPa=s_in,
                   c_min=float(c.x.array.min()), n_max=float(n.x.array.max()),
                   newton_its=prob_c.solver.getIterationNumber())
        if any(abs(t - to) < 1e-6 for to in out_times):
            snapshot(round(t, 6))
            rec["h_peak_um"] = column_thickness(0.0)
            rec["h_valley_um"] = column_thickness(info["Lx"] / 2)
            rec["syy_max_MPa"] = float(np.nanmax(snaps[round(t, 6)]["syy"]))
            rec["syy_min_MPa"] = float(np.nanmin(snaps[round(t, 6)]["syy"]))
            rec["vm_max_MPa"] = float(np.nanmax(snaps[round(t, 6)]["vm"]))
            for f in (c, n, uh, sm_cg):
                xdmf.write_function(f, t)
            for f in (vm_f, syy_f, sm_f):
                xdmf.write_function(f, t)
            if comm.rank == 0:
                print(f"  t={t:6.1f} h  h_TGO(avg)={h_avg:.3f} µm  peak={rec['h_peak_um']:.2f} "
                      f"valley={rec['h_valley_um']:.2f} µm  σyy∈[{rec['syy_min_MPa']:.0f}, "
                      f"{rec['syy_max_MPa']:.0f}] MPa  Mises_max={rec['vm_max_MPa']:.0f} MPa  "
                      f"steps={step}  ({time.time() - t_start:.0f}s)")
        hist.append(rec)
        dt = min(dt * 1.25, dt_max)
    xdmf.close()

    # ---------------- 输出 ----------------
    if comm.rank == 0:
        keys = ["t_h", "h_tgo_avg_um", "dh_tgo_avg_um", "sigma_inplane_tgo_MPa", "c_min", "n_max", "newton_its",
                "h_peak_um", "h_valley_um", "syy_max_MPa", "syy_min_MPa", "vm_max_MPa"]
        with open(os.path.join(outdir, "history.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for h in hist:
                wr.writerow({kk: h.get(kk, "") for kk in keys})
        # Fig.29 类型：三行 (n, c, σ_yy) × 四列 (10/20/50/100 h)
        ts = sorted(snaps.keys())
        fig, axs = plt.subplots(3, len(ts), figsize=(4.2 * len(ts), 10.5), squeeze=False)
        xs = np.linspace(0, info["Lx"], 400)
        vmax_s = max(np.nanpercentile(np.abs(snaps[tt]["syy"]), 99.5) for tt in ts)
        for j, tt in enumerate(ts):
            for i, (key, lab, cmap, kw) in enumerate((
                    ("n", "TGO phase field n", "jet", dict(vmin=0, vmax=1)),
                    ("c", r"O$_2$ concentration (mol/m$^3$)", "jet", dict(vmin=0, vmax=p.c_surf)),
                    ("syy", r"$\sigma_{yy}$ (MPa)", "coolwarm", dict(vmin=-vmax_s, vmax=vmax_s)))):
                ax = axs[i, j]
                im = ax.pcolormesh(sampler.XX, sampler.YY, snaps[tt][key], cmap=cmap, shading="auto", **kw)
                ax.plot(xs, A_ * np.cos(k * xs), "k-", lw=0.5)
                ax.set_aspect("equal")
                ax.set_title(f"{lab}\nt = {tt:g} h", fontsize=9)
                ax.set_xlabel("x (µm)")
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f"Chemo-mechanical diffusion model (Fig.29-type), TBC={'on' if p.tbc else 'off'}, "
                     f"{p.T_ox_C:g} °C")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig29_fields.png"), dpi=120)
        plt.close(fig)

        th = np.array([h["t_h"] for h in hist])
        hh = np.array([h["h_tgo_avg_um"] for h in hist])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.r_[0, th], np.r_[h_init, hh], "-", label="model: total TGO (normal-averaged)")
        ax.plot(np.r_[0, th], np.r_[0, hh - h_init], "-", label=r"model: growth $\Delta H$")
        tt = np.linspace(0, p.t_end, 200)
        ax.plot(tt, np.sqrt(kp * tt), ":", label=r"$\Delta H=\sqrt{k_p t}$ (experimental fit, Eq.6)")
        ax.set_xlabel("time (h)")
        ax.set_ylabel("TGO thickness (µm)")
        ax.legend()
        ax.set_title(f"TBC={'on' if p.tbc else 'off'}, {p.T_ox_C:g} °C")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "tgo_thickness.png"), dpi=130)
        plt.close(fig)
        np.savez_compressed(os.path.join(outdir, "snapshots.npz"),
                            X=sampler.XX, Y=sampler.YY,
                            **{f"{key}_{int(round(tt_))}h": snaps[tt_][key] for tt_ in ts for key in snaps[tt_]})
        p.to_json(os.path.join(outdir, "params.json"))
        print(f"[CM] done in {time.time() - t_start:.1f} s → {outdir}")
    return hist


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--gamma", type=float, default=3.0e-3, help="生长应变幅值 γ (Eq.20)")
    ap.add_argument("--omega", type=float, default=2.0e-6, help="氧偏摩尔体积 Ω (m^3/mol)")
    ap.add_argument("--D-calib", type=float, default=1.5, help="D_TGO 标定系数 (对齐实验 ΔH)")
    ap.add_argument("--no-stress-coupling", action="store_true", help="关闭 σ_m 对扩散的影响 (对比用)")
    ap.add_argument("--n-stagger", type=int, default=2, help="每步分离迭代次数")
    ap.add_argument("--dt-max", type=float, default=0.5, help="最大时间步 h")
    args = ap.parse_args()
    p = params_from_args(args, gamma_g=args.gamma, Omega=args.omega, D_calib=args.D_calib)
    if args.h_fine is None:
        p.h_fine = 0.25
    out = args.out or os.path.join(HERE, "results", f"chemo_mechanical_tbc_{args.tbc}")
    run(p, out, n_stagger=args.n_stagger, dt_max=args.dt_max, stress_coupling=not args.no_stress_coupling)


if __name__ == "__main__":
    main()
