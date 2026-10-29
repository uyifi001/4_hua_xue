# -*- coding: utf-8 -*-
"""
公共模块：材料参数、氧化动力学、带/不带 TBC 的几何(正弦界面)网格生成、后处理工具。

单位制（全程一致）：
    长度  µm
    时间  h
    应力  MPa   (E 用 MPa 表示, 位移 µm → 应变无量纲)
    浓度  mol/m^3
    温度  K

对应论文: S.S. Wang et al., "Oxidation mechanism and multi-physics coupling modeling of
thermal barrier coated Ni-based single crystal superalloys" (hou/9_30/4.md)。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh, geometry

R_GAS = 8.314  # J/(mol K)


# ----------------------------------------------------------------------------
# 1. 材料与动力学参数
# ----------------------------------------------------------------------------
@dataclass
class Material:
    E: float      # MPa
    nu: float
    alpha: float  # 1/K

    def lame(self):
        lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        return lmbda, mu


@dataclass
class Params:
    # ---------------- 开关 ----------------
    tbc: bool = True                 # True: 带 TBC (TC/TGO/BC)；False: 裸 Ni 基单晶 (氧化膜/基体)
    # ---------------- 温度 ----------------
    T_ox_C: float = 1050.0           # 等温氧化温度 (°C)，论文模拟取 1050 °C
    T_ref_C: float = 25.0            # 无应力参考温度 (°C)，升温段 T_ref → T_ox
    t_end: float = 100.0             # 氧化时间 (h)
    # ---------------- 几何 (论文 4.2.6) ----------------
    wavelength: float = 60.0         # 正弦界面波长 µm
    amplitude: float = 10.0          # 正弦界面振幅 µm
    h_tc: float = 300.0              # TC 厚度 µm（tbc=False 时不存在）
    h_tgo0: float = 1.0              # 初始 TGO/氧化膜 厚度 µm
    h_base: float = 200.0            # BC 厚度 (tbc) / 基体厚度 (no tbc) µm
    h_fine: float = 0.3              # 界面加密区网格尺寸 µm
    h_coarse_max: float = 20.0       # 远场最大网格尺寸 µm
    # ---------------- 动力学 (论文 Eq.1-3, 6-7, Sec.3.4) ----------------
    # 带涂层: TGO 增厚系数 K0=1.46e6 µm^2/h, Q=167 kJ/mol  (1050 °C → kp≈0.37 µm^2/h)
    # 无涂层: γ' 贫化层增厚系数 K0=6.78e5 µm^2/h, Q=146 kJ/mol (1050 °C → kp≈1.2 µm^2/h)
    K0_tbc: float = 1.46e6
    Q_tbc: float = 167e3
    K0_bare: float = 6.78e5
    Q_bare: float = 146e3
    # ---------------- 热-力耦合 (4.1) ----------------
    PBR: float = 1.28                # Al → Al2O3 Pilling-Bedworth 比
    h0_ref: float = 6.0              # Eq.9 中的 h0 (被氧化金属参考厚度, µm)；取为固定材料常数，
                                     # 使 ε_vol ∝ h(t)，不同工况(带/不带 TBC, 温度)之间可直接比较
    lateral_ratio: float = 0.0035    # 切向(侧向)生长应变 / ε_vol (论文未给, 标定值)
    normal_ratio: float = 0.0        # 法向(增厚方向)生长应变 / ε_vol；默认 0: 增厚由 TGO 层几何(h0)本身体现，
                                     # 不再作为本征应变施加 (否则约束下会产生 GPa 级非物理应力)
    use_paper_bc_cte: bool = False   # True: BC 线胀系数按论文表9原值 1.7e-6 (疑为笔误)
    # ---------------- 化学-力耦合扩散 (4.2) ----------------
    c_surf: float = 2.5              # 表面氧浓度 mol/m^3
    xi: float = 1.0e-4               # 反应常数 m^3/(mol s)，论文 Eq.11
    rho_al2o3: float = 3950.0        # kg/m^3
    M_al2o3: float = 0.10196         # kg/mol
    eta_O2: float = 1.5              # 每 mol Al2O3 消耗 1.5 mol O2
    Omega: float = 2.0e-6            # 氧偏摩尔体积 m^3/mol (论文未给)
    gamma_g: float = 3.0e-3          # 生长应变幅值 γ (论文 Eq.20, 未给, 标定值)
    D_calib: float = 1.5             # D_TGO 标定系数：补偿初始 TGO、反应区宽度与界面曲率使 ΔH(t) 对齐实验 √(k_p t)
    D_tc_factor: float = 1.0e3       # 多孔 TC 中氧扩散系数 / TGO 中扩散系数
    D_base_factor: float = 1.0       # BC/基体 中氧扩散系数 / TGO 中扩散系数
    # ---------------- 剥落判据 (Eq.24) ----------------
    Gc: float = 20.0                 # 界面断裂韧性 J/m^2 (论文未给, 文献常用量级)
    # ---------------- 数值 ----------------
    dim: int = 2
    n_heat: int = 5
    out_times: list = field(default_factory=lambda: [10.0, 20.0, 50.0, 100.0])

    # ---- 派生量 ----
    @property
    def T_ox(self):
        return self.T_ox_C + 273.15

    @property
    def dT(self):
        return self.T_ox_C - self.T_ref_C

    def kp(self, T=None):
        """TGO(带TBC) 或 氧化影响层(无TBC) 增厚系数 µm^2/h, Arrhenius Eq.2/7"""
        T = self.T_ox if T is None else T
        if self.tbc:
            return self.K0_tbc * np.exp(-self.Q_tbc / (R_GAS * T))
        return self.K0_bare * np.exp(-self.Q_bare / (R_GAS * T))

    def dh(self, t):
        """抛物线规律 Eq.6: ΔH = sqrt(kp t)"""
        return np.sqrt(self.kp() * np.maximum(t, 0.0))

    @property
    def kappa(self):
        """Eq.13: 生成 1 m^3 TGO 所需氧量 mol/m^3"""
        return self.rho_al2o3 / self.M_al2o3 * self.eta_O2

    @property
    def D_ox(self):
        """由抛物线速率反推 TGO 中氧扩散系数 (µm^2/h), 满足 kp = 2 D c_s / κ。
        与 Eq.17 一致: D ∝ exp(-Q/RT)，Q 取 Sec.3.4 拟合的活化能。"""
        return self.kp() * self.kappa / (2.0 * self.c_surf)

    @property
    def xi_h(self):
        """ξ 换算到 m^3/(mol h)"""
        return self.xi * 3600.0

    def materials(self):
        """返回 (top, oxide, base) 三种材料；tbc=False 时 top=None。
        表9 (1050 °C)：TC 22 GPa/0.12/1e-5；BC 110 GPa/0.33/1.7e-6(→默认 1.7e-5)；TGO 320 GPa/0.25/6.8e-6。
        裸合金基体 (论文未给, 取二代镍基单晶 1050 °C 典型值)：E=95 GPa, ν=0.38, α=1.6e-5。"""
        tc = Material(22e3, 0.12, 1.0e-5)
        tgo = Material(320e3, 0.25, 6.8e-6)
        if self.tbc:
            bc = Material(110e3, 0.33, 1.7e-6 if self.use_paper_bc_cte else 1.7e-5)
            return tc, tgo, bc
        sx = Material(95e3, 0.38, 1.6e-5)
        return None, tgo, sx

    def to_json(self, path):
        d = asdict(self)
        d.update(dict(kp_um2_per_h=self.kp(), D_ox_um2_per_h=self.D_ox, kappa=self.kappa))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)


def add_common_args(ap):
    """命令行公共参数（含带/不带 TBC 开关）"""
    ap.add_argument("--tbc", choices=["on", "off"], default="on",
                    help="带/不带热障涂层开关: on=TC/TGO/BC 体系, off=裸 Ni 基单晶")
    ap.add_argument("--T", type=float, default=1050.0, help="氧化温度 °C (980 或 1050)")
    ap.add_argument("--t-end", type=float, default=100.0, help="氧化时间 h")
    ap.add_argument("--amp", type=float, default=10.0, help="界面正弦振幅 µm")
    ap.add_argument("--h-fine", type=float, default=None, help="界面加密网格尺寸 µm")
    ap.add_argument("--out", type=str, default=None, help="输出目录 (默认 results/<model>_tbc_on|off)")
    return ap


def params_from_args(args, **kw):
    p = Params(tbc=(args.tbc == "on"), T_ox_C=args.T, t_end=args.t_end, amplitude=args.amp, **kw)
    if args.h_fine is not None:
        p.h_fine = args.h_fine
    p.out_times = [t for t in (10.0, 20.0, 50.0, 100.0) if t <= p.t_end + 1e-9]
    if not p.out_times or abs(p.out_times[-1] - p.t_end) > 1e-9:
        p.out_times.append(p.t_end)
    return p


# ----------------------------------------------------------------------------
# 2. 几何/网格：结构化网格 + 竖向坐标映射 + 正弦扭曲
#    参考竖直坐标 η: η=0 为 TC/TGO 界面 (无 TBC 时为氧化膜外表面)
#    TC: 0<η<h_tc ; 初始 TGO: -h_tgo0<η<0 ; BC/基体: η<-h_tgo0
#    实际坐标: y = η + A·s(x)·φ(η)，s(x)=cos(2πx/λ) (2D) 或 cos(2πx/λ)cos(2πy/λ) (3D)
# ----------------------------------------------------------------------------
def _graded_nodes(a, b, h0, hmax, ratio=1.18):
    """从 a 向 b 以几何级数加粗的一维节点 (不含 a)"""
    pts, x, h, sgn = [], a, h0, np.sign(b - a)
    while abs(b - x) > 1e-12:
        h = min(h * ratio, hmax)
        if abs(b - x) < 1.5 * h:
            x = b
        else:
            x = x + sgn * h
        pts.append(x)
    return pts


def eta_nodes(p: Params, depth_below):
    """竖向节点分布: 界面加密带 [-depth_below, top_fine] 内均匀 h_fine"""
    top = p.h_tc if p.tbc else 0.0
    bot = -(p.h_tgo0 + p.h_base)
    top_fine = min(2.0, top) if p.tbc else 0.0
    n_f = int(np.ceil((top_fine + depth_below) / p.h_fine))
    fine = np.linspace(-depth_below, top_fine, n_f + 1)
    # 保证 η=0 与 η=-h_tgo0 是节点 (材料界面与单元边界对齐)
    fine = np.unique(np.concatenate([fine, [0.0, -p.h_tgo0]]))
    fine = fine[np.concatenate([[True], np.diff(fine) > 0.25 * p.h_fine])]
    below = _graded_nodes(-depth_below, bot, p.h_fine, p.h_coarse_max)[::-1]
    above = _graded_nodes(top_fine, top, p.h_fine, p.h_coarse_max) if top > top_fine else []
    nodes = np.unique(np.concatenate([below, fine, above]))
    return nodes


def build_mesh(p: Params, depth_below: float, dim: int = 2, n_lat: int | None = None,
               comm=MPI.COMM_WORLD):
    """构造带正弦界面的周期单胞网格。
    2D: x∈[0, λ] (整个波长, 两侧对称面 ⇔ 周期边界)
    3D: x,y∈[0, λ/2] (1/4 单胞；界面为拉伸型正弦 z=A·cos(2πx/λ)，沿 y 方向等截面拉伸，
        与论文 Fig.22 "Three dimensional sine model" 一致；四侧对称面)
    返回 (domain, eta(CG1 函数), cell_eta(每个单元中心 η 的 numpy 数组), info)"""
    lam, A = p.wavelength, p.amplitude
    etas = eta_nodes(p, depth_below)
    nv = len(etas) - 1
    top, bot = etas[-1], etas[0]
    if dim == 2:
        Lx = lam
        nx = n_lat or int(round(Lx / p.h_fine))
        domain = dmesh.create_rectangle(comm, [[0.0, 0.0], [Lx, float(nv)]], [nx, nv],
                                        cell_type=dmesh.CellType.quadrilateral)
    else:
        Lx = lam / 2.0
        nx = n_lat or int(round(Lx / p.h_fine))
        domain = dmesh.create_box(comm, [[0.0, 0.0, 0.0], [Lx, Lx, float(nv)]], [nx, nx, nv],
                                  cell_type=dmesh.CellType.hexahedron)
    vd = dim - 1
    X = domain.geometry.x
    zeta = X[:, vd].copy()
    eta = np.interp(zeta, np.arange(nv + 1), etas)
    X[:, vd] = eta  # 先写入未扭曲的 η，计算单元标签/η 函数

    V1 = fem.functionspace(domain, ("Lagrange", 1))
    eta_fn = fem.Function(V1, name="eta")
    eta_fn.interpolate(lambda x: x[vd])

    tdim = domain.topology.dim
    ncell = domain.topology.index_map(tdim).size_local + domain.topology.index_map(tdim).num_ghosts
    mid = dmesh.compute_midpoints(domain, tdim, np.arange(ncell, dtype=np.int32))
    cell_eta = mid[:, vd].copy()

    # 正弦扭曲 (仅竖向平移)，φ(η) 在加密带内为 1，向顶/底线性衰减到 0
    band_lo, band_hi = -depth_below, (min(2.0, top) if p.tbc else 0.0)

    def phi(e):
        w = np.ones_like(e)
        w = np.where(e < band_lo, (e - bot) / (band_lo - bot), w)
        if top > band_hi:
            w = np.where(e > band_hi, (top - e) / (top - band_hi), w)
        return np.clip(w, 0.0, 1.0)

    k = 2 * np.pi / lam
    if dim == 2:
        s = np.cos(k * X[:, 0])
    else:
        # 论文 Fig.22 的 3D 正弦界面 = 2D 正弦沿 y 方向拉伸（等截面），与 y 无关
        s = np.cos(k * X[:, 0])
    X[:, vd] = eta + A * s * phi(eta)

    info = dict(Lx=Lx, top=top, bot=bot, nv=nv, nx=nx, vd=vd, k=k,
                n_cells=int(domain.topology.index_map(tdim).size_global))
    return domain, eta_fn, cell_eta, info


def interface_normal(domain, p: Params, dim):
    """TC/TGO 界面(正弦面)的单位法向 (UFL 表达式)，用于区分生长应变的法向/切向分量"""
    x = ufl.SpatialCoordinate(domain)
    k = 2 * np.pi / p.wavelength
    A = p.amplitude
    if dim == 2:
        g = ufl.as_vector((A * k * ufl.sin(k * x[0]), 1.0))
    else:
        # 拉伸型正弦：界面与 y 无关，法向只有 x、z 分量
        g = ufl.as_vector((A * k * ufl.sin(k * x[0]), 0.0, 1.0))
    return g / ufl.sqrt(ufl.dot(g, g))


# ----------------------------------------------------------------------------
# 3. 平面应变 / 三维 弹性工具（含本征应变 = 生长应变 + 热应变）
# ----------------------------------------------------------------------------
def eps(u):
    return ufl.sym(ufl.grad(u))


def stress_with_eigen(u, lmbda, mu, eig, eig_zz, dim):
    """σ = λ tr(ε-ε*) I + 2μ(ε-ε*)
    2D 平面应变：ε_zz = 0，本征应变的 zz 分量 eig_zz 进入 tr 与 σ_zz。
    返回 (σ 面内张量, σ_zz 标量 或 None)"""
    I = ufl.Identity(dim)
    e_el = eps(u) - eig
    if dim == 2:
        tr = ufl.tr(e_el) - eig_zz
        sig = lmbda * tr * I + 2 * mu * e_el
        szz = lmbda * tr - 2 * mu * eig_zz
        return sig, szz
    tr = ufl.tr(e_el)
    return lmbda * tr * I + 2 * mu * e_el, None


def von_mises(sig, szz, dim):
    if dim == 2:
        sxx, syy, sxy = sig[0, 0], sig[1, 1], sig[0, 1]
        return ufl.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                        + 3 * sxy ** 2 + 1e-12)
    s = ufl.dev(sig)
    return ufl.sqrt(1.5 * ufl.inner(s, s) + 1e-12)


def hydrostatic(sig, szz, dim):
    if dim == 2:
        return (sig[0, 0] + sig[1, 1] + szz) / 3.0
    return ufl.tr(sig) / 3.0


def equivalent_strain(e, dim):
    """等效(von Mises)应变 ε_eq = sqrt(2/3 e_dev:e_dev)，e 为 2D/3D 应变张量"""
    if dim == 2:
        e3 = ufl.as_matrix([[e[0, 0], e[0, 1], 0.0], [e[1, 0], e[1, 1], 0.0], [0.0, 0.0, 0.0]])
    else:
        e3 = e
    d = ufl.dev(e3)
    return ufl.sqrt(2.0 / 3.0 * ufl.inner(d, d) + 1e-20)


def symmetry_bcs(V, domain, info, dim):
    """侧面对称(法向位移=0，等价于论文的周期边界 + 对称单胞)，底面固定"""
    bcs = []
    fdim = domain.topology.dim - 1
    Lx = info["Lx"]
    for comp in range(dim - 1):
        for val in (0.0, Lx):
            facets = dmesh.locate_entities_boundary(
                domain, fdim, lambda x, c=comp, v=val: np.isclose(x[c], v))
            Vc, _ = V.sub(comp).collapse()
            dofs = fem.locate_dofs_topological((V.sub(comp), Vc), fdim, facets)
            zero = fem.Function(Vc)
            bcs.append(fem.dirichletbc(zero, dofs, V.sub(comp)))
    vd = info["vd"]
    bot = info["bot"]
    facets = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[vd], bot))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bcs.append(fem.dirichletbc(np.zeros(dim, dtype=float), dofs, V))
    return bcs


def interp_expr(expr, W, out=None):
    """把 UFL 表达式插值到空间 W (DG0 = 单元中心值)；expr 也可以是预编译的 fem.Expression"""
    f = out if out is not None else fem.Function(W)
    if not isinstance(expr, fem.Expression):
        expr = fem.Expression(expr, W.element.interpolation_points)
    f.interpolate(expr)
    return f


# ----------------------------------------------------------------------------
# 4. 后处理：规则网格采样(2D 截面) 以便 matplotlib 绘图
# ----------------------------------------------------------------------------
class PlaneSampler:
    """在 x–y(2D) 或 x–z(3D, y≈0 截面) 平面上规则采样有限元函数"""

    def __init__(self, domain, dim, xlim, ylim, nx=240, ny=240):
        self.dim = dim
        xs = np.linspace(*xlim, nx)
        ys = np.linspace(*ylim, ny)
        self.XX, self.YY = np.meshgrid(xs, ys)
        pts = np.zeros((nx * ny, 3))
        pts[:, 0] = np.clip(self.XX.ravel(), 1e-6, xlim[1] - 1e-6)
        if dim == 2:
            pts[:, 1] = self.YY.ravel()
        else:
            pts[:, 1] = 1e-6
            pts[:, 2] = self.YY.ravel()
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cand = geometry.compute_collisions_points(tree, pts)
        coll = geometry.compute_colliding_cells(domain, cand, pts)
        idx, cells = [], []
        for i in range(pts.shape[0]):
            c = coll.links(i)
            if len(c) > 0:
                idx.append(i)
                cells.append(c[0])
        self.idx = np.array(idx, dtype=np.int64)
        self.cells = np.array(cells, dtype=np.int32)
        self.pts = pts

    def __call__(self, f):
        out = np.full(self.pts.shape[0], np.nan)
        if len(self.idx):
            v = f.eval(self.pts[self.idx], self.cells)
            out[self.idx] = v[:, 0] if v.ndim > 1 else v
        return out.reshape(self.XX.shape)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
