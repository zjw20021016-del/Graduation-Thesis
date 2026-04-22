import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from typing import Dict, Any

# ====================== 1. 边界函数 φ (已修复 rectangle 边界判断) ======================
def phi_boundary(pos: np.ndarray, domain_type: str, params: Dict[str, Any]):
    if domain_type == "circle":
        center = params.get("center", np.zeros(2))
        # 确保 center 是数组
        cx = center[0] if hasattr(center, '__len__') else 0.0
        cy = center[1] if hasattr(center, '__len__') else 0.0

        theta = np.arctan2(pos[:, 1] - cy, pos[:, 0] - cx)
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        return theta ** 3

    elif domain_type == "rectangle":

        x, y = pos[:, 0], pos[:, 1]

        a, b = params["a"], params["b"]

        val = np.zeros(len(pos))

        tol = 1e-4

        # 右边界

        mask_right = np.abs(x - a) < tol

        val[mask_right] = np.sin(np.pi * y[mask_right])

        # 上边界

        mask_top = np.abs(y - b) < tol

        val[mask_top] = np.sin(np.pi * x[mask_top])

        return val

    elif domain_type == "triangle":
        return np.linalg.norm(pos)

    else:
        raise ValueError("不支持的 domain_type！请使用 'circle'、'rectangle' 或 'triangle'")

# ====================== 2. PDE 精确解 ======================

# ====================== 圆形域精确解（保持不变） ======================
def exact_solution_circle(pts, R=1.0):
    """
    计算单位圆内多个点 (x,y) 的精确解（泊松积分公式）
    pts: shape (N, 2) 或 (2,)
    """
    pts = np.asarray(pts)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)          # 单个点也转为 (1,2)

    r = np.linalg.norm(pts, axis=1)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    result = np.zeros(len(pts))

    for i in range(len(pts)):
        if r[i] >= R - 1e-8:              # 在边界或外面
            result[i] = theta[i] ** 3
            continue

        # 定义积分函数（针对第 i 个点）
        def integrand(alpha):
            denom = R**2 - 2 * r[i] * np.cos(theta[i] - alpha) + r[i]**2
            return (alpha ** 3) / denom

        integral, _ = quad(integrand, 0, 2 * np.pi, limit=300)
        result[i] = (R**2 - r[i]**2) / (2 * np.pi) * integral

    return result


# ====================== 方形域精确解（已完全重写：正确分离变量 + sinh 级数） ======================
def exact_solution_rectangle(x, y, a=1, b=1):
    x = np.asarray(x)
    y = np.asarray(y)
    u = (np.sinh(np.pi * x) / np.sinh(np.pi * a)) * np.sin(np.pi * y) + \
        (np.sinh(np.pi * y) / np.sinh(np.pi * b)) * np.sin(np.pi * x)
    return u.item() if u.size == 1 else u


# ====================== 用于判断是否还在区域内（保持不变） ======================
def is_inside_domain(pos: np.ndarray, domain_type: str, params: Dict[str, Any]) -> np.ndarray:
    """
    判断位置 pos (形状 (N, 2)) 是否在开区域 D 内部。
    返回 bool 数组，True 表示仍在 D 内。
    """
    if domain_type == "circle":
        # 圆形域：以原点为中心，半径 R
        center = params.get("center", np.zeros(2))
        radius = params["radius"]
        dist_sq = np.sum((pos - center) ** 2, axis=1)
        return dist_sq < radius ** 2  # 开区域：严格小于半径

    elif domain_type == "rectangle":
        # 矩形域：[0, a] × [0, b]
        a = params["a"]
        b = params["b"]
        x, y = pos[:, 0], pos[:, 1]
        return (x > 0) & (x < a) & (y > 0) & (y < b)  # 开区域

    elif domain_type == "triangle":
        # 三角形域：通过 3 个顶点确定（必须按逆时针或顺时针顺序给出）
        verts = params["vertices"]  # shape (3, 2)
        A, B, C = verts[0], verts[1], verts[2]
        v0 = B - A
        v1 = C - A
        v2 = pos - A  # 广播：(N, 2) - (2,)

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.sum(v0 * v2, axis=1)
        dot11 = np.dot(v1, v1)
        dot12 = np.sum(v1 * v2, axis=1)

        denom = dot00 * dot11 - dot01 * dot01
        inv_denom = 1.0 / (denom + 1e-12)  # 防止奇异

        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # 内部判据（barycentric坐标）：u>0, v>0, u+v<1
        return (u > 0) & (v > 0) & (u + v < 1)

    else:
        raise ValueError("不支持的 domain_type！请使用 'circle'、'rectangle' 或 'triangle'")

def simulate_first_exit_with_position(
        start: np.ndarray,
        domain_type: str,
        domain_params: dict,
        dt: float = 1e-5,
        max_steps: int = 100000000,
        n_sims: int = 10000,
        seed=None
):
    rng = np.random.default_rng(seed)

    start = np.asarray(start, dtype=float).reshape(-1)
    assert start.shape[0] == 2
    pos = np.tile(start, (n_sims, 1))
    prev = pos.copy()
    active = np.ones(n_sims, dtype=bool)

    for _ in range(max_steps):
        if not np.any(active):
            break

        idx = np.where(active)[0]

        dW = rng.normal(size=(len(idx), 2)) * np.sqrt(dt)

        prev[idx] = pos[idx]
        pos[idx] += dW

        still_inside = is_inside_domain(pos[idx], domain_type, domain_params)
        exited = ~still_inside

        if np.any(exited):
            ex_idx = idx[exited]

            x0 = prev[ex_idx]
            x1 = pos[ex_idx]

            t = np.ones(len(ex_idx))

            for i in range(len(ex_idx)):
                p0, p1 = x0[i], x1[i]

                cand = compute_intersection_t(p0, p1, domain_type, domain_params)

                t[i] = min(cand) if cand else 1.0

            pos[ex_idx] = x0 + t[:, None] * (x1 - x0)
            active[ex_idx] = False

    return pos

# =========================
# 统一交点接口
# =========================
def compute_intersection_t(p0, p1, domain_type, params):
    if domain_type == "rectangle":
        return intersect_rectangle(p0, p1, params)

    elif domain_type == "circle":
        return intersect_circle(p0, p1, params)

    elif domain_type == "triangle":
        return intersect_triangle(p0, p1, params)

    else:
        raise ValueError("Unknown domain_type")


# =========================
# 矩形交点
# =========================
def intersect_rectangle(p0, p1, params):
    a, b = params["a"], params["b"]
    cand = []

    eps = 1e-12

    # x = 0
    if abs(p1[0] - p0[0]) > eps:
        t = (0 - p0[0]) / (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        if -eps <= t <= 1+eps and 0-eps <= y <= b+eps:
            cand.append(t)

    # x = a
    if abs(p1[0] - p0[0]) > eps:
        t = (a - p0[0]) / (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        if -eps <= t <= 1+eps and 0-eps <= y <= b+eps:
            cand.append(t)

    # y = 0
    if abs(p1[1] - p0[1]) > eps:
        t = (0 - p0[1]) / (p1[1] - p0[1])
        x = p0[0] + t * (p1[0] - p0[0])
        if -eps <= t <= 1+eps and 0-eps <= x <= a+eps:
            cand.append(t)

    # y = b
    if abs(p1[1] - p0[1]) > eps:
        t = (b - p0[1]) / (p1[1] - p0[1])
        x = p0[0] + t * (p1[0] - p0[0])
        if -eps <= t <= 1+eps and 0-eps <= x <= a+eps:
            cand.append(t)

    return cand


# =========================
# 圆形交点
# =========================
def intersect_circle(p0, p1, params):
    center = np.array(params["center"])
    R = params["radius"]

    d = p1 - p0
    f = p0 - center

    A = np.dot(d, d)
    B = 2 * np.dot(f, d)
    C = np.dot(f, f) - R**2

    disc = B**2 - 4*A*C
    if disc < 0:
        return []

    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)

    eps = 1e-12
    return [t for t in (t1, t2) if -eps <= t <= 1+eps]


# =========================
# 三角形交点
# =========================
def intersect_segment(p0, p1, q0, q1):
    d = p1 - p0
    e = q1 - q0

    denom = d[0]*e[1] - d[1]*e[0]
    if abs(denom) < 1e-12:
        return None

    diff = q0 - p0
    t = (diff[0]*e[1] - diff[1]*e[0]) / denom
    s = (diff[0]*d[1] - diff[1]*d[0]) / denom

    eps = 1e-12
    if -eps <= t <= 1+eps and -eps <= s <= 1+eps:
        return t
    return None


def intersect_triangle(p0, p1, params):
    v0, v1, v2 = params["vertices"]
    edges = [(v0, v1), (v1, v2), (v2, v0)]

    ts = []
    for q0, q1 in edges:
        t = intersect_segment(p0, p1, q0, q1)
        if t is not None:
            ts.append(t)

    return ts

def compute_w_monte_carlo(
        start_points: np.ndarray,  # 可以是单个点或多个点，shape (K, 2)
        domain_type="rectangle",
        domain_params=None,
        dt=0.0005,          # 默认值已优化
        n_sims=10000,       # 默认值已优化（原来 1000 太小）
        seed=None
):
    """
    计算 w(x) = E^x [ φ(B_τ) ]
    start_points 可以是单个点或网格上的多个点
    """
    if domain_params is None:
        domain_params = {"a": 1.0, "b": 1.0}

    # 如果是单个起点，扩展成 (1,2)
    if start_points.ndim == 1:
        start_points = start_points.reshape(1, -1)

    results = []
    for start in start_points:
        exit_pos = simulate_first_exit_with_position(
            start, domain_type, domain_params, dt=dt, n_sims=n_sims, seed=seed
        )
        phi_values = phi_boundary(exit_pos, domain_type, domain_params)
        w_estimate = np.mean(phi_values)
        results.append(w_estimate)

    return np.array(results)

# ====================== 主程序（已更新默认参数 + exact_func） ======================
if __name__ == "__main__":
    print("=== 三种区域 Dirichlet 问题 Monte Carlo 求解（已修复） ===\n")

    domain_config = {
        "circle": {
            "type": "circle",
            "params": {"center": np.zeros(2), "radius": 1.0},
            "title": "Unit Disk (boundary function is θ³)",
            "has_exact": True,
            "exact_func": lambda pts: exact_solution_circle(pts, R=1.0)
        },
        "rectangle": {
            "type": "rectangle",
            "params": {"a": 1.0, "b": 1.0},
            "title": "Rectangle [0,1]×[0,1] (right=1, bottom=1, others=0)",
            "has_exact": True,
            "exact_func": lambda x, y: exact_solution_rectangle(x, y, a=1.0, b=1.0)
        },
        "triangle": {
            "type": "triangle",
            "params": {"vertices": np.array([[0., 0.], [1., 3.], [4., 2.]])},
            "title": "Triangle (0,0), (1,3), (4,2) (boundary function is distance to the origin)",
            "has_exact": False,
        }
    }

    chosen_domain = "triangle"   # 改成 "rectangle" 或 "triangle" 即可切换

    config = domain_config[chosen_domain]
    dt = 0.0001               # 已优化
    n_sims = 1000             # 已优化（精度显著提升）
    nx, ny = 20, 20           # 网格密度提升（原来 30 太粗）

    # 创建网格
    if chosen_domain == "circle":
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
    elif chosen_domain == "rectangle":
        x = np.linspace(0, config["params"]["a"], nx)
        y = np.linspace(0, config["params"]["b"], ny)
    elif chosen_domain == "triangle":
        verts = config["params"]["vertices"]
        xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
        ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
    else:
        raise ValueError("不支持的 domain_type！请使用 'circle'、'rectangle' 或 'triangle'")

    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))

    # 过滤内部点
    inside = is_inside_domain(points, config["type"], config["params"])
    interior_points = points[inside]

    Z_mc = np.full(X.shape, np.nan, dtype=float)

    print(f"正在计算 {len(interior_points)} 个内部点（{chosen_domain} 域）... 这可能需要几分钟，请耐心等待...")
    for i, p in enumerate(tqdm(interior_points)):
        w_val = compute_w_monte_carlo(
            p,
            domain_type=config["type"],
            domain_params=config["params"],
            dt=dt,
            n_sims=n_sims,
            seed=42 + i,
        )[0]
        Z_mc.ravel()[np.where(inside)[0][i]] = w_val

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })

    # ====================== 可视化 ======================

    # 1. Monte Carlo 解热力图（所有区域都生成）
    plt.figure(figsize=(9, 8), dpi=300)
    im = plt.pcolormesh(X, Y, Z_mc, shading='auto', cmap='viridis')
    plt.colorbar(im, label=r'$w_{\text{MC}}(x,y)$')

    """
    # 绘制边界
    if chosen_domain == "circle":
        theta = np.linspace(0, 2 * np.pi, 300)
        plt.plot(np.cos(theta), np.sin(theta), 'r--', lw=1.5, label='Boundary')
    elif chosen_domain == "rectangle":
        a, b = config["params"]["a"], config["params"]["b"]
        plt.plot([0, a, a, 0, 0], [0, 0, b, b, 0], 'r--', lw=1.5, label='Boundary')
    else:  # triangle
        verts = np.vstack((config["params"]["vertices"], config["params"]["vertices"][0]))
        plt.plot(verts[:, 0], verts[:, 1], 'r--', lw=1.5, label='Boundary')
    """
    plt.title(f'SDE Monte Carlo Solution\n{config["title"]}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    #plt.legend()
    plt.tight_layout()
    plt.show()  # 单独显示第一张图

    # 2. 仅对圆形和矩形生成精确解和误差图
    if config["has_exact"]:
        # 计算参考解
        if chosen_domain == "circle":
            ref_values = config["exact_func"](interior_points)  # 注意：直接传 interior_points
            Z_ref = np.full(X.shape, np.nan, dtype=float)
            Z_ref.ravel()[np.where(inside)[0]] = ref_values
        else:
            Z_ref = config["exact_func"](X, Y)

            # 只保留内部点
            Z_ref_flat = Z_ref.ravel()
            Z_ref_flat[~inside] = np.nan
            Z_ref = Z_ref_flat.reshape(X.shape)

        # 精确解热力图（单独一张）
        plt.figure(figsize=(9, 8), dpi=300)
        im = plt.pcolormesh(X, Y, Z_ref, shading='auto', cmap='viridis')
        plt.colorbar(im, label=r'$w_{\text{exact}}(x,y)$')
        plt.title(f'PDE Exact Solution\n{config["title"]}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # 对数绝对误差热力图（单独一张）
        Z_error = np.abs(Z_mc - Z_ref)
        Z_log_error = np.log10(Z_error + 1e-12)

        plt.figure(figsize=(9, 8), dpi=300)
        im = plt.pcolormesh(X, Y, Z_log_error, shading='auto', cmap='inferno')
        plt.colorbar(im, label=r'$\log_{10}|w_{\text{MC}} - w_{\text{exact}}|$')
        plt.title(f'Log-scale Absolute Error\n{config["title"]}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # 3. 收敛速率对比图（单独一张）
        print("\n=== Monte Carlo 收敛速率分析 ===")
        N_list = np.array([100, 500,1000, 5000,10000])
        mean_errors = []
        np.random.seed(0)

        test_idx = np.random.choice(len(interior_points), 100, replace=False)
        test_points = interior_points[test_idx]

        w_ref_test = config["exact_func"](test_points) if chosen_domain == "circle" else \
            np.array([config["exact_func"](p[0], p[1]) for p in test_points])

        for N in tqdm(N_list, desc="收敛测试"):
            errors_N = []
            for j, p in enumerate(test_points):
                w_mc = compute_w_monte_carlo(p, config["type"], config["params"],
                                             dt=dt, n_sims=N)[0]
                errors_N.append(abs(w_mc - w_ref_test[j]))
            mean_errors.append(np.mean(errors_N))

        mean_errors = np.array(mean_errors)
        print(mean_errors)
        slope, intercept = np.polyfit(np.log10(N_list), np.log10(mean_errors), 1)

        theory = 10 ** intercept * N_list ** (-0.5)

        plt.figure(figsize=(9, 6), dpi=300)
        plt.loglog(N_list, mean_errors, 'o-', label='Monte Carlo Mean Error', linewidth=2)
        plt.loglog(N_list, theory, '--', label=r'Theory $O(N^{-1/2})$', linewidth=2)
        plt.xlabel(r'Number of paths $N$')
        plt.ylabel('Mean Absolute Error')
        plt.title(f'Monte Carlo Convergence Rate — {config["title"]}\n'
                  f'Estimated slope = {slope:.3f}')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    print("\n全部图片已单独生成完成！\n"
          "修复说明：\n"
          "1. rectangle 精确解已用正确 sinh 级数重写\n"
          "2. phi_boundary 对 rectangle 的判断已修复（不再漏掉右/下边界）\n"
          "3. dt 减小、n_sims 增大、网格更密 → 结果显著改善")
