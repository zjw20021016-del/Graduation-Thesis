import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpmath as mp
from tqdm import tqdm
from matplotlib.path import Path
from numba import njit


# ====================== 1. 边界条件函数 φ ======================
def phi_circle(y):
    theta = np.arctan2(y[1], y[0])
    if theta < 0:
        theta += 2 * np.pi
    return np.exp(-theta ** 2)


def phi_triangle(y):
    return y[0] ** 3


# ====================== 2. 圆形精确解 ======================
def exact_circle_pde(point, R=1.0):
    p = np.asarray(point, dtype=float)
    r = np.linalg.norm(p)
    if r >= R:
        return phi_circle(p)
    theta = float(np.arctan2(p[1], p[0]))
    if theta < 0:
        theta += 2 * np.pi

    def integrand(alpha):
        alpha = mp.mpf(alpha)
        kernel = (R ** 2 - r ** 2) / (R ** 2 - 2 * R * r * mp.cos(theta - alpha) + r ** 2)
        return kernel * mp.exp(-(alpha) ** 2)

    integral = mp.quad(integrand, [0, 2 * mp.pi])
    return float(integral / (2 * mp.pi))


# ====================== 3. 向量化 is_inside 和 phi ======================
def make_circle_is_inside_vec(R=1.0):
    def is_inside_vec(pos):
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        return np.linalg.norm(pos, axis=1) < R

    return is_inside_vec


def make_phi_circle_vec():
    def phi_vec(pos):
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        theta = np.arctan2(pos[:, 1], pos[:, 0])
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        return np.exp(-theta ** 2)

    return phi_vec


def make_square_is_inside_vec(a=1.0, b=1.0):
    def is_inside_vec(pos):
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        x, y = pos[:, 0], pos[:, 1]
        return (x > 0) & (x < a) & (y > 0) & (y < b)

    return is_inside_vec


def make_phi_square_vec(a=1.0, b=1.0):
    def phi_vec(pos):
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        x = np.clip(pos[:, 0], 0, a)
        y = np.clip(pos[:, 1], 0, b)
        res = np.zeros(len(pos))
        res[(np.isclose(x, a, atol=1e-8)) | (np.isclose(y, 0, atol=1e-8))] = 1.0
        res[(np.isclose(x, 0, atol=1e-8)) | (np.isclose(y, b, atol=1e-8))] = 0.0
        return res

    return phi_vec


def make_triangle_is_inside_vec(vertices):
    path = Path(vertices)

    def is_inside_vec(pos):
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        return path.contains_points(pos, radius=1e-12)

    return is_inside_vec


def make_phi_triangle_vec():
    def phi_vec(pos):
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        return pos[:, 0] ** 3

    return phi_vec


# ====================== 4. 正方形精确解 ======================
@njit(fastmath=True)
def _exact_square_pde_jit(x, y, M=300, N=300):
    w = 0.0
    pi = np.pi
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            denom = (m * pi) ** 2 + (n * pi) ** 2
            sin_mx = np.sin(m * pi * x)
            sin_ny = np.sin(n * pi * y)
            I_L = 0.0
            I_R = (1.0 - (-1) ** n) / (n * pi)
            I_B = (1.0 - (-1) ** m) / (m * pi)
            I_T = 0.0
            bracket = (-m * pi * I_L + m * pi * ((-1) ** m) * I_R -
                       n * pi * I_B + n * pi * ((-1) ** n) * I_T)
            term = sin_mx * sin_ny / denom * bracket
            w += term
    return -4.0 * w


def exact_square_pde(point):
    x, y = float(point[0]), float(point[1])
    if x <= 0 or x >= 1 or y <= 0 or y >= 1:
        if np.isclose(x, 0) or np.isclose(y, 1):
            return 0.0
        if np.isclose(x, 1) or np.isclose(y, 0):
            return 1.0
        return 0.0
    return _exact_square_pde_jit(x, y)


# ====================== 5. SDE 求解器 ======================
class SDESolver:
    def __init__(self, is_inside_vec, phi_vec, dt=0.001, max_steps=200000):
        self.is_inside_vec = is_inside_vec
        self.phi_vec = phi_vec
        self.dt = dt
        self.max_steps = max_steps
        self.sqrt_dt = np.sqrt(dt)

    def compute_w(self, x0, n_paths=10000):
        x0 = np.asarray(x0, dtype=np.float64).reshape(1, 2)
        positions = np.tile(x0, (n_paths, 1))
        active = np.ones(n_paths, dtype=bool)
        values = np.zeros(n_paths, dtype=np.float64)

        for step in range(self.max_steps):
            if not np.any(active):
                break
            active_idx = np.flatnonzero(active)
            n_active = len(active_idx)
            dW = self.sqrt_dt * np.random.randn(n_active, 2)
            positions[active_idx] += dW
            inside = self.is_inside_vec(positions[active_idx])
            hit_mask = ~inside
            if np.any(hit_mask):
                hit_local = np.flatnonzero(hit_mask)
                hit_global = active_idx[hit_local]
                hit_pos = positions[hit_global]
                values[hit_global] = self.phi_vec(hit_pos)
                active[hit_global] = False

        mean = float(np.mean(values))
        std_err = float(np.std(values) / np.sqrt(n_paths)) if n_paths > 1 else 0.0
        return mean, std_err

    def simulate_one_path(self, x0):
        x0 = np.asarray(x0, dtype=np.float64).reshape(1, 2)
        positions = [x0.copy().flatten()]
        x = x0.copy()
        for _ in range(self.max_steps):
            dW = self.sqrt_dt * np.random.randn(1, 2)
            x += dW
            positions.append(x.flatten().copy())
            if not self.is_inside_vec(x)[0]:
                break
        return np.array(positions)


# ====================== 6. 随机点生成 ======================
def generate_random_points(n_points=100, domain="circle", **kwargs):
    np.random.seed(42)
    if domain == "circle":
        R = kwargs.get('R', 1.0)
        r = R * np.sqrt(np.random.rand(n_points))
        theta = 2 * np.pi * np.random.rand(n_points)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    elif domain == "square":
        a = kwargs.get('a', 1.0)
        b = kwargs.get('b', 1.0)
        x = a * np.random.rand(n_points)
        y = b * np.random.rand(n_points)
        return np.stack([x, y], axis=1)
    else:  # triangle
        verts = kwargs['verts']
        minx, maxx = verts[:, 0].min(), verts[:, 0].max()
        miny, maxy = verts[:, 1].min(), verts[:, 1].max()
        is_inside_vec = make_triangle_is_inside_vec(verts)
        points = []
        batch_size = n_points * 4
        while len(points) < n_points:
            x = np.random.uniform(minx, maxx, batch_size)
            y = np.random.uniform(miny, maxy, batch_size)
            cand = np.column_stack([x, y])
            inside_mask = is_inside_vec(cand)
            points.extend(cand[inside_mask])
        return np.array(points[:n_points])


# ====================== 主程序 ======================
if __name__ == "__main__":
    dt = 0.001
    nx_grid = 100  # 分辨率：可增大到 50~60（计算时间会变长）
    n_paths_grid = 10000  # 每个网格点路径数

    # ==================== 求解器初始化 ====================
    solver_square = SDESolver(make_square_is_inside_vec(), make_phi_square_vec(), dt)
    solver_circle = SDESolver(make_circle_is_inside_vec(), make_phi_circle_vec(), dt)

    verts_triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    solver_triangle = SDESolver(make_triangle_is_inside_vec(verts_triangle), make_phi_triangle_vec(), dt)

    # ==================== MC + PDE 误差输出 ====================
    points_square = generate_random_points(1000, "square")
    print("=== Square MC + PDE Comparison ===\n")
    errors_square = [abs(solver_square.compute_w(p, 10000)[0] - exact_square_pde(p)) for p in
                     tqdm(points_square, desc="Square")]
    print(f"10000 Paths Square | Mean error {np.mean(errors_square):.2e}  (Max {np.max(errors_square):.2e})")

    points_circle = generate_random_points(1000, "circle")
    print("\n=== Circle MC + PDE Comparison ===\n")
    errors_circle = [abs(solver_circle.compute_w(p, 10000)[0] - exact_circle_pde(p)) for p in
                     tqdm(points_circle, desc="Circle")]
    print(f"10000 Paths Circle | Mean error {np.mean(errors_circle):.2e}  (Max {np.max(errors_circle):.2e})")

    # ====================== 计算网格数据（含圆形误差） ======================
    print("\nGenerating grids for heatmaps and 3D error plots...")

    # Square
    x_s = np.linspace(0.02, 0.98, nx_grid)
    y_s = np.linspace(0.02, 0.98, nx_grid)
    X_square, Y_square = np.meshgrid(x_s, y_s)
    Z_sde_square = np.zeros_like(X_square)
    Z_pde_square = np.zeros_like(X_square)
    for i in tqdm(range(nx_grid), desc="Square Grid"):
        for j in range(nx_grid):
            pt = [X_square[i, j], Y_square[i, j]]
            Z_sde_square[i, j] = solver_square.compute_w(pt, n_paths_grid)[0]
            Z_pde_square[i, j] = exact_square_pde(pt)
    vmin_s = min(Z_sde_square.min(), Z_pde_square.min())
    vmax_s = max(Z_sde_square.max(), Z_pde_square.max())
    Z_error_square = np.abs(Z_sde_square - Z_pde_square)

    # Circle（新增误差计算）
    x_c = np.linspace(-1.02, 1.02, nx_grid)
    y_c = np.linspace(-1.02, 1.02, nx_grid)
    X_circle, Y_circle = np.meshgrid(x_c, y_c)
    inside_circle = make_circle_is_inside_vec()(np.stack([X_circle.ravel(), Y_circle.ravel()], axis=1)).reshape(nx_grid,
                                                                                                                nx_grid)
    Z_sde_circle = np.full((nx_grid, nx_grid), np.nan)
    Z_pde_circle = np.full((nx_grid, nx_grid), np.nan)
    Z_error_circle = np.full((nx_grid, nx_grid), np.nan)
    for i in tqdm(range(nx_grid), desc="Circle Grid"):
        for j in range(nx_grid):
            if inside_circle[i, j]:
                pt = [X_circle[i, j], Y_circle[i, j]]
                sde_val = solver_circle.compute_w(pt, n_paths_grid)[0]
                pde_val = exact_circle_pde(pt)
                Z_sde_circle[i, j] = sde_val
                Z_pde_circle[i, j] = pde_val
                Z_error_circle[i, j] = abs(sde_val - pde_val)
    valid_c = ~np.isnan(Z_error_circle)
    vmin_c = min(Z_sde_circle[valid_c].min(), Z_pde_circle[valid_c].min())
    vmax_c = max(Z_sde_circle[valid_c].max(), Z_pde_circle[valid_c].max())

    # Triangle（保持原有）
    x_t = np.linspace(-0.05, 1.05, nx_grid)
    y_t = np.linspace(-0.05, 0.95, nx_grid)
    X_triangle, Y_triangle = np.meshgrid(x_t, y_t)
    inside_triangle = make_triangle_is_inside_vec(verts_triangle)(
        np.stack([X_triangle.ravel(), Y_triangle.ravel()], axis=1)).reshape(nx_grid, nx_grid)
    Z_sde_triangle = np.full((nx_grid, nx_grid), np.nan)
    for i in tqdm(range(nx_grid), desc="Triangle Grid"):
        for j in range(nx_grid):
            if inside_triangle[i, j]:
                pt = [X_triangle[i, j], Y_triangle[i, j]]
                Z_sde_triangle[i, j] = solver_triangle.compute_w(pt, n_paths_grid)[0]

    # ====================== 可视化（每张独立） ======================
    print("\nGenerating all figures...")

    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    theta_b = np.linspace(0, 2 * np.pi, 200)

    # 1. Square Brownian Paths
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', lw=2, label='Square Boundary')
    for i in range(5):
        path = solver_square.simulate_one_path([0.5, 0.5])
        ax.plot(path[:, 0], path[:, 1], color=colors[i], lw=1.2, label=f'Path {i + 1}')
    ax.plot(0.5, 0.5, 'go', markersize=8, label='Start')
    ax.set_title('Multiple Brownian Motion Paths - Square Domain')
    ax.set_xlabel('x');
    ax.set_ylabel('y')
    ax.legend();
    ax.grid(True)
    plt.tight_layout();
    plt.show()

    # 2. Circle Brownian Paths
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(np.cos(theta_b), np.sin(theta_b), 'k-', lw=2, label='Circle Boundary (R=1)')
    for i in range(5):
        path = solver_circle.simulate_one_path([0.5, 0.0])
        ax.plot(path[:, 0], path[:, 1], color=colors[i], lw=1.2, label=f'Path {i + 1}')
    ax.plot(0.5, 0.0, 'go', markersize=8, label='Start')
    ax.set_title('Multiple Brownian Motion Paths - Circle Domain')
    ax.set_xlabel('x');
    ax.set_ylabel('y')
    ax.legend();
    ax.grid(True);
    ax.axis('equal')
    plt.tight_layout();
    plt.show()

    # 3. Square 3D Error
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_square, Y_square, Z_error_square, cmap='YlOrRd', alpha=0.95)
    plt.colorbar(surf, ax=ax, shrink=0.6, label='|SDE - PDE|')
    ax.set_title('3D Absolute Error - Square Domain\n(Light = Low Error, Dark Red = High Error)')
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.set_zlabel('Absolute Error')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout();
    plt.show()

    # 4. Circle 3D Error
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_circle, Y_circle, Z_error_circle, cmap='YlOrRd', alpha=0.95)
    plt.colorbar(surf, ax=ax, shrink=0.6, label='|SDE - PDE|')
    ax.set_title('3D Absolute Error - Circle Domain\n(Light = Low Error, Dark Red = High Error)')
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.set_zlabel('Absolute Error')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout();
    plt.show()

    # 5. Square SDE 2D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X_square, Y_square, Z_sde_square, cmap='viridis', vmin=vmin_s, vmax=vmax_s, shading='auto')
    plt.colorbar(im, label='u(x,y)')
    ax.set_title('2D Heatmap - SDE Monte Carlo (Square)')
    ax.set_aspect('equal')
    plt.tight_layout();
    plt.show()

    # 6. Square PDE 2D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X_square, Y_square, Z_pde_square, cmap='viridis', vmin=vmin_s, vmax=vmax_s, shading='auto')
    plt.colorbar(im, label='u(x,y)')
    ax.set_title('2D Heatmap - PDE Exact (Square)')
    ax.set_aspect('equal')
    plt.tight_layout();
    plt.show()

    # 7. Circle SDE 2D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X_circle, Y_circle, Z_sde_circle, cmap='viridis', vmin=vmin_c, vmax=vmax_c, shading='auto')
    plt.colorbar(im, label='u(x,y)')
    ax.plot(np.cos(theta_b), np.sin(theta_b), 'k-', lw=1.5)
    ax.set_title('2D Heatmap - SDE Monte Carlo (Circle)')
    ax.set_aspect('equal')
    plt.tight_layout();
    plt.show()

    # 8. Circle PDE 2D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X_circle, Y_circle, Z_pde_circle, cmap='viridis', vmin=vmin_c, vmax=vmax_c, shading='auto')
    plt.colorbar(im, label='u(x,y)')
    ax.plot(np.cos(theta_b), np.sin(theta_b), 'k-', lw=1.5)
    ax.set_title('2D Heatmap - PDE Exact (Circle)')
    ax.set_aspect('equal')
    plt.tight_layout();
    plt.show()

    # 9. Triangle SDE 2D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X_triangle, Y_triangle, Z_sde_triangle, cmap='viridis', shading='auto')
    plt.colorbar(im, label='u(x,y)')
    ax.plot([0, 1, 0.5, 0], [0, 0, 0.866, 0], 'k-', lw=1.5)
    ax.set_title('2D Heatmap - SDE Monte Carlo (Triangle)')
    ax.set_aspect('equal')
    plt.tight_layout();
    plt.show()

    # ====================== 收敛性分析：矩形域不同路径数 N 的误差分布 ======================
    print("\n=== Convergence Analysis: Error vs Number of Paths (Square Domain) ===")

    # 定义要测试的路径数（幂次增长）
    N_list = np.array([100, 1000, 10000, 100000])
    mean_errors = []
    all_errors = []  # 每个 N 对应的所有点误差列表

    # 在正方形内部随机生成固定的一组测试点（避免每次重新生成）
    np.random.seed(42)
    n_test_points = 50
    test_points = generate_random_points(n_test_points, "square")

    # 预先计算每个测试点的精确解（加速）
    exact_vals = np.array([exact_square_pde(pt) for pt in test_points])

    for N in tqdm(N_list, desc="Convergence Test"):
        errors = []
        for i, pt in enumerate(test_points):
            # 计算 SDE 数值解（只返回均值，忽略标准差）
            mc_val, _ = solver_square.compute_w(pt, n_paths=N)
            err = abs(mc_val - exact_vals[i])
            errors.append(err)
        mean_errors.append(np.mean(errors))
        all_errors.append(errors)

    mean_errors = np.array(mean_errors)

    # 拟合 log(mean_error) ~ log(N) 的斜率
    slope, intercept = np.polyfit(np.log(N_list), np.log(mean_errors), 1)
    print(f"\n实测收敛斜率: {slope:.3f} (理论值: -0.500)")

    # 理论曲线 (C / sqrt(N))，C 由第一个点的平均误差决定
    C_fit = mean_errors[0] * np.sqrt(N_list[0])
    theory = C_fit * N_list ** (-0.5)

    # 绘制箱线图（误差分布）
    plt.figure(figsize=(8, 6), dpi=300)
    bp = plt.boxplot(all_errors, labels=N_list, showfliers=False,
                     patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    plt.xscale('log')
    # 设置横坐标刻度和标签为 10^2, 10^3, 10^4, 10^5
    ax = plt.gca()
    ax.set_xticks(N_list)
    ax.set_xticklabels([f'$10^{{{int(np.log10(n))}}}$' for n in N_list])
    plt.xlabel('Number of paths N')
    plt.ylabel('Absolute error')
    plt.title('Error Distribution at Different N (Square Domain)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 绘制平均误差随 N 的变化（双对数坐标 + 理论曲线）
    plt.figure(figsize=(8, 6), dpi=300)
    plt.loglog(N_list, mean_errors, 'o-', label='MC error')
    plt.loglog(N_list, theory, '--', label='O(N^{-1/2})')
    plt.xlabel('Number of paths N')
    plt.ylabel('Mean absolute error')
    plt.title(f'Convergence of Monte Carlo (Square, {n_test_points} random points)')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ====================== 收敛性分析：圆形域不同路径数 N 的误差分布 ======================
    print("\n=== Convergence Analysis: Error vs Number of Paths (Circle Domain) ===")

    # 定义要测试的路径数（幂次增长）
    N_list = np.array([100, 1000, 10000, 100000])
    mean_errors = []
    all_errors = []  # 每个 N 对应的所有点误差列表

    # 在正方形内部随机生成固定的一组测试点（避免每次重新生成）
    np.random.seed(42)
    n_test_points = 50
    test_points = generate_random_points(n_test_points, "Circle")

    # 预先计算每个测试点的精确解（加速）
    exact_vals = np.array([exact_circle_pde(pt) for pt in test_points])

    for N in tqdm(N_list, desc="Convergence Test"):
        errors = []
        for i, pt in enumerate(test_points):
            # 计算 SDE 数值解（只返回均值，忽略标准差）
            mc_val, _ = solver_circle.compute_w(pt, n_paths=N)
            err = abs(mc_val - exact_vals[i])
            errors.append(err)
        mean_errors.append(np.mean(errors))
        all_errors.append(errors)

    mean_errors = np.array(mean_errors)

    # 拟合 log(mean_error) ~ log(N) 的斜率
    slope, intercept = np.polyfit(np.log(N_list), np.log(mean_errors), 1)
    print(f"\n实测收敛斜率: {slope:.3f} (理论值: -0.500)")

    # 理论曲线 (C / sqrt(N))，C 由第一个点的平均误差决定
    C_fit = mean_errors[0] * np.sqrt(N_list[0])
    theory = C_fit * N_list ** (-0.5)

    # 绘制箱线图（误差分布）
    plt.figure(figsize=(8, 6), dpi=300)
    bp = plt.boxplot(all_errors, labels=N_list, showfliers=False,
                     patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    plt.xscale('log')
    # 设置横坐标刻度和标签为 10^2, 10^3, 10^4, 10^5
    ax = plt.gca()
    ax.set_xticks(N_list)
    ax.set_xticklabels([f'$10^{{{int(np.log10(n))}}}$' for n in N_list])
    plt.xlabel('Number of paths N')
    plt.ylabel('Absolute error')
    plt.title('Error Distribution at Different N (Circle Domain)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 绘制平均误差随 N 的变化（双对数坐标 + 理论曲线）
    plt.figure(figsize=(8, 6), dpi=300)
    plt.loglog(N_list, mean_errors, 'o-', label='MC error')
    plt.loglog(N_list, theory, '--', label='O(N^{-1/2})')
    plt.xlabel('Number of paths N')
    plt.ylabel('Mean absolute error')
    plt.title(f'Convergence of Monte Carlo (Square, {n_test_points} random points)')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()