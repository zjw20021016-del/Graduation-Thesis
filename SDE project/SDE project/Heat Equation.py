import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mpmath as mp
from numba import njit

# ====================== 1. 边界函数 φ ======================
@njit(fastmath=True)
def phi_heat(x):
    result = np.zeros_like(x)
    mask = np.abs(x) <= 1.0
    result[mask] = np.cos(np.pi * x[mask])
    return result


# ====================== 2. PDE 精确解 ======================
def exact_heat_pde(x, s=0.0, T=1.0, dps=18):
    tau = T - s
    if tau <= 0:
        return float(phi_heat(np.array([x]))[0])

    x_mp = mp.mpf(x)
    tau_mp = mp.mpf(tau)

    def integrand(y):
        y_mp = mp.mpf(y)
        kernel = (1 / mp.sqrt(2 * mp.pi * tau_mp)) * mp.exp(-y_mp**2 / (2 * tau_mp))
        phi_val = phi_heat(np.array([float(x_mp - y_mp)]))[0]
        return kernel * mp.mpf(phi_val)

    mp.mp.dps = dps
    integral = mp.quad(integrand, [-mp.inf, mp.inf])
    return float(integral)


# ====================== 3. numba Monte Carlo ======================
@njit(fastmath=True)
def monte_carlo_core(x0, steps, n_paths, sqrt_dt):
    positions = np.full(n_paths, x0, dtype=np.float64)
    for _ in range(steps):
        dW = sqrt_dt * np.random.randn(n_paths)
        positions += dW
    return positions


# ====================== 4. SDE 求解器 ======================
class HeatSDESolver:
    def __init__(self, dt=0.001, T=1.0):
        self.dt = dt
        self.T = T
        self.sqrt_dt = np.sqrt(dt)

    def compute_w(self, x0, s=0.0, n_paths=10000):
        tau = self.T - float(s)
        if tau <= 0:
            return float(phi_heat(np.array([x0]))[0]), 0.0

        steps = int(tau / self.dt)
        positions = monte_carlo_core(float(x0), steps, n_paths, self.sqrt_dt)

        values = phi_heat(positions)
        mean = float(np.mean(values))
        std_err = float(np.std(values) / np.sqrt(n_paths))
        return mean, std_err


# ====================== 主程序 ======================
if __name__ == "__main__":
    T = 1.0
    dt = 0.001

    n_paths_grid = 10000
    ns_grid = 30
    nx_grid = 120

    solver = HeatSDESolver(dt=dt, T=T)

    print("=== Heat Equation: SDE vs PDE (Publication Version) ===\n")

    # ====================== 网格 ======================
    s_grid = np.linspace(0, T, ns_grid)
    x_grid = np.linspace(-3, 3, nx_grid)

    S, X = np.meshgrid(s_grid, x_grid)

    Z_sde = np.zeros_like(S)
    Z_exact = np.zeros_like(S)

    print("Computing grid...")
    for i in tqdm(range(nx_grid)):
        for j in range(ns_grid):
            Z_sde[i, j] = solver.compute_w(
                x_grid[i], s=s_grid[j], n_paths=n_paths_grid
            )[0]

            Z_exact[i, j] = exact_heat_pde(
                x_grid[i], s=s_grid[j], T=T, dps=15
            )

    # ====================== 误差 ======================
    Z_error = np.abs(Z_sde - Z_exact)
    print(f"10000 Paths Mean error {np.mean(Z_error):.2e}  (Max {np.max(Z_error):.2e})")
    vmin = min(Z_sde.min(), Z_exact.min())
    vmax = max(Z_sde.max(), Z_exact.max())

    print("\nGenerating publication figures...")

    # ====================== SDE ======================
    plt.figure(figsize=(8, 6), dpi=300)
    im = plt.pcolormesh(S, X, Z_sde, shading='auto',
                        cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=r'$u(s,x)$')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$x$')
    plt.title('SDE Monte Carlo Solution')
    plt.tight_layout()
    plt.show()

    # ====================== PDE ======================
    plt.figure(figsize=(8, 6), dpi=300)
    im = plt.pcolormesh(S, X, Z_exact, shading='auto',
                        cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=r'$u(s,x)$')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$x$')
    plt.title('Exact PDE Solution')
    plt.tight_layout()
    plt.show()

    # ====================== 误差（log） ======================
    plt.figure(figsize=(8, 6), dpi=300)

    Z_log_error = np.log10(Z_error + 1e-10)
    vmin_err = np.percentile(Z_log_error, 5)
    vmax_err = np.percentile(Z_log_error, 95)

    im = plt.pcolormesh(S, X, Z_log_error, shading='auto',
                        cmap='inferno', vmin=vmin_err, vmax=vmax_err)

    cbar = plt.colorbar(im)
    cbar.set_label(r'$\log_{10} |u_{SDE} - u_{PDE}|$')

    plt.xlabel(r'$s$')
    plt.ylabel(r'$x$')
    plt.title('Log-scale Absolute Error')
    plt.tight_layout()
    plt.show()

    # ====================== Monte Carlo 收敛 ======================
    print("\nMonte Carlo convergence analysis...")

    N_list = np.array([100,1000,10000,100000])
    mean_errors = []

    np.random.seed(0)
    test_points_mc = np.random.uniform(-2, 2, 80)

    all_errors = []

    for N in N_list:
        errors_N = []
        for x in test_points_mc:
            w_mc, _ = solver.compute_w(x, s=0.0, n_paths=N)
            w_exact = exact_heat_pde(x, s=0.0)
            errors_N.append(abs(w_mc - w_exact))

        mean_errors.append(np.mean(errors_N))
        all_errors.append(errors_N)

    mean_errors = np.array(mean_errors)

    # 拟合收敛阶
    slope, intercept = np.polyfit(np.log(N_list), np.log(mean_errors), 1)
    print(f"Estimated convergence rate: {slope:.3f} (theory: -0.5)")

    theory = np.exp(intercept) * N_list**(-0.5)

    # 收敛图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.loglog(N_list, mean_errors, 'o-', label='Monte Carlo Error')
    plt.loglog(N_list, theory, '--', label=r'Theory $O(N^{-1/2})$')
    plt.xlabel(r'$N$')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Convergence Rate (slope ≈ {slope:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 箱线图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.boxplot(all_errors, labels=N_list, showfliers=False)
    plt.xscale('log')
    plt.xlabel(r'$N$')
    plt.ylabel('Absolute Error')
    plt.title('Error Distribution')
    plt.tight_layout()
    plt.show()
