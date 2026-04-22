import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import math

# ====================== 1. 正态分布 CDF ======================
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ====================== 2. Black-Scholes 精确解 ======================
def bs_call_price(S, K, T, t, r, sigma):
    tau = T - t
    if tau <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return S * norm_cdf(d1) - K * np.exp(-r * tau) * norm_cdf(d2)


def bs_put_price(S, K, T, t, r, sigma):
    tau = T - t
    if tau <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return K * np.exp(-r * tau) * norm_cdf(-d2) - S * norm_cdf(-d1)


# ====================== 3. Euler-Maruyama ======================
@njit(fastmath=True)
def euler_maruyama_gbm(S0, r, sigma, dt, steps, n_paths):
    S = np.full(n_paths, S0)
    for _ in range(steps):
        dW = np.sqrt(dt) * np.random.randn(n_paths)
        S = S + r * S * dt + sigma * S * dW
    return S


# ====================== 4. Monte Carlo ======================
def mc_option_price(S0, K, T, t, r, sigma, n_paths, dt, option):
    tau = T - t
    if tau <= 0:
        return max(S0 - K, 0) if option == "call" else max(K - S0, 0)

    steps = int(tau / dt)
    S_T = euler_maruyama_gbm(S0, r, sigma, dt, steps, n_paths)

    if option == "call":
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)

    return np.exp(-r * tau) * np.mean(payoff)


# ====================== 主程序 ======================
if __name__ == "__main__":

    # 参数
    T = 1.0
    r = 0.01
    sigma = 0.2
    K = 1.0

    dt = 0.001
    n_paths_grid = 10000

    ns = 30
    nS = 120

    t_grid = np.linspace(0, T, ns)
    S_grid = np.linspace(0.01, 3, nS)

    T_grid, S_mesh = np.meshgrid(t_grid, S_grid)

    call_mc = np.zeros_like(S_mesh)
    call_exact = np.zeros_like(S_mesh)

    print("Computing grid...")

    for i in tqdm(range(nS)):
        for j in range(ns):
            S = S_grid[i]
            t = t_grid[j]

            call_mc[i, j] = mc_option_price(S, K, T, t, r, sigma,
                                           n_paths_grid, dt, "call")

            call_exact[i, j] = bs_call_price(S, K, T, t, r, sigma)

    # ====================== 误差 ======================
    call_error = np.abs(call_mc - call_exact)
    print(f"看涨期权 10000 Paths Mean error {np.mean(call_error):.2e}  (Max {np.max(call_error):.2e})")
    # 统一颜色范围
    vmin = min(call_mc.min(), call_exact.min())
    vmax = max(call_mc.max(), call_exact.max())

    print("\nGenerating publication-quality figures...")

    # ====================== 1. SDE ======================
    plt.figure(figsize=(8, 6), dpi=300)
    im = plt.pcolormesh(T_grid, S_mesh, call_mc,
                        shading='auto', cmap='viridis',
                        vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Price')
    plt.xlabel('t')
    plt.ylabel('S')
    plt.title('Call Option (Monte Carlo)')
    plt.tight_layout()
    plt.show()

    # ====================== 2. PDE ======================
    plt.figure(figsize=(8, 6), dpi=300)
    im = plt.pcolormesh(T_grid, S_mesh, call_exact,
                        shading='auto', cmap='viridis',
                        vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Price')
    plt.xlabel('t')
    plt.ylabel('S')
    plt.title('Call Option (Exact)')
    plt.tight_layout()
    plt.show()

    # ====================== 3. log误差 ======================
    plt.figure(figsize=(8, 6), dpi=300)

    Z_log = np.log10(call_error + 1e-10)
    vmin_err = np.percentile(Z_log, 5)
    vmax_err = np.percentile(Z_log, 95)

    im = plt.pcolormesh(T_grid, S_mesh, Z_log,
                        shading='auto', cmap='inferno',
                        vmin=vmin_err, vmax=vmax_err)

    cbar = plt.colorbar(im)
    cbar.set_label('log10 error')

    plt.xlabel('t')
    plt.ylabel('S')
    plt.title('Call Option Log Error (MC vs Exact)')
    plt.tight_layout()
    plt.show()

    # ====================== Monte Carlo 收敛 ======================
    print("\nMonte Carlo convergence...")

    N_list = np.array([100,1000,10000,100000])
    mean_errors = []
    all_errors = []

    np.random.seed(0)
    test_S = np.random.uniform(0.5, 2.0, 50)

    for N in N_list:
        errors = []
        for S in test_S:
            mc = mc_option_price(S, K, T, 0.0, r, sigma, N, dt, "call")
            exact = bs_call_price(S, K, T, 0.0, r, sigma)
            errors.append(abs(mc - exact))

        mean_errors.append(np.mean(errors))
        all_errors.append(errors)

    mean_errors = np.array(mean_errors)

    # 拟合斜率
    slope, intercept = np.polyfit(np.log(N_list), np.log(mean_errors), 1)

    print(f"slope ≈ {slope:.3f} (theory = -0.5)")

    theory = np.exp(intercept) * N_list**(-0.5)

    # 收敛图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.loglog(N_list, mean_errors, 'o-', label='MC error')
    plt.loglog(N_list, theory, '--', label='O(N^{-1/2})')
    plt.xlabel('Number of paths N')
    plt.ylabel('Mean absolute Error')
    plt.title(f'Call Option Convergence (slope ≈ {slope:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    put_mc = np.zeros_like(S_mesh)
    put_exact = np.zeros_like(S_mesh)

    print("Computing grid...")

    for i in tqdm(range(nS)):
        for j in range(ns):
            S = S_grid[i]
            t = t_grid[j]

            put_mc[i, j] = mc_option_price(S, K, T, t, r, sigma,
                                            n_paths_grid, dt, "put")

            put_exact[i, j] = bs_put_price(S, K, T, t, r, sigma)

    # ====================== 误差 ======================
    put_error = np.abs(put_mc - put_exact)
    print(f"看跌期权 10000 Paths Mean error {np.mean(put_error):.2e}  (Max {np.max(put_error):.2e})")
    # 统一颜色范围
    vmin = min(put_mc.min(), put_exact.min())
    vmax = max(put_mc.max(), put_exact.max())

    print("\nGenerating publication-quality figures...")

    # ====================== 1. SDE ======================
    plt.figure(figsize=(8, 6), dpi=300)
    im = plt.pcolormesh(T_grid, S_mesh, put_mc,
                        shading='auto', cmap='viridis',
                        vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Price')
    plt.xlabel('t')
    plt.ylabel('S')
    plt.title('Put Option (Monte Carlo)')
    plt.tight_layout()
    plt.show()

    # ====================== 2. PDE ======================
    plt.figure(figsize=(8, 6), dpi=300)
    im = plt.pcolormesh(T_grid, S_mesh, put_exact,
                        shading='auto', cmap='viridis',
                        vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Price')
    plt.xlabel('t')
    plt.ylabel('S')
    plt.title('Put Option (Exact)')
    plt.tight_layout()
    plt.show()

    # ====================== 3. log误差 ======================
    plt.figure(figsize=(8, 6), dpi=300)

    Z_log = np.log10(put_error + 1e-10)
    vmin_err = np.percentile(Z_log, 5)
    vmax_err = np.percentile(Z_log, 95)

    im = plt.pcolormesh(T_grid, S_mesh, Z_log,
                        shading='auto', cmap='inferno',
                        vmin=vmin_err, vmax=vmax_err)

    cbar = plt.colorbar(im)
    cbar.set_label('log10 error')

    plt.xlabel('t')
    plt.ylabel('S')
    plt.title('Put Option Log Error (MC vs Exact)')
    plt.tight_layout()
    plt.show()

    # ====================== Monte Carlo 收敛 ======================
    print("\nMonte Carlo convergence...")

    N_list = np.array([100, 1000, 10000, 100000])
    mean_errors = []
    all_errors = []

    np.random.seed(0)
    test_S = np.random.uniform(0.5, 2.0, 50)

    for N in N_list:
        errors = []
        for S in test_S:
            mc = mc_option_price(S, K, T, 0.0, r, sigma, N, dt, "put")
            exact = bs_put_price(S, K, T, 0.0, r, sigma)
            errors.append(abs(mc - exact))

        mean_errors.append(np.mean(errors))
        all_errors.append(errors)

    mean_errors = np.array(mean_errors)

    # 拟合斜率
    slope, intercept = np.polyfit(np.log(N_list), np.log(mean_errors), 1)

    print(f"slope ≈ {slope:.3f} (theory = -0.5)")

    theory = np.exp(intercept) * N_list ** (-0.5)

    # 收敛图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.loglog(N_list, mean_errors, 'o-', label='MC error')
    plt.loglog(N_list, theory, '--', label='O(N^{-1/2})')
    plt.xlabel('Number of paths N')
    plt.ylabel('Mean absolute error')
    plt.title(f'Put Option Convergence (slope ≈ {slope:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.show()


