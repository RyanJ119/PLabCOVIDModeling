import numpy as np
import matplotlib.pyplot as plt
import os, imageio.v3 as iio


# =========================
# 1) Model & grid settings
# =========================
A_max = 100.0     # maximum age (years)
n_age = 400       # number of age grid points
da = A_max / n_age
ages = np.linspace(0, A_max - da, n_age)  # cell-centered ages

T_max = 200.0     # simulation time (years)
dt = 0.20 * da    # CFL: dt <= da (here 0.2 * da is safe)
n_steps = int(np.ceil(T_max / dt))
save_every = max(1, n_steps // 400)  # thin saving for plotting

# =========================
# 2) Demography & disease
# =========================
L = 80.0  # average lifespan (years)
def mu(a):
    """Age-specific mortality. Here: baseline 1/L with mild senescence."""
    base = 1.0 / L
    senescence = 0.0 + 0.0008 * np.maximum(a - 60.0, 0.0)  # optional bump after 60
    return base + senescence

def gamma(a):
    """Recovery rate (1/infectious duration). Constant here: ~7 days ~ 0.0192 y."""
    return 0.0192

def beta(a):
    """Age-specific infectiousness/susceptibility scale."""
    return 0.8  # constant; feel free to make it age-dependent

# Births: choose per-capita birth rate to roughly balance deaths in disease-free state
b_rate = 1.0 / L  # per-capita births per year
def births(N_t):
    """Boundary S(t,0) as a density (people per unit age)."""
    return b_rate * N_t

# =========================
# 3) Age-mixing kernel K(a, a')
# =========================
def make_contact_kernel(ages, a_scale=15.0):
    """
    Exponential age-difference kernel:
      K(a,a') ~ exp(-|a-a'|/a_scale)
    We'll row-normalize so each row integrates to 1 over age.
    """
    A = ages[:, None]
    Aprime = ages[None, :]
    K = np.exp(-np.abs(A - Aprime) / a_scale)
    # Row-normalize so sum_j K_ij * da = 1
    row_sums = K.sum(axis=1, keepdims=True) * da
    K /= np.maximum(row_sums, 1e-12)
    return K

K = make_contact_kernel(ages, a_scale=12.0)  # somewhat assortative mixing

# =========================
# 4) Initial conditions
# =========================
# Start with a demographically plausible S(a), tiny infected seed around age 30
N0 = 1.0  # total population scale (arbitrary units)
S0 = (np.exp(-ages / L) / L)  # exponential age density that integrates to ~1
S0 = N0 * S0 / (S0.sum() * da)  # normalize to total N0

I0 = np.zeros_like(S0)
mid = np.argmin(np.abs(ages - 30.0))
I0[mid-1:mid+2] = 1e-4 * N0 / (3 * da)  # small bump around age~30
S0 = np.maximum(S0 - I0, 0.0)
R0 = np.zeros_like(S0)

S, I, R = S0.copy(), I0.copy(), R0.copy()

# For plotting & diagnostics
ts, Ns, Is, Rs, Rt = [], [], [], [], []   # totals over time
snap_ages, snap_S, snap_I, snap_R = [], [], [], []


snap_N = []  # total age distribution snapshots (S+I+R)# =========================
# 5) Time stepping (upwind in age, forward Euler in time)
# =========================
for step in range(n_steps + 1):
    t = step * dt

    # Totals and force of infection
    N_tot = (S + I + R).sum() * da
    # Force of infection: lambda(a) = beta(a) * (K * I)(a) / N
    # (K @ I) ~ integral over a' of K(a,a') I(a') da'
    KI = K @ (I) * da
    lam = beta(ages) * KI / max(N_tot, 1e-15)

    # Save for plots
    if step % save_every == 0:
        ts.append(t)
        Ns.append(N_tot)
        Is.append((I).sum() * da)
        Rs.append((R).sum() * da)
        Rt.append(((gamma(ages) * I).sum() * da))  # total recoveries per unit time
        snap_ages.append(ages.copy())
        snap_S.append(S.copy())
        snap_I.append(I.copy())
        snap_R.append(R.copy())
        snap_N.append((S+I+R).copy())

    # Right-hand sides (without advection term)
    mu_a = mu(ages)
    gam_a = gamma(ages)

    dS = -lam * S - mu_a * S
    dI =  lam * S - (gam_a + mu_a) * I
    dR =  gam_a * I - mu_a * R

    # Upwind advection in age: ∂a X ≈ (X[i] - X[i-1]) / da; boundary X[0] set by inflow
    # First do a provisional Euler update for reaction (no advection)
    S_star = S + dt * dS
    I_star = I + dt * dI
    R_star = R + dt * dR

    # Then apply age-advection (shift "to the right" in age)
    # X^{n+1}_i = X_star_i - (dt/da) * (X_star_i - X_star_{i-1})
    # with boundary X_{0} set to inflow boundary condition.

    # Susceptibles: boundary is births(N_tot)
    S_next = np.empty_like(S)
    S_next[0] = births(N_tot)                # boundary value at age 0 (density)
    S_next[1:] = S_star[1:] - (dt/da) * (S_star[1:] - S_star[:-1])

    # Infecteds: no infected births
    I_next = np.empty_like(I)
    I_next[0] = 0.0
    I_next[1:] = I_star[1:] - (dt/da) * (I_star[1:] - I_star[:-1])

    # Recovereds: no recovered births
    R_next = np.empty_like(R)
    R_next[0] = 0.0
    R_next[1:] = R_star[1:] - (dt/da) * (R_star[1:] - R_star[:-1])

    # Clip small negatives from numerical diffusion
    S, I, R = np.maximum(S_next, 0.0), np.maximum(I_next, 0.0), np.maximum(R_next, 0.0)

# =========================
# 6) Plots
# =========================
# Totals over time
plt.figure()
plt.plot(ts, Ns, label="Total population")
plt.plot(ts, Is, label="Total infected")
plt.plot(ts, Rs, label="Total recovered")
plt.xlabel("Time (years)")
plt.ylabel("Population (integrated over age)")
plt.legend()
plt.title("Age-structured SIR totals over time")
plt.show()

# Age profiles at a few snapshots
idxs = np.linspace(0, len(snap_ages)-1, 5, dtype=int)
for k in idxs:
    plt.figure()
    plt.plot(snap_ages[k], snap_S[k], label=f"S, t={ts[k]:.1f}")
    plt.plot(snap_ages[k], snap_I[k], label=f"I, t={ts[k]:.1f}")
    plt.plot(snap_ages[k], snap_R[k], label=f"R, t={ts[k]:.1f}")
    plt.xlabel("Age (years)")
    plt.ylabel("Density per unit age")
    plt.legend()
    plt.title("Age distributions at selected times")
    plt.show()
    
idxs2 = [0, 10, 20, 30, 40, 50] 
    
for k in idxs2:
    plt.figure()
    plt.plot(snap_ages[k], snap_I[k], label=f"I, t={ts[k]:.1f}")
    plt.xlabel("Age (years)")
    plt.ylabel("Density per unit age")
    plt.legend()
    plt.title("Age distributions at selected times")
    plt.show()
        
    
    
    
plt.figure()
plt.plot(ages, S0, color='navy', lw=2)
plt.xlabel("Age (years)")
plt.ylabel("Initial S₀(a) density")
plt.title("Initial age distribution of susceptibles")
plt.grid(True, alpha=0.3)
plt.show()

    

plt.figure()
plt.plot(ages, I0, color='navy', lw=2)
plt.xlabel("Age (years)")
plt.ylabel("Initial S₀(a) density")
plt.title("Initial age distribution of susceptibles")
plt.grid(True, alpha=0.3)
plt.show()


# =========================
# 8) GIF: Total age distribution snapshots every 5 years
# =========================
try:
    import numpy as np, os, imageio.v3 as iio, matplotlib.pyplot as plt
    MAKE_GIF = True
except Exception as e:
    MAKE_GIF = False
    print("GIF prerequisites not met:", e)

if MAKE_GIF and len(ts) > 1 and len(snap_N) == len(ts):
    dt_gif = 5.0  # years between frames
    ts_arr = np.array(ts)
    snap_N_arr = np.array(snap_N)   # shape: [num_snaps, n_age]

    # Select target times and map to closest stored index
    snap_times = np.arange(0.0, ts_arr[-1] + 1e-9, dt_gif)
    idxs = [int(np.argmin(np.abs(ts_arr - t))) for t in snap_times]

    out_dir = "frames_age_dist"
    os.makedirs(out_dir, exist_ok=True)
    ymax = 1.05 * np.max(snap_N_arr)

    frame_paths = []
    for t_target, k in zip(snap_times, idxs):
        fig = plt.figure()
        plt.plot(ages, snap_N_arr[k])
        plt.xlim(0, max(ages))
        plt.ylim(0, ymax)
        plt.xlabel("Age (years)")
        plt.ylabel("Population density per unit age")
        plt.title(f"Total age distribution N(a,t) at t = {ts_arr[k]:.1f} years")
        fname = os.path.join(out_dir, f"frame_{ts_arr[k]:08.3f}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        frame_paths.append(fname)

    gif_path = "age_distribution.gif"
    frames = [iio.imread(p) for p in frame_paths]
    iio.imwrite(gif_path, frames, duration=0.25, loop=0)
    print(f"[OK] GIF written to: {gif_path}")
else:
    print("Skipping GIF creation: missing snapshots or prerequisites.")





