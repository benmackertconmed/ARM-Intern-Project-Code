import numpy as np
import matplotlib.pyplot as plt

# 4th-order monitor polynomial (Ω ← Varmₘ)
def rpad_m(varm):
    return (
          715.13226 * varm**4
        -1620.07948 * varm**3
        +1536.71187 * varm**2
        - 534.80120 * varm
        +  56.73790
    )

# Invert rpad_m(varm)=R_target via np.roots, pick the real root in [0,3]
def ideal_varm_for_resistance(R_target, vmin=0, vmax=3):
    coeffs = [
        715.13226,
       -1620.07948,
        1536.71187,
        -534.80120,
        56.73790 - R_target
    ]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.isreal(r) and vmin <= r.real <= vmax]
    if not real_roots:
        raise ValueError(f"No valid root for R={R_target}")
    mid = (vmin + vmax)/2
    return min(real_roots, key=lambda r: abs(r-mid))

# Raw test data
data = {
    "TF7":  {"load": np.arange(1,181,1)[np.isin(np.arange(1,181,1),[1]+list(range(10,191,10)))],
             "voltage": np.array([0.46,0.58,0.67,0.76,0.81,0.89,0.94,0.99,1.03,1.07,
                                  1.10,1.14,1.17,1.19,1.22,1.24,1.26,1.28,1.30])},
    "TF10": {"load": np.arange(1,181,1)[np.isin(np.arange(1,181,1),[1]+list(range(10,191,10)))],
             "voltage": np.array([0.46,0.60,0.71,0.79,0.82,0.91,0.95,0.98,1.01,1.04,
                                  1.06,1.07,1.09,1.10,1.12,1.13,1.14,1.15,1.16])},
    "TF8":  {"load": np.arange(1,181,1)[np.isin(np.arange(1,181,1),[1]+list(range(10,191,10)))],
             "voltage": np.array([0.45,0.55,0.65,0.70,0.77,0.83,0.88,0.92,0.97,1.012,
                                  1.047,1.078,1.108,1.134,1.159,1.182,1.202,1.222,1.241])}
}

# Base anchors
base_anchors = np.array([10, 150])

# All possible third points (exclude 10,150)
all_loads = data["TF7"]["load"]
candidates = [r for r in all_loads if r not in base_anchors]

best_R3     = None
best_avg_lms= np.inf

# Pre–compute ideal Varmₘ for 10Ω & 150Ω
ideal_10  = ideal_varm_for_resistance(10)
ideal_150 = ideal_varm_for_resistance(150)

# Iterate every third‐point candidate
for R3 in candidates:
    # compute ideal varm targets for anchors
    ideal_3 = ideal_varm_for_resistance(R3)
    y_targets = np.array([ideal_10, ideal_3, ideal_150])

    lms_list = []
    # compute per-transformer LMS
    for vals in data.values():
        loads   = vals["load"]
        raw_vol = vals["voltage"]

        # grab raw voltages at anchors
        v10  = raw_vol[np.where(loads==10)[0][0]]
        v3   = raw_vol[np.where(loads==R3)[0][0]]
        v150 = raw_vol[np.where(loads==150)[0][0]]
        v_anchors = np.array([v10, v3, v150])

        # fit a quadratic map raw_vol → Varmₘ
        coeffs = np.polyfit(v_anchors, y_targets, deg=2)
        poly   = np.poly1d(coeffs)

        # apply calibration & compute error
        varm_m_est = poly(raw_vol)
        rpad_est   = rpad_m(varm_m_est)
        err2 = (loads - rpad_est)**2
        lms_list.append(np.mean(err2))

    avg_lms = np.mean(lms_list)
    if avg_lms < best_avg_lms:
        best_avg_lms = avg_lms
        best_R3      = R3

# Report best third anchor
print(f"\nBest third anchor: {best_R3} Ω")
print(f"Average LMS error: {best_avg_lms:.2f} Ω² (across TF7, TF10, TF8)")

# ----------------------
# Final plotting with best_R3
# ----------------------
ideal_3    = ideal_varm_for_resistance(best_R3)
y_targets  = np.array([ideal_10, ideal_3, ideal_150])

fig, axs = plt.subplots(3, 1, figsize=(10,12), sharex=True)
lms_list  = []

for ax, (tf, vals) in zip(axs, data.items()):
    loads   = vals["load"]
    raw_vol = vals["voltage"]

    # anchor raw voltages
    v10   = raw_vol[np.where(loads==10)[0][0]]
    v3    = raw_vol[np.where(loads==best_R3)[0][0]]
    v150  = raw_vol[np.where(loads==150)[0][0]]
    v_anchors = np.array([v10, v3, v150])

    # fit & apply
    coeffs = np.polyfit(v_anchors, y_targets, deg=2)
    poly   = np.poly1d(coeffs)
    varm_m_est = poly(raw_vol)
    rpad_est   = rpad_m(varm_m_est)

    # LMS for this transformer
    errs    = loads - rpad_est
    lms_err = np.mean(errs**2)
    lms_list.append(lms_err)

    # Plot
    ax.plot(loads, loads,    'o-',  label='Measured',  color='black')
    ax.plot(loads, rpad_est, 's--', label='Estimated', color='purple')
    ax.set_title(f"{tf}: LMS = {lms_err:.2f} Ω²")
    ax.set_ylabel("Resistance (Ω)")
    ax.grid(True)
    ax.legend()

# annotate average
avg_lms = np.mean(lms_list)
fig.text(0.5, 0.04,
         f"3-Point Calibration Anchors = [10 Ω, {best_R3} Ω, 150 Ω]    "
         f"Average LMS = {avg_lms:.2f} Ω²",
         ha='center', fontsize=12)

axs[-1].set_xlabel("Load (Ω)")
plt.tight_layout(rect=[0,0.05,1,1])
plt.show()
