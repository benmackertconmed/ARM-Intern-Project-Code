import numpy as np
import matplotlib.pyplot as plt
import itertools

# 4th-order monitor polynomial (Ω ← Varmₘ)
def rpad_m(varm):
    return (
          715.13226 * varm**4
        -1620.07948 * varm**3
        +1536.71187 * varm**2
        - 534.80120 * varm
        +  56.73790
    )

# Invert rpad_m(varm) = R_target via np.roots, pick real root in [0,3]
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

# Raw test data: load points & monitor voltages for each transformer
data = {
    "TF7":  {"load":    np.array([  1,10,20,30,40,50,60,70,80,90,
                                   100,110,120,130,140,150,160,170,180]),
             "voltage": np.array([0.46,0.58,0.67,0.76,0.81,0.89,0.94,0.99,1.03,1.07,
                                  1.10,1.14,1.17,1.19,1.22,1.24,1.26,1.28,1.30])},
    "TF10": {"load":    np.array([  1,10,20,30,40,50,60,70,80,90,
                                   100,110,120,130,140,150,160,170,180]),
             "voltage": np.array([0.46,0.60,0.71,0.79,0.82,0.91,0.95,0.98,1.01,1.04,
                                  1.06,1.07,1.09,1.10,1.12,1.13,1.14,1.15,1.16])},
    "TF8":  {"load":    np.array([  1,10,20,30,40,50,60,70,80,90,
                                   100,110,120,130,140,150,160,170,180]),
             "voltage": np.array([0.45,0.55,0.65,0.70,0.77,0.83,0.88,0.92,0.97,1.012,
                                  1.047,1.078,1.108,1.134,1.159,1.182,1.202,1.222,1.241])}
}

# Base anchors: always include 10Ω and 150Ω
base_anchors = [10, 150]
all_loads   = data["TF7"]["load"]
candidates  = [r for r in all_loads if r not in base_anchors]

# Precompute ideal scaled Varmₘ for 10Ω and 150Ω
ideal_10   = ideal_varm_for_resistance(10)
ideal_150  = ideal_varm_for_resistance(150)

best_avg_lms = np.inf
best_combo   = None

# Search all 2-point combinations for the 3rd and 4th anchors
for r3, r4 in itertools.combinations(candidates, 2):
    # Compute ideal scaled targets for anchors
    ideal_3    = ideal_varm_for_resistance(r3)
    ideal_4    = ideal_varm_for_resistance(r4)
    y_targets  = np.array([ideal_10, ideal_3, ideal_4, ideal_150])

    lms_list = []
    # Evaluate each transformer
    for vals in data.values():
        loads   = vals["load"]
        volts   = vals["voltage"]
        # Extract raw voltages at the four anchors
        v10   = volts[np.where(loads==10)[0][0]]
        v3    = volts[np.where(loads==r3)[0][0]]
        v4    = volts[np.where(loads==r4)[0][0]]
        v150  = volts[np.where(loads==150)[0][0]]
        anchors_raw = np.array([v10, v3, v4, v150])

        # Fit cubic calibration (deg=3)
        coeffs = np.polyfit(anchors_raw, y_targets, deg=3)
        poly   = np.poly1d(coeffs)

        # Apply calibration
        varm_est = poly(volts)
        rpad_est = rpad_m(varm_est)
        err2     = (loads - rpad_est)**2
        lms_list.append(np.mean(err2))

    avg_lms = np.mean(lms_list)
    if avg_lms < best_avg_lms:
        best_avg_lms = avg_lms
        best_combo   = (r3, r4)

# Report the best combination
r3_best, r4_best = best_combo
print(f"Best 4-point calibration anchors: 10Ω, {r3_best}Ω, {r4_best}Ω, 150Ω")
print(f"Lowest average LMS error: {best_avg_lms:.2f} Ω²")

# Re-run final calibration with best anchors and plot
ideal_3   = ideal_varm_for_resistance(r3_best)
ideal_4   = ideal_varm_for_resistance(r4_best)
y_targs   = np.array([ideal_10, ideal_3, ideal_4, ideal_150])

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
lms_values = []

for ax, (tf, vals) in zip(axs, data.items()):
    loads = vals["load"]
    volts = vals["voltage"]
    # raw voltages at anchors
    v10  = volts[np.where(loads==10)[0][0]]
    v3   = volts[np.where(loads==r3_best)[0][0]]
    v4   = volts[np.where(loads==r4_best)[0][0]]
    v150 = volts[np.where(loads==150)[0][0]]
    anchors_raw = np.array([v10, v3, v4, v150])

    # fit & apply calibration
    coeffs = np.polyfit(anchors_raw, y_targs, deg=3)
    poly   = np.poly1d(coeffs)
    varm_est = poly(volts)
    rpad_est = rpad_m(varm_est)

    # compute LMS for this transformer
    errors  = loads - rpad_est
    lms_err = np.mean(errors**2)
    lms_values.append(lms_err)

    # plot measured vs estimated
    ax.plot(loads, loads,    'o-',  color='black', label='Measured')
    ax.plot(loads, rpad_est, 's--', color='purple', label='Estimated')
    ax.set_title(f"{tf} — LMS Error = {lms_err:.2f} Ω²")
    ax.set_ylabel("Resistance (Ω)")
    ax.grid(True)
    ax.legend()

# annotate average LMS and anchor info
avg_lms_final = np.mean(lms_values)
fig.text(0.5, 0.04,
         f"Anchors: 10Ω, {r3_best}Ω, {r4_best}Ω, 150Ω   |   "
         f"Average LMS = {avg_lms_final:.2f} Ω²",
         ha='center', fontsize=12)

axs[-1].set_xlabel("Load (Ω)")
plt.tight_layout(rect=[0,0.05,1,1])
plt.show()
