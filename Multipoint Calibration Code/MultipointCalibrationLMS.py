import numpy as np
import matplotlib.pyplot as plt

# 4th-order system polynomial: Rpad_m = f(Varm_m)
def rpad_m(varm):
    return (
        715.13226 * varm**4
      -1620.07948 * varm**3
      +1536.71187 * varm**2
      - 534.80120 * varm
      +  56.73790
    )

# Invert polynomial: find ideal Varm_m for a given target resistance
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
    mid = (vmin + vmax) / 2
    return min(real_roots, key=lambda r: abs(r - mid))

# Transformer measurement data
data = {
    "TF7":  {"load": np.array([1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]),
             "voltage": np.array([0.46,0.58,0.67,0.76,0.81,0.89,0.94,0.99,1.03,1.07,
                                  1.10,1.14,1.17,1.19,1.22,1.24,1.26,1.28,1.30])},
    "TF10": {"load": np.array([1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]),
             "voltage": np.array([0.46,0.60,0.71,0.79,0.82,0.91,0.95,0.98,1.01,1.04,
                                  1.06,1.07,1.09,1.10,1.12,1.13,1.14,1.15,1.16])},
    "TF8":  {"load": np.array([1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]),
             "voltage": np.array([0.45,0.55,0.65,0.70,0.77,0.83,0.88,0.92,0.97,1.012,
                                  1.047,1.078,1.108,1.134,1.159,1.182,1.202,1.222,1.241])}
}

# Prompt for calibration anchors
n_pts = int(input("How many calibration points? "))
cal_res = []
for i in range(n_pts):
    R = int(input(f"  Anchor #{i+1} — resistance (Ω): "))
    if R not in data["TF7"]["load"]:
        raise ValueError(f"{R} Ω not in test loads")
    cal_res.append(R)
cal_res = np.array(cal_res)

# Compute target Varm_m values by inverting system polynomial
y_targets = np.array([ideal_varm_for_resistance(R) for R in cal_res])

# Print anchor summary
print("\nAnchors and ideal scaled Varmₘ:")
for R, y in zip(cal_res, y_targets):
    print(f"  {R} Ω → {y:.4f} V")

# Create figure
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
lms_list = []

for ax, (tf, vals) in zip(axs, data.items()):
    loads   = vals["load"]
    raw_vol = vals["voltage"]

    v_anchors = np.array([raw_vol[np.where(loads == R)[0][0]] for R in cal_res])
    coeffs    = np.polyfit(v_anchors, y_targets, deg=n_pts-1)
    poly      = np.poly1d(coeffs)

    varm_m_est = poly(raw_vol)
    rpad_est   = rpad_m(varm_m_est)

    errors  = loads - rpad_est
    lms_err = np.mean(errors**2)
    lms_list.append(lms_err)

    ax.plot(loads, loads,    'o-', color='black', label='Measured')
    ax.plot(loads, rpad_est, 's--', color='purple', label='Estimated')
    ax.set_title(f"{tf} — LMS Error = {lms_err:.2f} Ω²")
    ax.set_ylabel("Resistance (Ω)")
    ax.grid(True)
    ax.legend()

# Top annotation: number of anchors and values
plt.subplots_adjust(top=0.90)
anchor_text_top = f"Calibration Anchors Used: {len(cal_res)} points — {', '.join(str(R) + ' Ω' for R in cal_res)}"
fig.text(0.5, 0.92, anchor_text_top, ha='center', fontsize=14)

# Bottom annotation: anchor list and average LMS
average_lms = np.mean(lms_list)
anchor_text_bottom = f"Anchors: {', '.join(str(R) + ' Ω' for R in cal_res)}    |    Average LMS Error: {average_lms:.2f} Ω²"
fig.text(0.5, 0.04, anchor_text_bottom, ha='center', fontsize=12)

axs[-1].set_xlabel("Load (Ω)")
plt.tight_layout(rect=[0, 0.06, 1, 0.88])
plt.show()
