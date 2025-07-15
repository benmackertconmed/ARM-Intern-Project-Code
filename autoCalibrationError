import numpy as np
import matplotlib.pyplot as plt

# Transformer data
data = {
    "TF10": {
        "measured": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]),
        "calculated": np.array([10, 24, 33, 40, 50, 59, 68, 79, 91, 98, 108, 120, 130, 140, 150, 161, 171, 180])
    },
    "TF7": {
        "measured": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]),
        "calculated": np.array([10, 20, 30, 38, 48, 58, 67, 77, 86, 97, 107, 117, 127, 139, 148, 158, 168, 178])
    },
    "TF8": {
        "measured": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]),
        "calculated": np.array([11, 20, 30, 39, 48, 57, 67, 77, 88, 97, 108, 118, 128, 139, 149, 158, 169, 179])
    }
}

# Plot measured vs. calculated for each transformer
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for i, (tf, values) in enumerate(data.items()):
    measured = values["measured"]
    calculated = values["calculated"]
    errors = measured - calculated
    squared_errors = errors ** 2
    lms_error = np.mean(squared_errors)

    axs[i].plot(measured, marker='o', linestyle='-', color='black', label='Measured')
    axs[i].plot(calculated, marker='s', linestyle='--', color='orange', label='Calculated')
    axs[i].set_title(f"{tf}: Measured vs Calculated Resistance\nLMS Error = {lms_error:.2f} Ω²")
    axs[i].set_ylabel("Resistance (Ω)")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Sample Index")
plt.tight_layout()
plt.show()
