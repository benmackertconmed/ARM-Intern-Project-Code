import numpy as np
import matplotlib.pyplot as plt
import serial
import time

SERIAL_PORT = 'COM5'  # Change this to your Arduino port
BAUD_RATE = 9600
NUM_STEPS = 19

def fit_resistance_vs_voltage(resistances, voltages, degree=4):
    return np.polyfit(voltages, resistances, degree)

def estimate_resistance(voltage, coeffs):
    return np.polyval(coeffs, voltage)

def plot_fit(voltage_samples, resistance_samples, coeffs):
    V_range = np.linspace(min(voltage_samples)-0.1, max(voltage_samples)+0.1, 300)
    R_fit = estimate_resistance(V_range, coeffs)

    plt.plot(voltage_samples, resistance_samples, 'o', label="Calibration Data")
    plt.plot(V_range, R_fit, '-', label="Fitted Curve")
    plt.xlabel("Measured Voltage (V)")
    plt.ylabel("Resistance (Ω)")
    plt.title("Resistance Estimation from Voltage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def read_voltages_from_arduino():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)

    voltages = []
    print("Reading voltages from Arduino...")

    while len(voltages) < NUM_STEPS:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                voltage = float(line)
                voltages.append(voltage)
                print(f"Step {len(voltages)-1}: {voltage:.3f} V")
            except ValueError:
                print(f"Ignored: {line}")

    ser.close()
    return np.array(voltages)

def main():
    # Voltages from Arduino (steps 0–18)
    voltage_samples = read_voltages_from_arduino()
    resistance_samples = np.array([
        37, 47, 57, 66, 76, 86, 95, 105, 115, 125,
        134, 144, 153, 163, 173, 182, 192, 202, 212
    ])

    # Manually measured low-resistance calibration points
    manual_voltages = np.array([0.437, 0.545, 0.634, 0.707])
    manual_resistances = np.array([1, 10, 20, 30])

    # Combine manual + Arduino data
    voltage_samples = np.concatenate((manual_voltages, voltage_samples))
    resistance_samples = np.concatenate((manual_resistances, resistance_samples))

    # Fit and plot
    coeffs = fit_resistance_vs_voltage(resistance_samples, voltage_samples, degree=4)
    plot_fit(voltage_samples, resistance_samples, coeffs)

if __name__ == "__main__":
    main()
