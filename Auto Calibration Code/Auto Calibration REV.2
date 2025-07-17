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

def read_calibration_voltages(ser):
    voltages = []
    print("Reading calibration voltages from Arduino...")

    while len(voltages) < NUM_STEPS:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                voltage = float(line)
                voltages.append(voltage)
                print(f"Step {len(voltages)-1}: {voltage:.3f} V")
            except ValueError:
                print(f"Ignored: {line}")
    return np.array(voltages)

def read_reference_voltage(prompt, ser):
    input(f"\n🔧 {prompt} Then press Enter to continue...")

    # Flush the serial buffer to discard old readings
    ser.reset_input_buffer()
    time.sleep(0.5)  # Give Arduino time to send a fresh value

    print("  Waiting for fresh voltage reading...")

    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                voltage = float(line)
                print(f"  → Measured Voltage: {voltage:.3f} V")
                return voltage
            except ValueError:
                print(f"Ignored: {line}")


def live_voltage_stream(coeffs):
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("\n--- Live Resistance Estimation ---")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    voltage = float(line)
                    resistance = estimate_resistance(voltage, coeffs)
                    print(f"Voltage: {voltage:.3f} V → Estimated Resistance: {resistance:.2f} Ω", end='')

                    if resistance < 10:
                        print("  🟢 Single foil pad detected")
                    elif resistance > 150:
                        print("  🔴 Pad Fault")
                    else:
                        print()
                except ValueError:
                    print(f"Ignored: {line}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

def main():
    # Open serial once and share it
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)

    # Digipot calibration data
    digipot_resistances = np.array([
        37, 47, 57, 66, 76, 86, 95, 105, 115, 125,
        134, 144, 153, 163, 173, 182, 192, 202, 212
    ])
    digipot_voltages = read_calibration_voltages(ser)

    # Manual low-end resistor data
    manual_resistances = np.array([1, 20, 30])
    manual_voltages = np.array([0.437, 0.634, 0.707])

    # Prompt user for reference resistors
    ref_10v = read_reference_voltage("Insert a 10Ω resistor.", ser)
    ref_150v = read_reference_voltage("Insert a 150Ω resistor.", ser)

    ser.close()

    ref_resistances = np.array([10, 150])
    ref_voltages = np.array([ref_10v, ref_150v])

    # Combine all data
    all_resistances = np.concatenate((manual_resistances, digipot_resistances, ref_resistances))
    all_voltages = np.concatenate((manual_voltages, digipot_voltages, ref_voltages))

    # Fit and plot
    coeffs = fit_resistance_vs_voltage(all_resistances, all_voltages, degree=4)
    plot_fit(all_voltages, all_resistances, coeffs)

    # Start live estimation
    live_voltage_stream(coeffs)

if __name__ == "__main__":
    main()
