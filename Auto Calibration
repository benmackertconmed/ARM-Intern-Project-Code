import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time
import datetime
import serial
import numpy as np

# === CONFIGURATION ===
SERIAL_PORT = 'COM5'
BAUD_RATE = 9600
NUM_STEPS = 19

# === GLOBAL STATE ===
coeffs = None
last_calibration_time = None
resistance_value = 0.0
stop_live = False
pause_live_monitor = False
shared_serial = None
lock_in_value = None
pad_type = None  # 'single', 'dual', or None
flash = False

# === GUI SETUP ===
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Pad Resistance Monitor")
app.geometry("1200x700")
app.resizable(False, False)

# === IMAGE CROPPING FUNCTION ===
def manual_crop_image(image_path, size):
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    left = 10
    top = 10
    right = width - 10
    bottom = height - 10
    img = img.crop((left, top, right, bottom))
    return ImageTk.PhotoImage(img.resize(size, Image.LANCZOS))

# === LOAD PAD IMAGES ===
canvas_width = 400
canvas_height = 525
pad_single_green_img = manual_crop_image("pad_single_green.png", (canvas_width, canvas_height))
pad_single_red_img = manual_crop_image("pad_single_red.png", (canvas_width, canvas_height))
pad_dual_green_img = manual_crop_image("pad_dual_green.png", (canvas_width, canvas_height))
pad_dual_red_img = manual_crop_image("pad_dual_red.png", (canvas_width, canvas_height))
pad_dual_peel_img = manual_crop_image("pad_dual_peel.png", (canvas_width, canvas_height))

# === HEADER ===
header = ctk.CTkLabel(app, text="Pad Resistance Monitor", font=("Arial", 28, "bold"))
header.pack(pady=10)

# === LEFT PANEL ===
left_frame = ctk.CTkFrame(app, width=300, corner_radius=10)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

calibrate_button = ctk.CTkButton(left_frame, text="Start Calibration", corner_radius=10)
calibrate_button.pack(pady=10)

lockin_button = ctk.CTkButton(left_frame, text="Lock In Pad", corner_radius=10)
lockin_button.pack(pady=10)

last_cal_label = ctk.CTkLabel(left_frame, text="Last Calibration:\n—", justify="left")
last_cal_label.pack(pady=10)

timer_label = ctk.CTkLabel(left_frame, text="", font=("Arial", 14))
timer_label.pack(pady=10)

progress_bar = ctk.CTkProgressBar(left_frame, width=200)
progress_bar.set(0)
progress_bar.pack(pady=10)

log_box = ctk.CTkTextbox(left_frame, width=230, height=300, font=("Courier", 10), corner_radius=10)
log_box.pack(pady=10)
log_box.insert("end", "Calibration Log:\n")
log_box.configure(state="disabled")

clock_label = ctk.CTkLabel(left_frame, text="Time: --:--:--", font=("Arial", 12))
clock_label.pack(side="bottom", pady=10)

# === RIGHT PANEL ===
right_frame = ctk.CTkFrame(app, corner_radius=10, fg_color="white")
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

resistance_label = ctk.CTkLabel(right_frame, text="Resistance: — Ω", font=("Arial", 24, "bold"))
resistance_label.pack(pady=20)

canvas = tk.Canvas(right_frame, width=canvas_width, height=canvas_height, bg="white", highlightthickness=0)
canvas.pack()
canvas_image = canvas.create_image(0, 0, anchor="nw", image=pad_dual_green_img)

pad_status_label = ctk.CTkLabel(right_frame, text="Pad: —", font=("Arial", 16))
pad_status_label.pack(pady=10)

# === FUNCTIONS ===
def read_voltage(ser):
    if not ser or not ser.is_open:
        return 0.0
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                return float(line)
        except Exception:
            continue

def read_reference_voltage(prompt, ser, label):
    messagebox.showinfo("Insert Resistor", prompt)
    ser.reset_input_buffer()
    time.sleep(0.5)
    v = read_voltage(ser)
    log_box.configure(state="normal")
    log_box.insert("end", f"{label}: {v:.3f} V\n")
    log_box.see("end")
    log_box.configure(state="disabled")
    return v

def fit_resistance_vs_voltage(resistances, voltages, degree=4):
    return np.polyfit(voltages, resistances, degree)

def estimate_resistance(voltage, coeffs):
    return np.polyval(coeffs, voltage)

def run_calibration():
    global coeffs, last_calibration_time, shared_serial, pause_live_monitor
    calibrate_button.configure(state="disabled")
    timer_label.configure(text="Starting calibration...")
    progress_bar.set(0)
    pause_live_monitor = True

    if shared_serial and shared_serial.is_open:
        shared_serial.close()
        time.sleep(1)

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    ser.reset_input_buffer()
    time.sleep(2)

    digipot_resistances = np.array([
        37, 47, 57, 66, 76, 86, 95, 105, 115, 125,
        134, 144, 153, 163, 173, 182, 192, 202, 212
    ])
    digipot_voltages = []

    log_box.configure(state="normal")
    log_box.insert("end", "Reading sweep voltages...\n")
    log_box.configure(state="disabled")

    for i in range(NUM_STEPS):
        v = read_voltage(ser)
        digipot_voltages.append(v)
        progress_bar.set((i + 1) / NUM_STEPS)
        timer_label.configure(text=f"Step {i}/{NUM_STEPS - 1}")
        log_box.configure(state="normal")
        log_box.insert("end", f"Step {i:2d}: {v:.3f} V\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    ref_10v = read_reference_voltage("Insert 10Ω resistor and press OK", ser, "10Ω Ref")
    ref_150v = read_reference_voltage("Insert 150Ω resistor and press OK", ser, "150Ω Ref")

    ser.close()

    # Add manual calibration points
    manual_resistances = np.array([1, 10, 20, 30]) 
    
    # manual_voltages = np.array([0.46, 0.58, 0.67, 0.76])  # TF7 39K
    # manual_voltages = np.array([0.45, 0.55, 0.65, 0.7])  # TF8 39K
    manual_voltages = np.array([0.46, 0.6, 0.71, 0.79]) #TF10 39k
    # manual_voltages = np.array([0.445, 0.598, 0.715, 0.799]) #TF10 36k

    all_resistances = np.concatenate((digipot_resistances, [10, 150], manual_resistances))
    all_voltages = np.concatenate((digipot_voltages, [ref_10v, ref_150v], manual_voltages))
    coeffs = fit_resistance_vs_voltage(all_resistances, all_voltages)

    last_calibration_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_cal_label.configure(text=f"Last Calibration:\n{last_calibration_time}")
    timer_label.configure(text="Calibration complete")
    progress_bar.set(1)
    calibrate_button.configure(state="normal")

    shared_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    shared_serial.reset_input_buffer()
    time.sleep(2)
    pause_live_monitor = False

def lock_in_pad():
    global lock_in_value, pad_type
    lock_in_value = resistance_value
    if lock_in_value < 8:
        pad_type = 'single'
    elif 12 < lock_in_value <= 150:
        pad_type = 'dual'
    else:
        pad_type = None

def live_monitor():
    global resistance_value, stop_live, shared_serial, pause_live_monitor, flash
    global fault_point
    clear_point = None
    pad_fault_active = False
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    shared_serial = ser
    ser.reset_input_buffer()
    time.sleep(2)
    fault_point = None  # SRS-580

    while not stop_live:
        if coeffs is None or pause_live_monitor or not shared_serial or not shared_serial.is_open:
            time.sleep(0.1)
            continue
        voltage = read_voltage(shared_serial)
        resistance_value = estimate_resistance(voltage, coeffs)
        resistance_label.configure(text=f"Resistance: {resistance_value:.2f} Ω")
        pad_text = "Pad: Unknown"
        if lock_in_value is None:
            if resistance_value < 8:
                canvas.itemconfig(canvas_image, image=pad_single_green_img)
                pad_text = "Pad: Single (Pre-Lock)"
            elif resistance_value < 130:
                canvas.itemconfig(canvas_image, image=pad_dual_green_img)
                pad_text = "Pad: Dual (Pre-Lock Valid)"
            elif resistance_value < 154:
                canvas.itemconfig(canvas_image, image=pad_dual_peel_img)
                pad_text = "Pad: Dual (Pre-Lock Peel)"
            else:
                canvas.itemconfig(canvas_image, image=pad_dual_red_img if flash else "")
                pad_text = "Pad: Pre-Lock Fault"
                flash = not flash
            fault_point = None
            clear_point = None
            pad_fault_active = False
        else:
            if pad_type == 'single':
                if resistance_value < 8:
                    canvas.itemconfig(canvas_image, image=pad_single_green_img)
                    pad_text = "Pad: Single (Valid)"
                else:
                    canvas.itemconfig(canvas_image, image=pad_single_red_img if flash else "")
                    pad_text = "Pad: Single (Fault)"
                    flash = not flash
            elif pad_type == 'dual':
                # --- SRS-580 Fault Point Logic ---
                fault_130 = lock_in_value * 1.3
                fault_10 = lock_in_value + 10
                if fault_130 > fault_10:
                    fault_point = fault_130
                else:
                    fault_point = fault_10
                if fault_point is not None and fault_point > 154:
                    fault_point = 154

                # --- Clear Point Calculation ---
                # Option 1: Initial Value × 1.30
                cp1 = lock_in_value * 1.3
                # Option 2: Nominal Fault Point × 0.80
                cp2 = fault_point * 0.8 if fault_point is not None else float('inf')
                # Option 3: 146
                cp3 = 146
                # Option 4: Nominal Fault Point − 5
                cp4 = fault_point - 5 if fault_point is not None else float('inf')
                clear_point = min(cp1, cp2, cp3, cp4)

                # Pad Fault State Machine
                if not pad_fault_active:
                    if fault_point is not None and resistance_value > fault_point:
                        canvas.itemconfig(canvas_image, image=pad_dual_red_img if flash else "")
                        pad_text = f"Pad: Fault > {fault_point:.1f}Ω"
                        flash = not flash
                        pad_fault_active = True
                    elif fault_point is not None and resistance_value > fault_point - 7:
                        canvas.itemconfig(canvas_image, image=pad_dual_peel_img)
                        pad_text = "Pad: Dual (Peel Alert)"
                    else:
                        canvas.itemconfig(canvas_image, image=pad_dual_green_img)
                        pad_text = "Pad: Dual (Valid)"
                else:
                    # Fault is active, require resistance to drop below clear_point to clear fault
                    if resistance_value <= clear_point:
                        pad_fault_active = False
                        # After clearing, immediately update pad state as normal
                        if fault_point is not None and resistance_value > fault_point - 7:
                            canvas.itemconfig(canvas_image, image=pad_dual_peel_img)
                            pad_text = "Pad: Dual (Peel Alert)"
                        else:
                            canvas.itemconfig(canvas_image, image=pad_dual_green_img)
                            pad_text = "Pad: Dual (Valid)"
                    else:
                        canvas.itemconfig(canvas_image, image=pad_dual_red_img if flash else "")
                        pad_text = f"Pad: Fault > {fault_point:.1f}Ω (Clear ≤ {clear_point:.1f}Ω)"
                        flash = not flash
        pad_status_label.configure(text=pad_text)
        time.sleep(0.2)
    ser.close()

def update_clock():
    if not stop_live:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        clock_label.configure(text=f"Time: {now}")
        app.after(1000, update_clock)

# === EVENT BINDINGS ===
calibrate_button.configure(command=lambda: threading.Thread(target=run_calibration, daemon=True).start())
lockin_button.configure(command=lock_in_pad)
threading.Thread(target=live_monitor, daemon=True).start()
update_clock()
app.mainloop()
stop_live = True
update_clock()
app.mainloop()
stop_live = True
now = datetime.datetime.now().strftime("%H:%M:%S")
clock_label.configure(text=f"Time: {now}")
app.after(1000, update_clock)

# === EVENT BINDINGS ===
calibrate_button.configure(command=lambda: threading.Thread(target=run_calibration, daemon=True).start())
lockin_button.configure(command=lock_in_pad)
threading.Thread(target=live_monitor, daemon=True).start()
update_clock()
app.mainloop()
stop_live = True
