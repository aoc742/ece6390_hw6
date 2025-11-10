import numpy as np
import matplotlib.pyplot as plt

def pulse_amplitude_modulation(binary_data, pulse_width_ns, bit_duration_ns, amplitude=5.0, samples_per_ns=1):
    """
    Simulates a Unipolar PAM signal based on input binary data.

    The simulation uses a high sampling rate (1 sample/ns) to accurately model
    the specified nanosecond pulse width.
    
    Args:
        binary_data (np.array): Array of binary values (0s and 1s).
        pulse_width_ns (int/float): The width of the pulse in nanoseconds (e.g., 20).
        bit_duration_ns (int/float): The duration of one bit period in nanoseconds (must be >= pulse_width_ns).
        amplitude (float): The peak amplitude of the pulse for a '1'.
        samples_per_ns (int): The number of samples to take per nanosecond (1 is 1 GSa/s).

    Returns:
        tuple: (time_vector, pam_signal, bit_duration_s, Fs, N)
    """
    # 1. Define Parameters and Time Vector
    
    # Convert parameters to standard time unit (seconds)
    pulse_width_s = pulse_width_ns * 1e-9
    bit_duration_s = bit_duration_ns * 1e-9
    
    # Calculate sampling frequency (Fs) and period (Ts)
    Fs = samples_per_ns * 1e9 # Sampling frequency in Hz (Default: 1 GHz)
    Ts = 1 / Fs
    
    # Determine array indices based on time
    samples_per_bit = int(bit_duration_s / Ts)
    samples_per_pulse = int(pulse_width_s / Ts)
    
    # Validate parameters
    if samples_per_pulse <= 0:
        raise ValueError("Pulse width is too small or sampling rate is too low to model accurately.")
    if samples_per_pulse > samples_per_bit:
        raise ValueError("Pulse width cannot exceed the bit duration.")

    # Create the time vector
    total_time_s = len(binary_data) * bit_duration_s
    time_vector = np.arange(0, total_time_s, Ts)
    N = len(time_vector)

    print(f"Nof samples: {N}")
    
    # Initialize the PAM signal to zero
    pam_signal = np.zeros(N)
    
    # 2. Implement PAM modulation (Unipolar NRZ-like pulse)
    for i, bit in enumerate(binary_data):
        # Calculate start and end indices for the current bit period
        start_index = i * samples_per_bit
        # The pulse only lasts for pulse_width_s duration within the bit period
        end_pulse_index = start_index + samples_per_pulse
        
        # Amplitude modulation:
        if bit == 1:
            # Set the pulse amplitude for the specified duration
            pam_signal[start_index:end_pulse_index] = amplitude
        # For '0', the amplitude remains 0 (handled by initialization)
            
    return time_vector, pam_signal, bit_duration_s, Fs, N

# --- Simulation Setup ---

# The binary sequence to modulate
binary_data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])

# Required pulse width
pulse_width_ns = 20

# Chosen bit duration (1 microsecond) to make the short pulse visible
bit_duration_ns = 1000 
amplitude = 5.0 # Peak voltage for '1'

# PAM
time_vector, pam_signal, bit_duration_s, Fs, N = pulse_amplitude_modulation(
    binary_data, 
    pulse_width_ns, 
    bit_duration_ns,
    amplitude=amplitude
)


# FFT / Plot Tx Results


# 1. Compute FFT
Y = np.fft.fft(pam_signal)

# 2. Calculate Single-Sided Spectrum (Magnitude)
# P2 is the two-sided power spectrum magnitude (normalized by N)
P2 = np.abs(Y / N)
# P1 is the single-sided spectrum (only positive frequencies)
P1 = P2[:N//2]
# Multiply by 2 because power is concentrated in the positive frequencies 
# (except for DC and Nyquist which should not be doubled)
P1[1:] = 2 * P1[1:] 

# 3. Calculate Frequency Vector
f = Fs * np.arange(N//2) / N # Frequency in Hz
f_ghz = f / 1e9 # Convert frequency to GHz for plotting

# 4. Calculate Power Spectral Density (PSD)
# PSD [W/Hz] = (|Y|^2) / (Fs * N)
Pxx = (np.abs(Y)**2) / (Fs * N)
Pxx_db_single = 10 * np.log10(Pxx[:N//2]) # Single-sided PSD in dB/Hz

# --- Plotting ---

# Convert time vector to microseconds (μs) for clear plotting labels
time_vector_us = time_vector * 1e6
total_time_us = time_vector_us[-1]

plt.figure(figsize=(16, 12))

# --- Subplot 1: Time Domain Signal ---
plt.subplot(3, 1, 1)
plt.plot(time_vector_us, pam_signal, drawstyle='steps-post', 
            color='#059669', linewidth=2.5) 

plt.title(f'Time Domain PAM Signal (T_pulse={pulse_width_ns} ns, T_bit={bit_duration_ns} ns)', 
            fontsize=16, fontweight='bold', color='#1F2937')
plt.xlabel('Time ($\mu$s)', fontsize=12)
plt.ylabel('Amplitude (Volts)', fontsize=12)

# Highlight bit periods and annotate bits
num_bits = len(binary_data)
for i in range(num_bits):
    bit_boundary = (i + 1) * bit_duration_s * 1e6
    plt.axvline(x=bit_boundary, color='#374151', linestyle='--', alpha=0.3)
    mid_bit = (i * bit_duration_s + bit_duration_s/2) * 1e6
    plt.text(mid_bit, amplitude + 0.5, str(binary_data[i]), 
                ha='center', fontsize=10, fontweight='bold', color='#DC2626')

plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.ylim(-0.5, amplitude + 1.5)
plt.xlim(0, total_time_us)

# --- Subplot 2: Magnitude Spectrum (FFT) ---
plt.subplot(3, 1, 2)
# Find the frequency where the first zero crossing occurs (1/T_pulse)
first_null_freq = 1 / (pulse_width_ns * 1e-9) / 1e9

plt.plot(f_ghz, P1, color='#1D4ED8', linewidth=2)

plt.title('Magnitude Spectrum (FFT) of the PAM Signal', 
            fontsize=16, fontweight='bold', color='#1F2937')
plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel('|Y(f)|', fontsize=12)

# Highlight the first null
plt.axvline(x=first_null_freq, color='#DC2626', linestyle='--', alpha=0.7, label=f'First Null (1/{pulse_width_ns}ns = {first_null_freq:.2f} GHz)')

plt.legend()
plt.grid(True, which='major', linestyle=':', alpha=0.6)
plt.xlim(0, 50 / pulse_width_ns) # Limit x-axis to clearly show the first few lobes

# --- Subplot 3: Power Spectral Density (PSD) ---
plt.subplot(3, 1, 3)

plt.plot(f_ghz, Pxx_db_single, color='#9333EA', linewidth=2)

plt.title('Power Spectral Density (PSD) in dB/Hz', 
            fontsize=16, fontweight='bold', color='#1F2937')
plt.xlabel('Frequency (GHz)', fontsize=12)
plt.ylabel('PSD (dB/Hz)', fontsize=12)

plt.axvline(x=first_null_freq, color='#DC2626', linestyle='--', alpha=0.7)

plt.grid(True, which='major', linestyle=':', alpha=0.6)
plt.xlim(0, 50 / pulse_width_ns)
plt.ylim(np.max(Pxx_db_single) - 80, np.max(Pxx_db_single) + 5)

plt.tight_layout(pad=3.0)
plt.show()
