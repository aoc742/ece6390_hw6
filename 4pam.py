import numpy as np
import matplotlib.pyplot as plt

# Parameters
Fs = 10000  # Sampling frequency
T = 1 / Fs  # Sampling period
duration = 0.1  # Signal duration in seconds
t = np.arange(0, duration, T)  # Time vector

# Message signal (e.g., a sine wave)
fm = 10  # Message signal frequency
message_signal = np.sin(2 * np.pi * fm * t)

# PAM parameters
num_levels = 4  # Number of amplitude levels (e.g., 2-ASK, 4-ASK)
max_amplitude = np.max(np.abs(message_signal))
quantization_levels = np.linspace(-max_amplitude, max_amplitude, num_levels)

# Sampling and Quantization
# For simplicity, we'll sample at a lower rate for symbol generation
symbol_rate = 100 # Symbols per second
samples_per_symbol = int(Fs / symbol_rate)

# Sample the message signal
sampled_message = message_signal[::samples_per_symbol]

# Quantize the sampled values to the nearest PAM level
quantized_symbols = np.zeros_like(sampled_message)
for i, sample in enumerate(sampled_message):
    idx = np.argmin(np.abs(quantization_levels - sample))
    quantized_symbols[i] = quantization_levels[idx]

# Generate PAM signal
pam_signal = np.zeros_like(t)
pulse_duration = samples_per_symbol * T # Duration of each pulse

for i, symbol in enumerate(quantized_symbols):
    start_index = i * samples_per_symbol
    end_index = start_index + samples_per_symbol
    pam_signal[start_index:end_index] = symbol

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, message_signal)
plt.title('Original Message Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(t[::samples_per_symbol], quantized_symbols, linefmt='r-', markerfmt='ro', basefmt=' ')
plt.title(f'Sampled and Quantized Symbols ({num_levels}-PAM)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude Level')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, pam_signal)
plt.title(f'Generated {num_levels}-PAM Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()