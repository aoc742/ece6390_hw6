import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
sampling_rate = 1000  # samples per second
duration = 2          # seconds
time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Pulse train parameters
pulse_frequency = 5    # Hz (frequency of the pulse train)
pulse_width_ratio = 0.5 # Duty cycle of the square wave (e.g., 0.5 for 50%)

# Carrier wave parameters
carrier_frequency = 50 # Hz (frequency of the carrier signal)

# 1. Generate the pulse train (square wave)
pulse_train = signal.square(2 * np.pi * pulse_frequency * time, pulse_width_ratio)

# 2. Generate the carrier wave (sine wave)
carrier_wave = np.sin(2 * np.pi * carrier_frequency * time)

# 3. Modulate the carrier with the pulse train
# This example uses amplitude modulation (multiplying the carrier by the pulse train)
# Other modulation schemes are possible depending on the application.
modulated_signal = carrier_wave * pulse_train

# Plotting the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, pulse_train)
plt.title('Pulse Train (Modulating Signal)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, carrier_wave)
plt.title('Carrier Wave')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, modulated_signal)
plt.title('Pulse Train Carrier (Modulated Signal)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()