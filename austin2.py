import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import random

def pulse_amplitude_modulation(data):
    pulse_width = 400 * 1e-9 # Pulse width 20 ns
    Tb = 1000 * 1e-9  # Bit duration 1us
    bitrate = 1/Tb

    print(f"bitrate: {int(bitrate)} bps")

    Fs = 1e9 # 100MHz sample rate
    Ts = 1 / Fs # Ts = period (time between each sample)
    
    samples_per_bit = int(Tb / Ts)
    samples_per_pulse = int(pulse_width / Ts)
    print(f"Samples per bit: {samples_per_bit}. Samples per pulse: {samples_per_pulse}")

    duration = len(data) * Tb
    t = np.arange(-duration/2, duration/2, Ts) # nof_samples
    
    pam_signal = np.zeros(len(t))
    for i, bit in enumerate(data):
        start = i * samples_per_bit
        end = start + samples_per_pulse
        pam_signal[start:end] = 5 if bit == 1 else 0
            
    return t, pam_signal, Fs, duration

# The binary sequence to modulate
# data = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1])
data = np.random.randint(2, size=5)

# PAM
t, pam_signal, Fs, duration = pulse_amplitude_modulation(data)

plt.subplot(3, 2, 1)
plt.plot(t, pam_signal) 
plt.title("PAM (time)")

# AM upconvert to 2.45 GHz center frequency
carrier_amp = 1
carrier_freq = 2.45e9
carrier_phase = 0
modulation_index = 0.8 # ratio of carrier to sideband amplitude (aka carrier envelope)
am_signal = (1 + modulation_index * pam_signal) * carrier_amp * np.sin(2*np.pi* carrier_freq * t + carrier_phase)

plt.subplot(3, 2, 2)
plt.plot(t, am_signal)
plt.title("AM modulation")

output_signal = pam_signal

# Zero pad before taking FFT
next_power_of_2 = int(np.ceil(np.log2(len(output_signal)))) + 2
target_length = 2**next_power_of_2
deficit = target_length - len(output_signal)    
padded_output_signal = np.pad(output_signal, (0, deficit), mode='constant', constant_values=0)
print(f"Length of padded output: {len(padded_output_signal)}")

# FFT 
output_fft = np.fft.fft(padded_output_signal)


f_axis = np.arange(-Fs/2, Fs/2, Fs/len(padded_output_signal))

output_fft_scaled = abs(np.fft.fftshift(output_fft))*(duration)/len(output_fft)

plt.subplot(3, 2, 3)
plt.plot(f_axis, output_fft_scaled)
plt.title("Shifted FFT")

# PSD
psd = 10*np.log10(output_fft_scaled)
# # psd = abs(output_fft)**2 + 0.25*abs(output_fft[0])**2
plt.subplot(3, 2, 4)
# plt.psd(output_fft_scaled, Fs=Fs, )
plt.plot(f_axis, psd)
plt.title("PSD")


plt.tight_layout()
plt.show()
