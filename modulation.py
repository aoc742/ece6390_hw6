import numpy as np
from matplotlib import pyplot as plt
import random
from scipy import signal

# https://www.sciencedirect.com/topics/computer-science/pulse-amplitude-modulation
# bandwidth is inversely proportional to pulse duration
# 1/50e6 (b/c +/-50MHz is our FCC limitation) -> Pulse T = 2e-8, or 20 nanosecond pulse width


if __name__ == "__main__":
    t1 = 0 # start time
    t2 = 1 # end time
    duration = t2 - t1
    N = 2**8 # nof samples
    t = np.linspace(t1, t2, N)
    f_axis = np.arange(-N/2, N/2) / (t2-t1) # frequency range
    sample_rate = N/(t2-t1) # sample rate

    # Your message data to transmit
    mt = [random.randint(0, 1) for x in t]
    mt = np.array(mt)
    print(type(mt))
    print(len(mt))

    # Pulse width 20 nanoseconds corresponds to 1/20-8, or 50MHz
    # pw = 2e-8
    # pri = 4e-8 # No dwell time between pulses

    pulse_frequency = 50e6
    pulse_width_ratio = 0.5 # Duty cycle of the square wave (e.g., 0.5 for 50%)
    pulse_train = signal.square(2 * np.pi * pulse_frequency * t, pulse_width_ratio)


    # Map message bits to PAM symbols
    # 0 -> -1
    # 1 -> 1
    for bit in mt:
        if bit == 0:
            bit = -1

    # Carrier wave
    carrier = np.cos(2 * np.pi * 50 * t)
    pam_signal = carrier * pulse_train


    # Amplitude Modulation changes the amplitude of the carrier
    # inputs for AM should be an unmodulated RF carrier (sine wave)
    #   and a low frequency modulating signal.
    # The envelope of the result will be a copy of the modulating signal
    carrier_amp = 1
    carrier_freq = 2.45e9
    carrier_phase = 0
    modulation_index = 0.8 # ratio of carrier to sideband amplitude (aka carrier envelope)
    am_signal = pam_signal * modulation_index * carrier_amp * np.sin(2*np.pi* carrier_freq * t + carrier_phase)
    f_axis = f_axis + carrier_freq
    # Ex: if your modulating signal is a 1000Hz tone, then in frequency domain
    # it will appear 1000Hz above and below the carrier frequency
    # So your overall bandwidth is always 2 x the highest modulated frequency 
    # (in this case, 2000Hz)

    plt.subplot(3, 1, 1)
    plt.plot(t, mt)
    plt.title("Data")

    plt.subplot(3, 1, 2)
    plt.plot(t, pam_signal)
    plt.title("PAM (time)")

    plt.subplot(3, 1, 3)
    plt.plot(t, am_signal)
    plt.title("AM (time)")
    plt.tight_layout()
    plt.show()

    # Zero pad the time domain signal before FFT for finer resolution
    padded_am_signal = np.pad(am_signal, int(2**np.ceil(np.log2(len(am_signal)))))

    ## FFT of modulated signal
    X = np.fft.fftshift(np.fft.fft(padded_am_signal))
    XX = np.abs(X)*(t2-t1)/N

    plt.plot(X)
    plt.title("Frequency")
    plt.show()

    # PSD = np.abs(np.fft.fft(am_signal))**2 / (N*sample_rate)
    # PSD_log = 10.0*np.log10(PSD)
    # PSD_shifted = np.fft.fftshift(PSD_log)   


    plt.psd(X, 512, 1/0.01)
    plt.title("PSD")
    plt.show()