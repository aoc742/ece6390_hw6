import numpy as np
from matplotlib import pyplot as plt

# Tx Pulse amplitude modulation
def pam_mod():
    return

# Tx Amplitude modulation
def am_mod():
    return

# Rx Amplitude demodulation
def am_demod():
    return

# Rx Matched filter
def matched_filter():
    return

if __name__ == "__main__":
    # Bit rate Rb = 1 / Tb
    # Shape of pulse p(t)
    # Max 4Tb time support (Tb = bit duration for a single bit)

    # Tx AM must be 2.4-2.5 GHz with 50dB less than peak outside band
    # 20*log10(|Pm(f)|/|Pm(fpeak)|) < 50 dB

    # Goal: Find the best possible pulse shape for this system and attach plots
    # of the pulse spectrum that demonstrate the validity of your pulse design.

    t1 = -10 # start time
    t2 = 10 # end time
    N = 2**8 # nof samples
    t = np.linspace(t1, t2, N)
    fs = N/(t2-t1)
    x = np.sinc(0.5-np.abs(t))
    # X = np.fft.fft(x)
    X = np.fft.fftshift(np.fft.fft(x))
    XX = np.abs(X)*(t2-t1)/N

    PSD = np.abs(np.fft.fft(x))**2 / (N*fs)
    PSD_log = 10.0*np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)
    
    center_freq = 2.45e9 # 2.45 GHz
    f = np.arange(fs/-2.0, fs/2.0, fs/N) # start, stop, step size centered about 0 Hz
    f += center_freq
    plt.plot(f, PSD_shifted)
    plt.show()
    

    plt.plot(t, x, '.-')
    plt.show()

    plt.plot(t, XX, '.-')
    plt.show()


