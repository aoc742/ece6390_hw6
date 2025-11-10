import numpy as np
from matplotlib import pyplot as plt

# Tx Pulse amplitude modulation
def pam_mod():
    return

# Tx Amplitude modulation
def am_mod():
    return

if __name__ == "__main__":
    # Bit rate Rb = 1 / Tb
    # Shape of pulse p(t)
    # Max 4Tb time support (Tb = bit duration for a single bit)

    # Tx AM must be 2.4-2.5 GHz with 50dB less than peak outside band
    # 20*log10(|Pm(f)|/|Pm(fpeak)|) < 50 dB

    # Goal: Find the best possible pulse shape for this system and attach plots
    # of the pulse spectrum that demonstrate the validity of your pulse design.

    t1 = 0 # start time
    t2 = 1 # end time
    N = 2**8 # nof samples
    t = np.linspace(t1, t2, N)
    fs = N/(t2-t1) # sample rate
    Ts = 1/fs # period

    # carrier signal (sine wave)
    carrier = np.sin(2*np.pi*t)
    plt.figure()
    plt.plot(carrier)
    plt.grid()
    plt.show()


    x = np.sinc(0.5-np.abs(t))
    # X = np.fft.fft(x)
    X = np.fft.fftshift(np.fft.fft(x))
    XX = np.abs(X)*(t2-t1)/N

    PSD = np.abs(np.fft.fft(x))**2 / (N*fs)
    PSD_log = 10.0*np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)
    



