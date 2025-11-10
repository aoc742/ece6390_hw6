import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


# Tx Pulse amplitude modulation
def pam_mod():
    return

# Tx Amplitude modulation
def am_mod():
    return

if __name__ == "__main__":
    t1 = 0 # start time
    t2 = 1 # end time
    N = 512
    t = np.linspace(t1, t2, N)
    f = np.arange(-N/2, N/2) / (t2-t1) # frequency range

    x = np.sinc(0.5-abs(t))
    padded_x = np.pad(x, int(2**np.ceil(np.log2(len(x)))))
    X = np.fft.fft(padded_x)

    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title("Sinc (time)")

    XX = abs(np.fft.fftshift(X))*(t2-t1)/N

    plt.subplot(2, 1, 2)
    plt.plot(XX)
    plt.title("Shifted FFT")

    plt.show()



