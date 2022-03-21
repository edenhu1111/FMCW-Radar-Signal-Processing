############################################
# Name:     czt
# version:  -.0 
# Author:   Eden HU
# Date:     2020/1/11
# Describtion:
# to realize chirp-z transformation on array x
#
############################################
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
def czt(x_in,M_in,A_in,W_in):
# Copy the input values to local parameters
    N = np.size(x_in)
    x = np.reshape(x_in,(N,1))
    M = int(M_in)
    A = complex(A_in)
    W = complex(W_in)
# Select a proper FFT value(2^n) for convolution
    i = 0
    while np.log2(N + M - 1) > i:
        i = i + 1
    L = 2**i
# Define and initialize the signal g(n) and h(n)
    g = np.zeros(L,dtype=complex)
    h = np.zeros(L,dtype=complex)
    for ii in range(0,N):
        g[ii] = x[ii,0] * A**(-ii) * W**(ii**2/2)
    for ii in range(0,L):
        if ii < M:
            h[ii] = W**(-ii**2/2)
        else:
            h[ii] = W**(-(L-ii)**2/2)
    G = fft(g)
    H = fft(h)

    q = ifft(G*H)

    Xz_out = q[0:M] / h[0:M]
    return Xz_out

if __name__ =='__main__':
    t = np.linspace(0,10,128)
    x = np.exp(1j*2*np.pi*t)+2
    X1 = czt(x,1024,1,np.exp(-1j*2*np.pi/4096))
    # X2 = czt(x,1024,1,np.exp(1j*2*np.pi/2048))
    # X2 = fft(x)
    p1 = plt.plot(abs(X1),label="czt")
    # p2 = plt.plot(abs(X2),label="fft")
    plt.legend()
    plt.show()