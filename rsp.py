from os_cfar import os_cfar
import numpy as np
from numbers import Complex
from numpy.core.fromnumeric import shape
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from czt import czt
from data_gen import data_gen
from data_disp_3d import data_disp_3d
c = 3e+8                                             #The velocity of light

class radarSignal():
    def __init__(self,rawData:np.array,Fs:float,BW:float,Fc:float,PRF:float) -> None:
        self.data = rawData
        self.shape = self.data.shape
        self.fs = Fs
        self.bw = BW
        self.fc = Fc
        self.nr = self.shape[0]
        self.nd = self.shape[1]
        self.prf = PRF
        self.k = self.bw * self.prf
        self.w = np.hamming(self.nr)
        self.w_t = np.reshape(np.hamming(self.nr),[self.nr,1])

    def print(self) -> None:
        plt.imshow(np.abs(self.data))
        plt.show()

    def ft_r(self,rStart:float,rRes:float,rEnd:float):
        fStart = rStart * 2 * self.k / c
        fRes = rRes * 2 * self.k / c
        # fEnd = rEnd * 2 * self.k / c
        m = int((rEnd - rStart)/rRes)
        A = np.exp( 1j*fStart/self.fs*2*np.pi)
        W = np.exp(-1j*fRes  /self.fs*2*np.pi)
        ft_r_out = np.zeros([m,self.nd],dtype=complex)
        # ft_r_out = np.zeros([self.nr,self.nd],dtype=complex)
        for jj in range(0,self.nd):
            ft_r_out[:,jj] = czt(self.w * self.data[:,jj],m,A,W)
        return ft_r_out
    
    def ft_r_decoupled(self,rStart:float,rRes:float,rEnd:float):
        ft_r_result = self.ft_r(rStart,rRes,rEnd)
        fRes = rRes * 2 * self.k / c
        for i in range(0,ft_r_result.shape[0]):
            ft_r_result[i,:] = ft_r_result[i,:] * np.exp(-1j*np.pi*(i*fRes)**2/self.k)
        return ft_r_result

    def ft_r_MTI(self,rStart:float,rRes:float,rEnd:float):
        
        fStart = rStart * 2 * self.k / c
        fRes = rRes * 2 * self.k / c
        fEnd = rEnd * 2 * self.k / c
        m = int((fEnd - fStart)/fRes)
        A = np.exp( 1j*fStart/self.fs*2*np.pi)
        W = np.exp(-1j*fRes  /self.fs*2*np.pi)
        ft_r_out = np.zeros([m,self.nd],dtype=complex)
        ft_r_original = np.zeros([m],dtype=complex)
        ft_r_last = np.zeros([m],dtype=complex)
        # ft_r_out = np.zeros([self.nr,self.nd],dtype=complex)
        for jj in range(0,self.nd):
            # ft_r_out[:,jj] = czt(w * self.data[:,jj],m,A,W)
            ft_r_original = czt(self.w * self.data[:,jj],m,A,W)
            # ft_r_out[:,jj] = fft(self.data[:,jj])
            ft_r_out[:,jj] = ft_r_original - ft_r_last
            ft_r_last = ft_r_original
        return ft_r_out
    # def optiInit(self,rStart:float,rRes:float,rEnd:float):
    #     fStart = rStart * 2 * self.k / c
    #     fRes = rRes * 2 * self.k / c
    #     # fEnd = rEnd * 2 * self.k / c
    #     A = np.exp( 1j*fStart/self.fs*2*np.pi)
    #     W = np.exp(-1j*fRes  /self.fs*2*np.pi)
    #     N = self.nr
    #     M= int((rEnd - rStart)/rRes)
    #     i = 0
    #     while np.log2(N + M - 1) > i:
    #         i = i + 1
    #     L = 2**i
    #     # Define and initialize the signal g(n) and h(n)
    #     self.g_o = np.zeros([L,1],dtype=complex)
    #     h = np.zeros([L,1],dtype=complex)
    #     for ii in range(0,N):
    #         self.g_o[ii,0] = A**(-ii) * W**(ii**2/2)
    #     for ii in range(0,L):
    #         if ii < M:
    #             h[ii,0] = W**(-ii**2/2)
    #         else:
    #             h[ii,0] = W**(-(L-ii)**2/2)
    #     self.H = np.reshape(fft(h),[L,1])
    # def ft_r_optimized(self):
    #     g = self.data * self.w_t * self.g_o
    #     G = fft(g,axis=0)
    #     return ifft(G*self.H,axis=0)

    def update_data(self,newData:np.array):
        self.data = newData


def main():
    Sif1 = 0.001 * data_gen(20,5,300e+3,0,100e+6,2.4e+9,512,64,0.5e+3) + 0.0004 * data_gen(10,7,300e+3,0,100e+6,2.4e+9,512,64,0.5e+3)
    Sif1 = Sif1 + np.random.normal(0,0.005,Sif1.shape)

    radarData = radarSignal(Sif1,300e+3,100e+6,2.4e+9,0.5e+3)
    radarData.optiInit(5,0.5,25)
    ft_r = radarData.ft_r_optimized()
    # ft_r = radarData.ft_r(0,0.5,50)

    plt.figure(1)
    plt.imshow(np.abs(ft_r),cmap='GnBu')
    plt.yticks(np.arange(0,100,20),np.arange(0,50,10))
    plt.ylabel('range/m')
    # plt.plot(np.linspace(0,100,200),np.abs(ft_r[:,0]))
    plt.xlabel('N')
    plt.title('FT_Result')
    plt.figure(3)
    plt.plot(np.abs(ft_r[:,0]))
    plt.xticks(np.arange(0,100,20),np.arange(0,50,10))
    plt.title('SNR=-24dB')
    plt.xlabel('Range/m')
    plt.ylabel('Voltage')
    for i in range(0,ft_r.shape[0]):
        ft_r[i,:] = fft(ft_r[i,:])
    plt.figure(2)
    cfar_out,Vt = os_cfar(ft_r,10,7,14,3)
    data_disp_3d((np.abs(cfar_out)),300e+3,100e+6,2.4e+9)

    # data_disp_3d((np.abs(ft_r)),300e+3,100e+6,2.4e+9)
    # plt.imshow(np.abs(ft_r),cmap='GnBu')
    # plt.yticks(np.arange(0,101,20),np.arange(0,51,10))
    # plt.xticks(np.arange(0,128,20),np.arange(0,101,10))
    # plt.ylabel('range/m')
    # plt.colorbar()
    plt.show()


    # radarData.print()

if __name__  == '__main__':
    main()