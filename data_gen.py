############################################
# Name:     data_gen
# version:  -.0 
# Author:   Eden HU
# Date:     2020/1/11
# Describtion:
# 生成数据
# Param:
# R:距离
# V:速度
# Fs:采样频率
# Rref:Dechirp参考距离，一般取0
# BW：扫频带宽
# Fc:载频
# Nr:快时间点数
# Nd：慢时间点数
# PRF：脉冲发射频率
############################################
from numbers import Complex
import os
import numpy as np
from numpy.core.function_base import linspace
from scipy.fftpack import fft,fft2
import matplotlib.pyplot as plt
from data_disp_3d import data_disp_3d
from czt import czt
from os_cfar import os_cfar

c = 3e+8
pi = np.pi
def data_gen(R,V,Fs,Rref,BW,Fc,Nr,Nd,PRF):
    K = BW * PRF
    S = np.zeros((Nr,Nd),dtype=complex)
    tau_ref = 2*Rref/c
    for ii in range(0,Nd):
        tau = 2*(R+ii*V/PRF)/c
        for jj in range(0,Nr):
            tau = tau + 2/c*V/Fs
            if (tau < jj/Fs) and (jj/Fs > tau_ref):
                S[jj,ii] = np.exp(1j*2*pi*(Fc*(tau - tau_ref)+K*(jj/Fs - tau_ref)*(tau - tau_ref) - K/2*(tau-tau_ref)**2))
    return S

# The function main here shows the influence of Range-Doppler coupling

def main():        
    Sif1 = 0.001 * data_gen(20,20,2e+6,0,100e+6,2.4e+9,1024,128,1.5e+3)
    Sif1 = Sif1 + np.random.normal(0,0.05,Sif1.shape)
    # Sif2 = data_gen(100,5,2e+6,0,100e+6,2.4e+9,1024,512)
    # for ii in range(0,Sif1.shape[0]):
    #     Sif1[ii,:] = Sif1[ii,:]*np.hanning(Sif1.shape[1])
    #     # Sif2[ii,:] = Sif2[ii,:]*np.hanning(Sif1.shape[1])
    for jj in range(0,Sif1.shape[1]):
        Sif1[:,jj] = Sif1[:,jj]*np.hanning(Sif1.shape[0])
        # Sif2[:,jj] = Sif2[:,jj]*np.hanning(Sif1.shape[0])

    Result_2D1 = fft2(Sif1)
    # Result_2D2 = fft2(Sif2)
    cfar_result,Vt = os_cfar(Result_2D1,16,8,10,5)
    # plt.subplot(1,2,1)
    plt.imshow(20*np.log(np.abs(Result_2D1[0:100,:])),cmap='hot_r')
    plt.colorbar()
    plt.yticks(np.linspace(0,100,10),np.linspace(0,100*2e+6/1024*c/2/(100e+6*1.5e+3),10))
    # plt.yticks(np.linspace(0,128,10),np.linspace(0,1.5e+3))
    # data_disp_3d((np.abs(Result_2D1)),2e+6,100e+6,5e+9)
    # plt.subplot(1,2,2)
    # plt.imshow((np.abs(cfar_result)),cmap='hot_r')
    # data_disp_3d(np.abs(Result_2D1),2e+6,100e+6,2.4e+9)
    data_disp_3d((np.abs(cfar_result)),2e+6,100e+6,2.4e+9)
    # plt.figure(2)
    # plt.plot(np.abs(fft(Sif1[:,0])))
    # plt.xticks(np.arange(0,1024,500/3),np.arange(0,1024*1.5,250))
    # plt.xlabel('Range/m')
    # plt.title('FFT Result')
    plt.show()

    

if __name__ == '__main__':
    main()