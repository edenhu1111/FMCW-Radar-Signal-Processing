############################################
# Name:     sar_simu
# version:  1.0 
# Author:   Eden HU
# Date:     2020/12/22
# Describtion:
# 加入了距离校正
# 用函数封装了成像算法
############################################
from numbers import Complex
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from czt import czt


# Basic Parameter about the radar system
Fs = 1e+6                                #Fs refers to Sampling Frequency
fastTimeDotsNum = 512                    #The number of each fast time sampling
slowTimeDotsNum = 128
PRF = 200                                #PRF refers to Pulse Repeating Frequency
stopGapDistance = 1              
velocityOfSAR   = stopGapDistance * PRF  #equivalent velocity of SAR
La = slowTimeDotsNum * stopGapDistance   #synthetic aperture

c  = 3e+8                                #Ray Velocity 3e+8
fc = 5e+9                                #center frequency of the radar system
bw = 100e+6                              #bandwidth of the radar system
K  = bw * Fs/fastTimeDotsNum             #chirp rate

#Target Point
Rb = 50                                 #The coordinate of the point target is (5,50)
xp = 0.2                                 #supposed that the target is always in the detected range
def r_dImaging(raw_data,fs,V):
    N = raw_data.shape[0]
    M = raw_data.shape[1]
    fft_dpl = np.zeros(raw_data.shape,dtype=complex)
    fft_range = np.zeros(raw_data.shape,dtype=complex)
    range_adj = np.zeros(raw_data.shape,dtype=complex)
    dplCmprs_TF = np.zeros(raw_data.shape,dtype=complex)
    plt.figure(1)
    f1 = plt.subplot(2,3,1)
    plt.imshow(np.abs(raw_data[0:M,0:M]),cmap='gray')
    plt.colorbar()
    f1.set_title('Raw Data')
###### 显示距离随雷达位移的变化 #######
    for i in range(0,M):
        fft_range[:,i] = fft(np.hamming(N) * raw_data[:,i])
        plt.figure(1)
    f2 = plt.subplot(2,3,2)
    plt.colorbar()
    plt.imshow(np.abs(fft_range[0:slowTimeDotsNum,0:slowTimeDotsNum]),cmap='gray')
    f2.set_title('R-tm plot')
#####################################

###### 显示距离随雷达多普勒频率的变化 #######
    for i in range(0,N):
        fft_range[i,:] = fft(fft_range[i,:] * np.hamming(M))
    f3 = plt.subplot(2,3,3)
    plt.imshow(np.abs(fft_range[0:slowTimeDotsNum,0:slowTimeDotsNum]),cmap='gray')
    plt.colorbar()
    f3.set_title('R-fd plot')
##########################################

    for i in range(0,N):
        fft_dpl[i,:] = fft(np.hamming(M)*raw_data[i,:])                                 #dpl-domain fft
    for i in range(0,M):
        if i<M/2:
            for j in range(0,N):
                range_adj[j,i] = np.exp(-1j*2*np.pi*K*5/c*(((i*PRF/M)/(2*V*(fc /c)))**2)*j/fs)
                # range_adj[j,i] = 1
        else:
            for j in range(0,N):
                range_adj[j,i] = np.exp(-1j*2*np.pi*K*5/c*((((i-M)*PRF/M)/(2*V*(fc /c)))**2)*j/fs)  
                # range_adj[j,i] = 1

        fft_dpl[:,i] = range_adj[:,i] * fft_dpl[:,i]
        fft_dpl[:,i] = fft(np.hamming(N) * fft_dpl[:,i])

    f6 = plt.subplot(2,3,6)
    plt.imshow(np.real(range_adj),cmap='gray')
    f6.set_title('Adjusting Function')
    ###### 显示校正后距离随雷达多普勒频率的变化 #######
    f4 = plt.subplot(2,3,4)
    plt.imshow(np.abs(fft_dpl[0:slowTimeDotsNum,0:slowTimeDotsNum]),cmap='gray')
    plt.colorbar()
    f4.set_title('After Adj')
    ##########################################
    for i in range(0,N):
        for j in range(0,int(M/2)):
            dplCmprs_TF[i,j] = np.exp(-1j*2*np.pi*(i*fs*c/2/N/K)/V*np.sqrt((2*V*(fc /c))**2 - (j*PRF/M)**2))
        fft_dpl[i,:] = fft_dpl[i,:] * dplCmprs_TF[i,:]
        fft_dpl[i,:] = ifft(fft_dpl[i,:])
    ############## 显示成像结果 ###############
    f5 = plt.subplot(2,3,5)
    plt.imshow(np.abs(fft_dpl[0:slowTimeDotsNum,0:slowTimeDotsNum]),cmap='gray')
    plt.colorbar()
    f5.set_title('Image')
    plt.show()
    ##########################################
    return fft_dpl

def main():
    s_Tx = np.zeros(fastTimeDotsNum)
    s_Rx = np.zeros([fastTimeDotsNum,slowTimeDotsNum])         
    for i in range(0,fastTimeDotsNum):
        t = i/Fs
        s_Tx[i] = np.cos(2*np.pi*fc*t + np.pi*K*t**2)
                                            #CHIRP Tx Signal

    #Produce original 2D-Data
    for i in range(0,slowTimeDotsNum):
        R = ((i*stopGapDistance - xp)**2 + Rb**2)**0.5                  #The distance between radar and target
        time_diff = 2*R/c                                               #time delay
        idx_diff = int(np.around(time_diff * Fs))
        for j in range(0,fastTimeDotsNum - idx_diff):
            s_Rx[j + idx_diff,i] = np.cos(-2.0*np.pi * (fc*time_diff + K*time_diff*j/Fs - K*(time_diff**2)/2.0))
        #相位差：-2*pi*timediff*fc - 2pi*K*timediff*t + 2pi*1/2*K*timediff**2
    r_dImaging(s_Rx,Fs,velocityOfSAR)

if __name__ == '__main__':
    main()
    # print(2*velocityOfSAR**2/c*fc*(slowTimeDotsNum/PRF/2)**2)