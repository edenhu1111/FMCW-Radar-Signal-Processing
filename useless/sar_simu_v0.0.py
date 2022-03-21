############################################
# Name:     sar_simu
# version:  0.0 
# Author:   Eden HU
# Date:     2020/12/22
# Describtion:
#
#
############################################
import numpy as np
from scipy.fftpack import fft,ifft,fft2,ifftshift
import matplotlib.pyplot as plt
#from matplotlib.pylab import mpl

#Basic Parameter about the radar system
Fs = 100e+6                              #Fs refers to Sampling Frequency
fastTimeDotsNum = 512                    #The number of each fast time sampling
slowTimeDotsNum = 128
PRF = Fs/fastTimeDotsNum                 #PRF refers to Pulse Repeating Frequency
stopGapDistance = 0.02              
velocityOfSAR   = stopGapDistance * PRF  #equivalent velocity of SAR
La = slowTimeDotsNum * stopGapDistance   #synthetic aperture

c  = 3e+8                                #Ray Velocity 3e+8
fc = 5e+9                                #center frequency of the radar system
bw = 100e+6                              #bandwidth of the radar system
K  = bw * PRF                            #chirp rate

#Target Point
Rb = 10                                   #The coordinate of the point target is (5,50)
xp = 0.2                                 #supposed that the target is always in the detected range

#Initialize the signal to transmit
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
# fft2_result = fft2(s_Rx)
fft2_result = np.zeros(s_Rx.shape,dtype=complex)
for i in range(0,slowTimeDotsNum):
    fft2_result[:,i] = fft((s_Rx[:,i] *np.hamming(fastTimeDotsNum)))
max_idx = np.argmax(np.abs(fft2_result[:,0]))
print(max_idx)
slow_time = fft2_result[max_idx,:]


for i in range(0,fastTimeDotsNum):  
    fft2_result[i,0:slowTimeDotsNum] = fft2_result[i,0:slowTimeDotsNum]
    fft2_result[i,:] = fft((fft2_result[i,:] - np.mean(s_Rx[i,:])))

mF_UIR = np.zeros([fastTimeDotsNum,slowTimeDotsNum])                                       #Create Matching Filter's Unit Impulse Response
mF_TF  = np.zeros(mF_UIR.shape,dtype=complex)
#Matching Filter
for j in range(1,fastTimeDotsNum):
    if j < fastTimeDotsNum/2:
        Rs = j * Fs/fastTimeDotsNum * c/(2*K)
    else:
        Rs = (fastTimeDotsNum - j) * Fs/fastTimeDotsNum * c/(2*K)
    for i in range(0,slowTimeDotsNum):
        mF_UIR[j,i] = np.cos(-2*np.pi*2*(Rs + ((velocityOfSAR*(i - slowTimeDotsNum/2)/PRF)**2)/(2*Rs))*fc/c + 4*np.pi*K*(Rs/c)**2)
    mF_UIR[j,:] = mF_UIR[j,:]  * np.hamming(slowTimeDotsNum)
    mF_TF[j,:] = fft((mF_UIR[j,:] - np.mean(mF_UIR[j,:])))                                #TF of Matching Filter 

#Compress  (bugs may exsit here!)
imaging_result = np.zeros(fft2_result.shape)                                    #The array to store the result of imaging
fft2_matching_result = np.zeros(fft2_result.shape,dtype = complex)              #The array to store the result of matching filter

for i in range(0,fastTimeDotsNum):
    fft2_matching_result[i,:] = fft2_result[i,:] * mF_TF[i,:]                        #Matching Filter
    imaging_result[i,:] = np.abs(ifft(fft2_matching_result[i,:]))               #ifft
imaging_result = np.append(imaging_result[:,int(slowTimeDotsNum/2):slowTimeDotsNum],imaging_result[:,0:int(slowTimeDotsNum/2)],axis=1)


# show the graphs
plt.figure(1)
plt.title("Results")
#原数据
a1 = plt.subplot(2,2,1)
plt.imshow(s_Rx[0:slowTimeDotsNum,0:slowTimeDotsNum],cmap='gray')
a1.set_title('Raw Data')
#2D-FFT结果
a2 = plt.subplot(2,2,2)
plt.imshow(np.abs(fft2_result[0:slowTimeDotsNum,0:slowTimeDotsNum]),cmap='gray')
a2.set_title('FFT Result')
#匹配滤波后，成像结果
a3 = plt.subplot(2,2,3)
a3.set_title('matching filted result')
plt.plot(abs((imaging_result[max_idx,:])))

a4 = plt.subplot(2,2,4)
a4.set_title('Image')
plt.imshow(20*np.log(imaging_result[0:slowTimeDotsNum,0:slowTimeDotsNum]),cmap='gray')


plt.figure(2)
# plt.subplot(1,2,1)
# plt.imshow(abs(fft2_result[:,0:slowTimeDotsNum]),cmap='gray')
# plt.plot(mF_UIR[0:slowTimeDotsNum])
plt.subplot(1,2,1)
plt.plot(slow_time)
plt.subplot(1,2,2)
plt.plot(mF_UIR[max_idx,:])
# plt.subplot(1,2,2)
# plt.subplot()
plt.show()