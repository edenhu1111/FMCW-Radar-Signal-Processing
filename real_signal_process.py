import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
from data_gen import data_gen
from rsp import radarSignal
from os_cfar import os_cfar
import scipy.io as scio
from data_disp_3d import data_disp_3d

rawdata_dict = scio.loadmat('realdata_1.mat')
rawdata = rawdata_dict['c']

radarData = radarSignal(rawdata,300e+3,200e+6,2.2e+9,500)
ft = radarData.ft_r(1,0.5,30)
# data_disp_3d(ft,300e+3,200e+6,2.2e+9)
for i in range(0,ft.shape[0]):
    ft[i,:] = fft(ft[i,:])
    ft[i,:] = fftshift(ft[i,:])
plt.imshow(np.abs(ft),'gray')
plt.yticks(np.arange(8,58,10),np.arange(5,30,5))
plt.xticks(np.arange(3,60,5/0.54),np.arange(-15,16,5))
plt.ylabel('Range/m')
plt.xlabel('Velocity/(m/s)')
plt.title('Still Object Matching Result')
plt.show()