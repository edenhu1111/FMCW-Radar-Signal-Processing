import numpy as np
from numbers import Complex
from numpy.core.fromnumeric import shape
from scipy.fftpack import fft,fft2
import matplotlib.pyplot as plt
from czt import czt
from data_gen import data_gen
from rsp import radarSignal
from radon import DiscreteRadonTransform
c = 3e+8                                             #The velocity of light

def main():
    Sif1 = 0.001 * data_gen(20,10,2e+6,0,100e+6,2.4e+9,1024,128,1.5e+3) + 0.001*data_gen(20,20,2e+6,0,100e+6,2.4e+9,1024,128,1.5e+3)
    Sif1 = Sif1 + np.random.normal(0,0.001,Sif1.shape)

    radarData = radarSignal(Sif1,2e+6,100e+6,2.4e+9,1.5e+3)
    ft_r = radarData.ft_r(0,0.5,50)
    # ft_r = radarData.ft_r(0,0.5,50)
    plt.figure(1)
    plt.imshow(np.abs(ft_r),cmap='GnBu')
    # plt.yticks(np.arange(0,101,10),np.arange(0,51,5))
    # plt.xticks(np.arange(0,60,10/0.73),np.arange(0,50,10))
    plt.title('Fast Time Fourier Transform')
    plt.ylabel('Range/m')
    plt.xlabel('N')
    plt.colorbar()
    # plt.show()
    # plt.figure(1)
    # plt.imshow(np.abs(ft_r),cmap='GnBu')
    # plt.yticks(np.arange(0,100,20),np.arange(0,50,10))
    # plt.ylabel('range/m')
    # plt.plot(np.linspace(0,50,100),np.abs(ft_r[:,0]))
    # plt.xlabel('Range(unit:m)')
    # plt.title('FT_Result')
    
    for i in range(0,ft_r.shape[0]):
        ft_r[i,:] = fft(ft_r[i,:])
    res = DiscreteRadonTransform(radarData,0,0.5,50,1)
    plt.figure(2)
    # plt.imshow(np.abs(ft_r),cmap='GnBu')
    plt.imshow(np.abs(res),cmap='GnBu')
    plt.yticks(np.arange(0,100,20),np.arange(0,50,10))
    plt.xticks(np.linspace(0,80,5),np.linspace(-40,40,5))
    # plt.yticks(np.linspace(0,128,5))
    plt.ylabel('range/m')
    plt.xlabel('velocity/(m/s)')
    plt.title('Radon-FT')
    plt.colorbar()
    plt.figure(3)
    # plt.imshow(np.abs(fft2(radarData.data))[0:100,0:60],cmap='GnBu')
    plt.imshow(np.abs(ft_r[:,0:60]),cmap='GnBu')
    plt.yticks(np.arange(0,101,10),np.arange(0,51,5))
    plt.xticks(np.arange(0,60,10/0.73),np.arange(0,50,10))
    plt.title('2D-Fourier Transform')
    plt.xlabel('velocity(m/s)')
    plt.ylabel('Range(m)')
    plt.colorbar()
    plt.show()


if __name__  == '__main__':
    main()