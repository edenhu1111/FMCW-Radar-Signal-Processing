############################################
# Name:     data_disp_3d
# version:  -.0 
# Author:   Eden HU
# Date:     2020/2/13
# Describtion:
# 显示3D-PLOT(渲染时间很长)
#
############################################
from numbers import Complex
import numpy as np
from numpy.core.function_base import linspace
from scipy.fftpack import fft,fft2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

c = 3e+8

def data_disp_3d(data,Fs,BW,fc):
    fig = plt.figure( figsize=(12, 8))
    ax = Axes3D(fig)
    DATA = np.zeros(np.shape(data))
    DATA = data
    Nr = DATA.shape[0]
    Nd = DATA.shape[1]
    PRF = 0.5e+3
    K = BW * PRF

    r = np.linspace(0,50,Nr,endpoint=False)
    v = np.linspace(0,c*PRF/2/fc,Nd,endpoint=False)

    V,R = np.meshgrid(v,r)

    ax.plot_surface(R, V, DATA, rstride=1, cstride=1,cmap='YlGn', edgecolor='none')
    ax.set_xlabel('Range')
    ax.set_ylabel('Velocity')
    plt.show()


    