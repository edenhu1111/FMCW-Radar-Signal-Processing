import threading
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
from data_gen import data_gen
from rsp import radarSignal
from os_cfar import os_cfar
from time import sleep


# plt.subplots_adjust(bottom=0.2)
counter = 0
R = 20
V = 5
raw_data = np.zeros([512,64])
radarData = radarSignal(raw_data,300e+3,200e+6,2.4e+9,0.5e+3)
lock = threading.Lock()
f = plt.imshow(raw_data[0:60,:],cmap='GnBu',vmin=0,vmax=100)
plt.yticks(np.arange(0,61,10),np.arange(0,31,5))
plt.ylabel('Range/m')
plt.xlabel('Velocity')
plt.xticks(np.arange(0,70,10/0.73),np.arange(-25,26,10))
plt.title('Radar Detection Result')
plt.colorbar()

class ThreadUSB(threading.Thread):                                                                          #Get Data thread
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):  
        global raw_data
        global radarData
        global counter  
        while True:                                                                          #cover the origin run() method
            raw_data = 0.3 * data_gen(R+V*0.01*counter,V,300e+3,0,200e+6,2.4e+9,512,64,0.5e+3) 
            raw_data = raw_data + np.random.normal(0,0.000001,raw_data.shape)
            counter = (counter + 1) % 200
            lock.acquire()
            radarData.update_data(raw_data)
            lock.release()

class ThreadProcessing(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global radarData
        global f
        while True:
            ft_r = radarData.ft_r_MTI(5,0.5,35)

            ft_r = fft(np.hanning(64)*ft_r,axis=1)
            ft_r = fftshift(ft_r,axes=1)
            # cfar_out,vt= os_cfar(ft_r,20,8,22,10)
            f.set_data(np.abs(ft_r))
            # f.set_data(np.abs(cfar_out))
            plt.draw()
            

if __name__ == '__main__':    
    t1 = ThreadUSB()
    t2 = ThreadProcessing()
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()
    # t1.join()
    # t2.join()
    plt.show()
    plt.close()