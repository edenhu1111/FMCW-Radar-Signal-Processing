import threading
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
from rsp import radarSignal
from os_cfar import os_cfar
import socket
hostAddress =   ('192.168.0.3',8080)    #主机地址
serverAddress = ('192.168.0.2',8080)    #从机(FPGA)地址
# plt.subplots_adjust(bottom = 0.2)
# counter = 0
# R = 20
# V = 5
data_counter = 0
raw_data = np.zeros([512,64],dtype=np.int16)
radarData = radarSignal(raw_data,300e+3,200e+6,2.4e+9,0.5e+3)
lock = threading.Lock()
w = np.hanning(64)
f = plt.imshow(raw_data[0:40,:],cmap='GnBu',vmin = 0,vmax = 1)
plt.yticks(np.arange(0,41,10),np.arange(5,26,5))
plt.ylabel('Range/m')
plt.xlabel('Velocity')
plt.xticks(np.arange(0,60,0.5328125*20),np.arange(-15,15,5))
plt.title('Radar Detection Result')
plt.colorbar()

class ThreadUDP(threading.Thread):                                                                        #Get Data thread
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):  
        global raw_data
        global radarData
        # global counter  
        global data_counter
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)
        s.bind(hostAddress)
        s.connect(serverAddress)
        buf = {}
        while True: 
            for i in range(64):
                buf = np.frombuffer(s.recv(1024), dtype=np.uint8)
                data_counter = data_counter + 1
                raw_data[:,i] = np.bitwise_or(np.left_shift(np.int16(buf[0::2]),8),np.int16(buf[1::2]))
                raw_data[raw_data>2048] = raw_data[raw_data>2048] - 4096
                if data_counter%50 == 0:
                    print(data_counter)
            lock.acquire()
            radarData.update_data(np.float32(raw_data)/2048*5)
            lock.release()

class ThreadProcessing(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global radarData
        global f
        while True:
            lock.acquire()
            ft_r = radarData.ft_r_MTI(5,0.5,25)
            lock.release()
            for i in range(0,ft_r.shape[0]):
                ft_r[i,:] = fft(w*ft_r[i,:])
                ft_r[i,:] = fftshift(ft_r[i,:])
            cfar_out,vt= os_cfar(ft_r,20,5,30,5)
            f.set_data(np.abs(cfar_out))
            plt.draw()

if __name__ == '__main__':    
    t1 = ThreadUDP()
    t2 = ThreadProcessing()
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()
    # t1.join()
    # t2.join()
    plt.show()
    plt.close()