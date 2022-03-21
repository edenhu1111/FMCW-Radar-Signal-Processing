import threading
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
from data_gen import data_gen
from rsp import radarSignal
from os_cfar import os_cfar
import socket
from time import sleep
hostAddress =   ('192.168.0.3',8080)
serverAddress = ('192.168.0.2',8080)
# plt.subplots_adjust(bottom = 0.2)
counter = 0
R = 20
V = 5
data_counter = 0
raw_data = np.zeros([512,1],dtype=np.int16)
radarData = radarSignal(raw_data,300e+3,200e+6,2.2e+9,0.5e+3)
lock = threading.Lock()


class ThreadUSB(threading.Thread):                                                                        #Get Data thread
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):  
        global raw_data
        global radarData
        global counter  
        global data_counter
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)
        s.bind(hostAddress)
        s.connect(serverAddress)
        buf = {}
        while True: 
            
            buf = np.frombuffer(s.recv(1024), dtype=np.uint8)
            data_counter = data_counter + 1
            raw_data = np.int16(np.bitwise_or(np.left_shift(np.uint16(buf[0::2]),8),np.uint16(buf[1::2])))
            raw_data[raw_data>2048] = raw_data[raw_data>2048] - 4096
            lock.acquire()
            radarData.update_data(np.reshape(np.float32(raw_data/2048),[512,1]))
            lock.release()
            if data_counter%50 == 0:
                print(data_counter)

class ThreadProcessing(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global radarData
        global f
        while True:
            plt.cla()

            ft_r_result = radarData.ft_r(0,0.5,25)
            ft_r_result[0,:] = 0
            plt.plot(np.abs(ft_r_result[10:50,:]))
            plt.ylabel('Amp')
            plt.ylim([0,64])
            plt.xlabel('Range/m')
            plt.xticks(np.arange(0,40,10),np.arange(5,25,5))
            # plt.xticks(np.arange(0,70,10/0.73),np.arange(-25,26,10))
            plt.title('Radar Mathcing Filtering Result') 
            plt.draw()
            sleep(0.25)

if __name__ == '__main__': 
    plt.figure(1)
    plt.plot(raw_data)
    # plt.yticks(np.arange(0,101,10),np.arange(0,51,5))
    plt.ylabel('Amp')
    plt.ylim([-2048,2048])
    plt.xlabel('t')
    # plt.xticks(np.arange(0,70,10/0.73),np.arange(-25,26,10))
    plt.title('Radar Detection Result')   
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