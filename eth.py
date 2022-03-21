##测试UDP传输
import socket
import threading
from time import sleep
from threading import Thread

addressS = ('127.0.0.1', 10001)     #收端地址+端口
addressC = ('127.0.0.1', 10002)     #发端地址+端口
msg = 'hello!'
data_rcv = '\x00'
lock = threading.Lock()
class threadClient(Thread):
    def __init__(self):
        Thread.__init__(self)
    
    def run(self):
        ## Initialize a new socket class
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, proto=socket.IPPROTO_UDP)
        ## 绑定一个IP地址和端口(绑定本机的ip和端口)
        sock.bind(addressC)
        while True:
            lock.acquire()                              #上线程锁
            sock.sendto(msg.encode('utf-8'),addressS)   #发送套接字
            lock.release()                              #解除线程锁
            sleep(1)                                    #sleep 1秒
        sock.close()
class threadServer(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        global data_rcv
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,proto=socket.IPPROTO_UDP)
        sock.bind(addressS)
        sock.connect(addressC)
        while True:
            # data_rcv,addr = sock.recvfrom(11)
            data_rcv = sock.recv(11)
            data_rcv = data_rcv.decode('utf-8')
            # print('Receive:',data_rcv,'from',addr)
            print('Receive:',data_rcv)
        sock.close()

def main1():
    t1 = threadClient()
    t2 = threadServer()

    t1.start()
    t2.start()
    while True:
        pass
if __name__ == '__main__':
    main1()