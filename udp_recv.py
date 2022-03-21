import socket

hostAddress =   ('192.168.0.3',8080)
serverAddress = ('192.168.0.2',8080)


def main():
    s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)
    s.bind(hostAddress)
    s.connect(serverAddress)
    while True:
        print(s.recv(1024))
        # s.recv(1024)

if __name__ == '__main__':
    main()