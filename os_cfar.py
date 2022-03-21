#######################################
# Function:os_cfar
# Author: Eden_HU
# Describtion: using os_cfar algorithm to detect moving target
# 
#######################################
import numpy as np

def os_cfar(data:np.array,nd:int,nr:int,K:float,L:float): #rectangle_window
    Nr = data.shape[0]
    Nd = data.shape[1]
    Vt = np.zeros(data.shape)
    CFAR_out = np.zeros(data.shape)
    for ii in range(0,Nr):
        for jj in range(0,Nd):
            V_ij = np.abs(data[ii,jj])
            if ii - nr < 0 :
                Rl = 0
            else:
                Rl = ii - nr
            
            if ii + nr > Nr - 1:
                Rh = Nr
            else:
                Rh = ii + nr

            if jj - nd < 0 :
                Dl = 0
            else:
                Dl = jj - nd
            
            if jj + nd > Nd - 1:
                Dh = Nd
            else:
                Dh = jj + nd
            
            Vt = np.sum((L*np.abs(data[Rl:ii-3,jj]) < V_ij)) + np.sum((L*np.abs(data[ii+4:Rh,jj]) < V_ij)) + np.sum((L*np.abs(data[ii,Dl:jj-3]) < V_ij)) + np.sum((L*np.abs(data[ii,jj+4:Dh]) < V_ij))

            if Vt >= K:
                CFAR_out[ii,jj] = 1
            else:
                CFAR_out[ii,jj] = 0

    return CFAR_out,Vt
