from scipy import ndimage
import numpy as np
from rsp import radarSignal

c = 3e+8

def DiscreteRadonTransform(radarsig:radarSignal,rStart:float,rRes:float,rEnd:float,v_res:float):
    ft_r = radarSignal.ft_r(radarsig,rStart,rRes,rEnd)
    fRes = rRes * 2 * radarsig.k / c
    vmax = 40
    res = np.zeros([ft_r.shape[0],int(2*vmax/v_res)],dtype='complex')
    M = res.shape[0]
    N = res.shape[1]
    for m in range(0,M):
        for n in range(0,N):
            R=m * rRes
            v = -vmax + n * v_res
            fd = 2*v/c*radarsig.fc
            for ii in range(0,radarsig.nd):
                fpeak = int(np.round((2*radarsig.k*(R+v*ii/radarsig.prf)/c+fd)/fRes))
                if fpeak >= M:
                    fpeak = M - 1
                elif fpeak < 0:
                    fpeak = 0
                res[m,n] = res[m,n] + ft_r[fpeak,ii]*np.exp(-2j*np.pi*(fd-4*radarsig.k*R*v/(c**2))*ii/radarsig.prf)
    return res
