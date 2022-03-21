## 延迟对消器幅频特性曲线绘图
import numpy as np
import matplotlib.pyplot as plt

w = np.linspace(-1.5,1.5,1000)
h1 = 1 - np.exp(-1j*2*np.pi*w)
h2 = h1**2
h3 = h1**3
H1 = 20*np.log10(np.abs(h1)/2)
H2 = 20*np.log10(np.abs(h2)/4)
H3 = 20*np.log10(np.abs(h3)/8)

plt.figure(1)
p1, = plt.plot(w,H1,'-',color='#000000')
p2, = plt.plot(w,H2,'--',color='#000000')
p3, = plt.plot(w,H3,'-.',color='#000000')

plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude(dB)')
plt.ylim([-50,0])
plt.grid(True)

plt.title('Mag-Freq Plot')
plt.legend([p1,p2,p3],['Single','Double','Tripple'],loc='lower right')
plt.show()