import numpy as np 
import matplotlib.pyplot as plt

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

sample_size=50
x = np.arange(sample_size-1,15000)
d = np.loadtxt('dlosses.txt', dtype=float)
g = np.loadtxt('glosses.txt', dtype=float)
plt.plot(x,moving_average(d,sample_size),'r.',label='discriminator')
plt.plot(x,moving_average(g,sample_size),'b.',label='generator')
plt.legend(loc='best')
plt.show()
