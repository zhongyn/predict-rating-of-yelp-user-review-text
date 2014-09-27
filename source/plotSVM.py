import matplotlib.pyplot as pl
import numpy as np


acc = np.load('accSVMAdj.npy')
accuracy = np.mean(acc, axis=2)
c = [1,10,100,1000]


fig,ax1 = pl.subplots()
fonts = 20
ax1.set_title('Support Vector Machine',size=fonts)
ax1.plot(c,accuracy[:,0],color='b',marker='o',label='linear',linewidth=2)
ax1.plot(c,accuracy[:,1],color='r',marker='o',label='polynominal',linewidth=2)
ax1.plot(c,accuracy[:,2],color='y',marker='o',label='rbf',linewidth=2)
ax1.legend(fancybox=True,loc='best',fontsize=18)
ax1.set_xlabel('c',size=fonts)
ax1.set_xscale('log')
ax1.set_ylabel('Average accuracy',size=fonts)


pl.show()