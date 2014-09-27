import matplotlib.pyplot as pl
import numpy as np

accNN = np.load('accNN.npy')
accAdj = np.load('accAdj.npy')
accAllWords = np.load('accAllWords.npy')

k = 10

meanAcc = np.zeros([3,k-1])

for index, item in enumerate(accNN):
	meanAcc[0,index] = np.mean(item)

for index, item in enumerate(accAdj):
	meanAcc[1,index] = np.mean(item)

for index, item in enumerate(accAllWords):
	meanAcc[2,index] = np.mean(item)



kfolds = np.arange(2,k+1)

fig,ax1 = pl.subplots()
fonts = 20
ax1.set_title('Naive Bayes Classification',size=fonts)
ax1.plot(kfolds,meanAcc[0,:],color='b',marker='o',label='noun')
ax1.plot(kfolds,meanAcc[1,:],color='r',marker='o',label='adjective')
# ax1.plot(np.arange(2,6),meanAcc[2,:4],color='y',marker='o',label='all words')
ax1.legend(fancybox=True,loc='best')
ax1.set_xlabel('k-fold',size=fonts)
ax1.set_ylabel('Average accuracy',size=fonts)


pl.show()