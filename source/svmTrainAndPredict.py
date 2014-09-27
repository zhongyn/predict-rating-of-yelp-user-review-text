import numpy as np
import svm


filetype = 'Adj'
# filetype = 'NN'
# filetype = 'AllWords'
groups = 5

folds = 2
C = [1,10,100,1000]
kernel = ['linear','poly','rbf']
predictAccCV = np.zeros([4,3,folds])


for i,cp in enumerate(C):
	for j,ke in enumerate(kernel):
		print 'C:',cp
		print 'kernel:',ke				
		predictAccCV[i,j] = svm.svmCV(folds, filetype, groups, cp, ke)
		print 'predictAccCV:',predictAccCV

np.save('accSVM'+filetype+'.npy', predictAccCV)