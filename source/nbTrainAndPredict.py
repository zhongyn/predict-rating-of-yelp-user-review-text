import numpy as np
import navieBayes as nb



folds = 10
predictAccCV = []
# filetype = 'Adj'
# filetype = 'NN'
filetype = 'AllWords'
groups = 5

for k in range(6,folds+1):
	predictAcc = nb.navieBayesMulCV(k, filetype, groups)
	predictAccCV.append(predictAcc)
	print "predictAcc: k",k,predictAcc

accuracy = np.array(predictAccCV)
np.save('acc'+filetype+'.npy', accuracy)

