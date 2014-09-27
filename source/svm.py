import numpy as np
from sklearn import svm
import featureVector as fvec
from sklearn.externals import joblib
import createDataVoca as dv


def svmCV(k, filetype, groups, cp, ke):

	predictAcc = np.zeros(k)

	for i in range(k):

		print 'k:',k,'run:',i
		result = dv.createDataVoca(k,i,filetype)
		data = result[0]
		voca = result[1]
		label = result[2]

		subsetSize = len(label)/k
		testLabel = label[i*subsetSize:(i+1)*subsetSize]
		trainLabel = np.append(label[:i*subsetSize],label[(i+1)*subsetSize:], axis=0)

		lenTestLab = len(testLabel)
		lenTrainLab = len(trainLabel)

		print 'len(testLabel):',lenTestLab
		print 'len(trainLabel):',lenTrainLab

		mask = np.logical_and((data[:,0]>i*subsetSize), (data[:,0]<=(i+1)*subsetSize))
		
		tem = np.array([i*subsetSize,0,0])
		testData = data[mask] - tem

		tem1 = np.array([subsetSize,0,0])
		maskhi = data[:,0]>(i+1)*subsetSize
		trainDatahi = data[maskhi] - tem1

		masklo = data[:,0]<=i*subsetSize
		trainDatalo = data[masklo]

		trainData = np.append(trainDatahi, trainDatalo, axis=0)
		
		# generate features for all the reviews
		trainFeat = fvec.mulFeatGen(trainData, voca, lenTrainLab)
		testFeat = fvec.mulFeatGen(testData, voca, lenTestLab)

		clf = svm.SVC(C=cp, kernel=ke, cache_size=1000)
		print "we are training our data:"
		clf.fit(trainFeat, trainLabel)
		print "we are testing our data:"
		predictMul = clf.predict(testFeat)		
		print "we are calculating accuracy:"
		predictAcc[i] = fvec.accPredict(testLabel,predictMul)
		
		print "accuracy: "+str(predictAcc[i])
		joblib.dump(clf, '../modelSave/svmModelAll_K'+str(k)+'_Run'+str(i)+'_'+filetype+'.pkl') 
	# averageAcc = np.mean(predictAcc)

	return predictAcc
