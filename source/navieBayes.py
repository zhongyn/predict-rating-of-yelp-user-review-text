import numpy as np
import featureVector as fvec
import createDataVoca as dv


def navieBayesMulTrain(groups, voca, trainData, trainLabel):
	# Label list
	labels = np.arange(1,groups+2)
	# Num of docs for each label
	numDocsY = np.histogram(trainLabel, bins=labels)[0]
	# Learn p(y) using MLE for different labels
	probY = np.histogram(trainLabel, bins=labels, density=True)[0]

	# Bernoulli: Count the number of docs containing word i for each label
	# Multinomial: Count the number of word i for each label
	vectors = fvec.featureCount(trainData, trainLabel, groups, voca)
	# berVectors = vectors[0]
	mulVectors = vectors

	numDocs = len(trainLabel)

	# Learn Pi|y for i = 1,...,|V| using Laplace smoothing for both models
	# probIyBer = (berVectors+1)/(numDocsY[:,np.newaxis]+numDocs)
	probIyMul = (mulVectors+1)/(np.sum(mulVectors,axis=1)[:,np.newaxis]+voca)

	return [probY, probIyMul]



def navieBayesMulTest(groups, voca, testData, testLabel, probY, probIyMul):

	# Number of testing docs
	numDocs = len(testLabel)

	# Generate feature vector for each doc
	testMulFeat = fvec.mulFeatGen(testData, voca, numDocs)


	# Apply Navie Bayes classifier to test data
	# Multinomial: log(p(x|y)) = sum(xi*log(Pi|y)) + log(p(y))
	# p(y=label|x) ~ p(y=label)*p(x|y=label)

	probYXmul = np.zeros((numDocs,groups))
	for index, item in enumerate(testMulFeat):
		probYXmul[index,:] = np.sum(item*np.log(probIyMul),axis=1)+np.log(probY)

	# Find the best prediction label for each doc
	predictMul = np.argmax(probYXmul, axis=1)+1

	return predictMul




def navieBayesMulCV(k, filetype, groups):

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

		print 'len(trainLabel):',len(trainLabel)
		print 'len(testLabel):',len(testLabel)

		mask = np.logical_and((data[:,0]>i*subsetSize), (data[:,0]<=(i+1)*subsetSize))
		
		tem = np.array([i*subsetSize,0,0])
		testData = data[mask] - tem

		tem1 = np.array([subsetSize,0,0])
		maskhi = data[:,0]>(i+1)*subsetSize
		trainDatahi = data[maskhi] - tem1

		masklo = data[:,0]<=i*subsetSize
		trainDatalo = data[masklo]

		trainData = np.append(trainDatahi, trainDatalo, axis=0)
		
		print "we are training our data:"
		result = navieBayesMulTrain(groups, voca, trainData, trainLabel)
		print "we are testing our data:"
		predictMul = navieBayesMulTest(groups, voca, testData, testLabel,result[0], result[1])
		print "we are calculating accuracy:"
		predictAcc[i] = fvec.accPredict(testLabel,predictMul)

		# if i==0:
		# 	print "we are calculating the confusion matrix:"
		# 	kkmatMul = fvec.confMatrix(testLabel,predictMul)
		# 	np.savetxt('mulConfMat_k_'+str(k)+'_run_'+str(i)+'_'+filetype+'.txt', kkmatMul, fmt='%4d')


	# averageAcc = np.mean(predictAcc)

	return predictAcc


