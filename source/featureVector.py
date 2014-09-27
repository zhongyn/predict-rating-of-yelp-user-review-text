
import numpy as np

def featureCount(data, label, groups, voca):

	# Initial feature vectors of each label for Bernoulli and Multinomial models
	# berVectors = np.zeros((groups,voca))
	mulVectors = np.zeros((groups,voca))

	# Bernoulli: Count the number of docs containing word i for each label
	# Multinomial: Count the number of word i for each label
	for index, item in enumerate(data):
		# print item[0]-1
		lab = label[item[0]-1]
		word = item[1]
		# berVectors[lab-1, word-1] += 1
		mulVectors[lab-1, word-1] += item[2]

	return mulVectors


def berFeatGen(data, voca, numDocs):

	# Generate feature vector of each document
	# docID = np.unique(data[:,0])
	# numDocs = len(docID)
	berFeature = np.zeros((numDocs,voca))

	# Bernoulli: add the presence of word i for each doc
	# Multinomial: Count the number of word i for each doc
	for index, item in enumerate(data):
		docId = item[0]-1
		wordId = item[1]-1
		berFeature[docId, wordId] = 1

	return berFeature

def mulFeatGen(data, voca, numDocs):

	# Generate feature vector of each document
	mulFeature = np.zeros((numDocs,voca))
	# print 'numDocs: '+str(numDocs)
	
	# Bernoulli: add the presence of word i for each doc
	# Multinomial: Count the number of word i for each doc
	for index, item in enumerate(data):
		docId = item[0]-1
		wordId = item[1]-1
		mulFeature[docId, wordId] = item[2]

	return mulFeature

def confMatrix(trueLabel, predictLabel):

	# Generate a k by k confusion matrix
	trueL = np.unique(trueLabel)
	numL = len(trueL)
	kkmat =  np.zeros((numL,numL))

	for index, item in enumerate(predictLabel):
		row = trueLabel[index]
		kkmat[row-1,item-1] += 1

	return kkmat

def accPredict(trueLabel, predictLabel):

	# Compute the prediction accuracy
	numDocs = len(trueLabel)
	compL = trueLabel - predictLabel
	accuracy = 1.0*len(compL[compL==0])/numDocs
	return accuracy

