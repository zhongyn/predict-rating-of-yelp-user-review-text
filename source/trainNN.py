import numpy as np
import navieBayes as nb


# Size of label 
groups = 0
with open("../data/group.txt",'r') as f:
	for line in f:
		groups += 1

# Size of vocabulary
voca = 0
with open("../data/vocabularyNN.txt",'r') as f:
	for line in f:
		voca += 1

# Load training data and label
data = np.loadtxt("../data/dataNN.txt", delimiter=' ', dtype=int)
label = np.loadtxt("../data/labelNN.txt", delimiter=' ', dtype=int)


folds = 11
predictAccCV = []

for k in range(2,folds+1):
	predictAcc = nb.navieBayesMulCV(groups, voca, data, label, k)
	predictAccCV.append(predictAcc)
	print predictAcc

accuracy = np.array(predictAccCV)
np.save('accNN.npy', accuracy)

