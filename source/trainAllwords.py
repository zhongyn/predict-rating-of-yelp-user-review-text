import numpy as np
import navieBayes as nb


# Size of label 
groups = 0
with open("../data/group.txt",'r') as f:
	for line in f:
		groups += 1

# Size of vocabulary
voca = 0
with open("../data/vocabulary.txt",'r') as f:
	for line in f:
		voca += 1

# Load training data and label
data = np.loadtxt("../data/data.txt", delimiter=' ', dtype=int)
label = np.loadtxt("../data/label.txt", delimiter=' ', dtype=int)


folds = 100
predictAccCV = []

for k in range(folds,folds+1):
	predictAcc = nb.navieBayesMulCV(groups, voca, data, label, k)
	predictAccCV.append(predictAcc)
	print predictAcc

accuracy = np.array(predictAccCV)
np.save('accAllWords.npy', accuracy)

