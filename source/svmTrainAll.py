import numpy as np
import svm


# Size of vocabulary
voca = 0
with open("../data/vocabulary.txt",'r') as f:
	for line in f:
		voca += 1

# Load training data and label
data = np.loadtxt("../data/data.txt", delimiter=' ', dtype=int)
label = np.loadtxt("../data/label.txt", delimiter=' ', dtype=int)

folds = 2
predictAccCV = []

for k in range(2,folds+1):
	predictAcc = svm.svmCV(voca, data, label, k)
	predictAccCV.append(predictAcc)
	print predictAcc

accuracy = np.array(predictAccCV)
np.save('accAllsvm.npy', accuracy)

