import json
import textProcess as tp
import numpy as np


def extractRevsLabel(filetype, pattern):

	review = open('../data/chres_review.json')
	revsKW = open('../data/revsKW'+filetype+'.json','w+')

	revs = []
	lab = []

	i = 1
	if pattern=='AllWords':
		for index,line in enumerate(review):
		    jre = json.loads(line)
		    jstar = jre['stars']  
		    text = jre['text']  
		    ws = tp.removeStopPunc(text)
		    lab.append(jstar)
		    revs.append(ws)
		    print i
		    # if i==50: break
		    i += 1
	else:			
		for index,line in enumerate(review):
		    jre = json.loads(line)
		    jstar = jre['stars']  
		    text = jre['text']  

		    tagText = tp.posTag(text)
		    adj = tp.posExtract(tagText,pattern)
		    adjs = ' '.join(adj)
		    ws = tp.removeStopPunc(adjs)

		    lab.append(jstar)
		    revs.append(ws)
		    print i
		    # if i==50: break
		    i += 1

	np.save('../data/label'+filetype+'.npy', np.array(lab))
	json.dump(revs, revsKW)

	review.close()
	revsKW.close()
	return 1



def createDataVoca(kfold,run,filetype):

	label = np.load('../data/label'+filetype+'.npy')

	revs = []
	dat = []

	with open('../data/revsKW'+filetype+'.json') as revsKW:
		revs = json.load(revsKW)

	subsetSize = len(label)/kfold

	trainRevs = revs[:run*subsetSize]+revs[(run+1)*subsetSize:]
	vocabulary = [word for rev in trainRevs for word in rev]
	vocabulary = list(set(vocabulary))
	# print vocabulary
	# print revs

	voca = len(vocabulary)

	for revid, rev in enumerate(revs):
		dat.append({})
		for w in rev:
			if w in vocabulary:
				k = vocabulary.index(w)+1
				if k not in dat[revid]:
					dat[revid][k] = 1
				else:
					dat[revid][k] += 1
		# print revid+1
	print "len(revs):",len(revs)

	data=[]
	for revid, rev in enumerate(dat):
		for k,v in rev.iteritems():
			data.append([revid+1, k, v])

	allData = np.array(data)
	# print allData

	return [allData,voca,label]