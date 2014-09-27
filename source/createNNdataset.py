import json
import textProcess as tp


review = open('../data/chres_review.json')
vocabulary = open('../data/vocabularyNN.txt','w+')
label = open('../data/labelNN.txt','w+')
data = open('../data/dataNN.txt','w+')

voca = []
revs = []
lab = []
dat = []

i=1
for line in review:
    jre = json.loads(line)
    jstar = jre['stars']  
    text = jre['text']  

    tagText = tp.posTag(text)
    adj = tp.posExtract(tagText,'NN.*')
    adjs = ' '.join(adj)
    ws = tp.removeStopPunc(adjs)

    if ws!=[]:
	    lab.append(jstar)
	    revs.append(ws)
	    voca.extend(ws)
	    print i
	    # if i==5: break
	    i += 1


print lab
for i in lab:
	label.write(str(i)+"\n")
print "successfully create label!"


voca = list(set(voca))
print len(voca)
for i in voca:
	vocabulary.write(i.encode('utf8')+"\n")
print "successfully create vocabulary!"


for revid, rev in enumerate(revs):
	dat.append({})
	for w in rev:
		if w in voca:
			k = voca.index(w)+1
			if k not in dat[revid]:
				dat[revid][k] = 1
			else:
				dat[revid][k] += 1
	print revid+1
print len(revs)

for revid, rev in enumerate(dat):
	for k,v in rev.iteritems():
		s = str(revid+1)+' '+str(k)+' '+str(v)+'\n'
		data.write(s)
print "successfully create data!"


review.close()
vocabulary.close()
data.close()
label.close()