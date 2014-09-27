import json

chres = open('../data/chinese_restaurants.json')
review = open('../data/yelp_academic_dataset_review.json')
chres_review = open('../data/chres_review.json','w+')

ids = []
for cline in chres:
    ids.append(json.loads(cline)['business_id'])

print ids

i=0
for rline in review:
    jreview = json.loads(rline)
    rid = jreview['business_id']
    if rid in ids:
        i+=1
        chres_review.write(json.dumps(jreview)+'\n')

print i
chres.close()
review.close()
chres_review.close()



    
    
                
