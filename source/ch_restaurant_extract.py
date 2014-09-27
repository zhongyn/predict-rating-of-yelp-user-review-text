
import json

i=0
with open('../data/chinese_restaurants.json','w+') as chres:
    with open('../data/yelp_academic_dataset_business.json') as business:
        for line in business:
               jbus=json.loads(line)
               jcate=jbus['categories']
               #if 'Chinese' in jdata and 'Restaurants' not in jdata:
               if 'Chinese' in jcate and 'Restaurants' in jcate:
                   i += jbus['review_count']                  
                   chres.write(json.dumps(jbus)+'\n')
                   #print jdata
        #print jdata  
    print "chinese_restaurant.json succefully created.",i
        

                
