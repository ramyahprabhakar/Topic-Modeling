# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:59:36 2019

@author: pingr
"""

#build a counter to know progress as large datasets load
count = 0

#open json file - 'r' means read, bc we don't want to write over the file
loadedjson = open('meta_Clothing_Shoes_and_Jewelry.json', 'r')

#this will be our final dataset for all the products
myproducts = {}
#all of the keys are product categories, and value-pair is all of the times this key appears in our data
listofcategories = {}

#go through reviews one line at a time
for line in loadedjson:
    count += 1
    if count % 100000 == 0:
        print(count)
    myproduct = eval(line)
    
#need to set a key for the best unique identifier (ASIN)
    myproducts[myproduct['asin']] = myproduct
#want to put categories inside this product into the list
#looks inside product data - we need to parse through these layers of categories
    for categories in myproduct['categories']:
        for acategory in categories:
            if acategory in listofcategories:
                listofcategories[acategory] += 1
            if acategory not in listofcategories:
                listofcategories[acategory] = 1
                
count = 0

#create new variable to store the ASINs
alltimberlandasins = set()

#set not list b/c it doesn't store duplicates
for myproduct in myproducts:
    theproduct = myproducts[myproduct]
#this avoids the messiness of reaching into layers each time
#write dictionary entry as its own variable
    count += 1
    if count % 100000 == 0:
        print(count/1503384)
#counter will now tell you how far you are through the dataset
    for categories in theproduct['categories']:
        for acategory in categories:
            if 'timberland' in acategory.lower():
                        
#force string to be lower case to remove capitalization issue
#if timberland is in this cateogory, then I know it's a Timberland category
                alltimberlandasins.add(theproduct['asin'])
                
#add to sets, append to lists, dictionaries keys are defined
#cannot inspect sets in Spyder, but you can print out to view
                
#Let's write the ASINs out to a file so we can 
#use them in the next segment to extract product reviews.

outputfile = open('allasins.txt', 'w')
outputfile.write(','.join(alltimberlandasins))

outputfile.close()

#open the reviews
loadedjson = open('reviews_Clothing_Shoes_and_Jewelry.json', 'r')

allreviews = {}
count = 0

for aline in loadedjson:
    count += 1
    if count % 100000 == 0:
        print(count)
    areview = eval(aline) # evaluates a line and treats it like a python entity (ex: dictionary)
    theasin = areview['asin']
    thereviewer = areview['reviewerID']
    if theasin in alltimberlandasins:
        thekey = '%s,%s' %(theasin,thereviewer) # make a unique key by using the asin and the reviewer ID. This is unique since one user can review a product only once
        allreviews[thekey] = areview
        
len(allreviews)

import json

json.dump(allreviews, open('alltimberlandreviews.json','w'))    
allreviews = json.load(open('alltimberlandreviews.json','r'))

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

stop_words = stopwords.words('english') # list of 153 common words that don't mean much for a topic
stop_words.append('timberland')

texts = set()

#pre processing section

def load_texts(topicdata): 
    for areview in topicdata:
        if 'reviewText' in topicdata[areview]:
            #print(topicdata[areview]['reviewText'])
            #sleep(1)
#            if 'overall' in topicdata[areview]:
#                if int(topicdata[areview]['overall'])<2:
#                    badreview = topicdata[areview]['reviewText'] # to extract bad reviews; play with the numbers to switch to good reviews
            reviewtext = topicdata[areview]['reviewText']
#            translatedtext = translator.translate(reviewtext)
#            sleep(1)
#            if translatedtext.src == "es":
#                reviewtext = translatedtext.text
            summary = topicdata[areview]['summary']
            asin = topicdata[areview]['asin']
            review = '%s %s %s' %(asin, summary, reviewtext)
            texts.add(review)

print('loading texts')
load_texts(allreviews)

#change the set texts into a list since NLTK likes lists
print("changing set texts to a list")
documents = list(texts) 

vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(documents) # matrix of data

true_k = 25

#model = KMeans(n_clusters= 50, max_iter=1000)
model = KMeans(n_clusters=true_k, max_iter=100000)
model.fit(X)

print("Top terms per cluster")

order_centroids = model.cluster_centers_.argsort()[:,::-1] # to get the key topics and terms associated with that
terms = vectorizer.get_feature_names()

outputfile = open('timberlandTopics.txt', 'w')

for i in range(true_k):
    topic_terms = [terms[ind] for ind in order_centroids[i,:4]]
    print('%d: %s' %(i, ' '.join(topic_terms)))
    outputfile.write('%d: %s \n' %(i, ' '.join(topic_terms)))
    
outputfile.close()

import os
outfiles={}

s=['']*len(allreviews)

try:
    os.mkdir('output')
    
except OSError:
    print('directory already exists')
    
else:
    print("successfully created the dictionary")
    
for atopic in range(true_k):
    topicterms=[terms[ind]for ind in order_centroids[atopic,:4]]
    outfiles[atopic]=open(os.path.join('output','_'.join(topicterms)+'.txt'),'w')

reviewtextonly = open('timberlandReviewsText.txt','w')
    
#Now add reviews that belong to each topic into the corresponding text file

for areview in allreviews:
    if 'reviewText' in allreviews[areview]:
        thereview = allreviews[areview]
        review = "%s %s %s" % (thereview['asin'], thereview['summary'], thereview['reviewText'])
        reviewtextonly.write("%s \n" %thereview['reviewText'])
        Y =  vectorizer.transform([review])
        predictions = model.predict(Y) # this line helps you see what each topic is
        
        # get scores for each document(review) for each of the topics and assign the review to the topic that has the highest tfidf score
        for prediction in model.predict(Y):
            outfiles[prediction].write('%s\n' % review)
            
for n, f in outfiles.items(): #f is the filename that was created
    f.close()

reviewtextonly.close()

#  to reduce the number of word overlap, reduce the number of topics, true_k
    
#LDA
#from sklearn.decomposition import LatentDirichletAllocation
#from multiprocessing import cpu_count 
## to count and use all the cores on the machine!!!
#
#k = 25
#
#lda_tfidf = LatentDirichletAllocation(n_topics = k, n_jobs = cpu_count())
#lda_tfidf.fit(X)
#
#import pyLDAvis        
##!pip install pyldavis
#import pyLDAvis.sklearn
#pyLDAvis.enable_notebook() # enables the python notebook functionality
#
#p = pyLDAvis.sklearn.prepare(lda_tfidf, X, vectorizer)
#pyLDAvis.save_html(p, 'pyLDAvis.html')
#
#tfidf_feature_names = vectorizer.get_feature_names()
#n_top_words = 5

#Vader for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

negativetexts =set()
positivetexts = set()

def load_neg_texts(topicdata):
    for areview in topicdata:
        if 'reviewText' in topicdata[areview]:  
            if analyser.polarity_scores(topicdata[areview]['reviewText'])['compound'] >= -0.30 and analyser.polarity_scores(topicdata[areview]['reviewText'])['compound'] <=-0.05 :
                reviewtext=topicdata[areview]['reviewText']
                summary=topicdata[areview]['summary']
                asin=topicdata[areview]['asin']
                review='%s %s %s' % (asin,summary,reviewtext)
                negativetexts.add(review)
                
load_neg_texts(allreviews)
negativedocuments = list(negativetexts) 

def load_pos_texts(topicdata):
    for areview in topicdata:
        if 'reviewText' in topicdata[areview]:  
            if analyser.polarity_scores(topicdata[areview]['reviewText'])['compound'] >= 0.5 :
                reviewtext=topicdata[areview]['reviewText']
                summary=topicdata[areview]['summary']
                asin=topicdata[areview]['asin']
                review='%s %s %s' % (asin,summary,reviewtext)
                positivetexts.add(review)

load_pos_texts(allreviews)
positivedocuments = list(positivetexts)






















