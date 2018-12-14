
# coding: utf-8

# In[ ]:


import os
# import nltk
import json
import time
import random
import datetime 
import numpy as np
import pandas as pd
# from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
# from nltk.stem.lancaster import LancasterStemmer
# stemmer = LancasterStemmer()
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_csv('svm_smo_wildlife_project.csv')['title'].values
j=pd.read_json('653labels.json')
jf=j['title'].values
labels=j['label'].values
labels=[-1 if i ==0 else 1 for i in labels ]
#dat=[*jf, *data]
dat=list(data)+list(jf)

# In[ ]:


count_vect = CountVectorizer(stop_words='english', lowercase=True)
tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
X_train = count_vect.fit_transform(dat)
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
x_train=X_train_tfidf.toarray()
y=labels


# In[ ]:


def al_svm(X=x_train, Y=y, C=100, G=0.1, K='linear', N=100, s=20000):
    x_train, x_test, y_train, y_test = train_test_split(X[:653], Y, test_size=0.25, random_state=42)
    #x_train = X[:500]
    #y_train = Y[:500]
    #x_test=X[500:653]
    #y_test=Y[500:653]
    unlabeled_x=X[653:s]
    ind=[nn for nn in range(s-653)]
    
    score=[]
    for i in range(int(len(unlabeled_x)/N)):
        t=time.time()
        inn=[nn for nn in range(len(x_train))]
        clf = SVC(C=200, kernel=K, probability=True)
        clf.fit(x_train,y_train)
#         r = unlabeled_x[np.argmin(np.abs(clf.decision_function(unlabeled_x)))] 
        indices=np.argsort(np.abs(clf.decision_function(unlabeled_x)), axis=0)[:N]
        r= unlabeled_x[indices]
#         r=random.sample(range(0,len(unlabeled_x)), N)
#         p=clf.predict(r.reshape(1,-1))
        p=clf.predict(r)
        score.append(clf.score(x_test, y_test))
        print(i,score[i])
#         x_train=np.append(x_train, unlabeled_x[r], axis = 0)
#         x_train=np.append(x_train, r.reshape(1,-1), axis = 0)
#         x_train+=r
        x_train=X[inn+list(np.asarray(indices)+653)]
        y_train=np.append(y_train, p, axis = 0) 
        unlabeled_x=np.delete(unlabeled_x, r, axis = 0)
#         unlabeled_x=np.delete(unlabeled_x, r, axis = 0)
#         print(time.time()-t)
    return score


# In[ ]:

cs=[10, 50, 100]
batches = [10, 50, 100, 200, 500]
scores=[]
for c in cs:
	temp=[]
	for batch in batches:
		tt=time.time()
		score=al_svm(X=x_train, Y=y, C=100, G=0.1, K='linear', N=200, s=10000)
		temp.append(score)
		print(time.time()-tt)
	scores.append(temp)


# In[ ]:

for i in range(5):
	plt.plot(scores[0][i])
plt.ylim(0.3, 1)
plt.savefig('score2_200_100_1000.png')
plt.show()


# In[ ]:


#plt.plot(score)
#plt.ylim(0.5, 1)
#plt.savefig('score2_200_100_1000.png')
#plt.show()


# In[ ]:


with open('score_200_100_1000.json', 'w') as f:
    f.write(json.dumps(score))

