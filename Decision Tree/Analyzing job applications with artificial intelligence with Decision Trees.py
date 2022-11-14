#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("C:/Users/90507/Desktop/Dersler/DecisionTreesClassificationDataSet.csv")


# In[3]:


duzetme_mapping = {'Y': 1, 'N': 0}

df['IseAlindi'] = df['IseAlindi'].map(duzetme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzetme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzetme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzetme_mapping)
duzetme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzetme_mapping_egitim)
df.head()


# In[5]:


y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis=1)


# In[6]:


X.head()


# In[7]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# In[10]:


#5 years of experience, working anywhere and working in 3 old companies, undergraduate education level.
print (clf.predict([[5,1,3,0,0,0]]))


# In[12]:


#A total of two years of work experience, changed jobs 7 times, is a very good school graduate, but is not working now.

print (clf.predict([[2,0,7,0,1,0]]))


# In[ ]:




