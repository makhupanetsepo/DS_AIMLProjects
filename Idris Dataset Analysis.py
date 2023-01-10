#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn.metrics import classification_report, accuracy_score
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[3]:


## Find a dataset that can be used for a classification problem
from sklearn.datasets import load_iris


# In[32]:


# load and preview the dataset
df_iris = load_iris()
df = pd.DataFrame(data = np.c_[df_iris['data'], df_iris['target']], columns=df_iris['feature_names'] + ['target'])
# Y = pd.DataFrame(df_iris.data, columns=df_iris.target_names)
df.head()


# In[42]:


# check for NaN values
pd.isna(df).sum()


# In[37]:


df_iris.target_names


# In[51]:


# add species names to the dataframe
species = []
for i in range(len(df['target'])):
    if int(df['target'][i]) == 0:
        species.append('setosa')
        
    elif int(df['target'][i]) == 1:
        species.append('versicolor')
        
    else:
        species.append('virginica')


# In[44]:


df_full = df


# In[46]:


df_full['species'] = species


# In[79]:


# new dataframe adding a new column; Species
df_full.head()


# In[80]:


# dataset summary
df_full.describe()


# In[65]:


df_full['target'] = df_full['target'].astype(int)


# In[85]:


df_full.head()


# In[91]:


from sklearn.tree import DecisionTreeClassifier
# create training and testing set
X = df_full[df_iris['feature_names']]
Y = df_full['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=0)


# ### <font color='black'>Based on petal and sepal length and width, can we predict the target? Let's use 2 different classification models</font>

# In[92]:


# 1st classification model
## create the classifier
clf_tree = DecisionTreeClassifier(random_state=0)

## train the classifier
clf_tree.fit(x_train, y_train)

## use the trained classifier to predict
y_pred = clf_tree.predict(x_test)


# In[95]:


## check the accuracy of the model
print('Model Accuracy: ', metrics.accuracy_score(y_test, y_pred))


# In[96]:


# 2nd classification model

## create the classifier
clf2 = MLPClassifier()

## train the classifier
clf2.fit(x_train, y_train)

## use the trained classifier to predict
y_pred_2 = clf2.predict(x_test)


# In[97]:


## check the 2nd model's accuracy
print('Model 2 Accuracy: ', metrics.accuracy_score(y_pred_2, y_test))


# In[ ]:




