
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

data=np.loadtxt('haberman1.txt')


# In[3]:

fp=open('haberman1.txt')


# In[4]:

data=np.loadtxt(fp)


# In[5]:

data=np.loadtxt('haberman1.txt')


# In[6]:

data


# In[7]:

X=data[:,[0,1,2]]
y=data[:,3]


# In[8]:

X


# In[9]:

from sklearn.cross_validation import train_test_split


# In[10]:

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[11]:

from sklearn.linear_model import LogisticRegression


# In[12]:

model_reg=LogisticRegression()
model_reg=model_reg.fit(X_train,y_train)


# In[13]:

model_reg.score(X_train,y_train)


# In[14]:

model_reg.score(X_test,y_test)


# In[15]:

from sklearn import svm
model_svm=svm.SVC()
model_svm=model_svm.fit(X_train,y_train)


# In[16]:

model_svm.score(X_train.y_train)


# In[17]:

model_svm.score(X_train,y_train)


# In[18]:

model_svm.score(X_test,y_test)


# In[19]:

from sklearn.tree import DecisonTreeClassifier
model_tree=model_tree.DecisionTreeClassifier()
model_tree=model_tree.fit(X_train,y_train)


# In[20]:

from sklearn.tree import DecisionTreeClassifier


# In[21]:

model_tree=model_tree.DecisionTreeClassifier()
model_tree=model_tree.fit(X_train,y_train)


# In[22]:

model_tree=DecisionTreeClassifier()
model_tree=model_tree.fit(X_train,y_train)


# In[23]:

model_tree.score(X_train,y_train)


# In[24]:

model_tree.score(X_test,y_test)


# In[25]:

model_tree=DecisionTreeClassifier(max_depth=3)
model_tree=model_tree.fit(X_train,y_train)


# In[26]:

model_tree.score(X_train,y_train)


# In[27]:

model_tree.score(X_test,y_test)


# In[28]:

model_tree=DecisionTreeClassifier(max_depth=4)
model_tree=model_tree.fit(X_train,y_train)
model_tree.score(X_train,y_train)


# In[29]:

model_tree.score(X_test,y_test)


# In[30]:

model_tree=DecisionTreeClassifier(max_depth=2)
model_tree=model_tree.fit(X_train,y_train)
model_tree.score(X_train,y_train)


# In[31]:

model_tree.score(X_test,y_test)


# In[ ]:



