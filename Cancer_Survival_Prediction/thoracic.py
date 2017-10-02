
# coding: utf-8

# <h1> Loading Important Packages </h1>

# In[24]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier


# <h1> Loading Dataset in a DataFrame </h1>

# In[3]:

frame = pd.read_csv('thoracic_data.csv')
#frame


# <h2> Description of features </h2>
# <p>1. DGN: Diagnosis - specific combination of ICD-10 codes for primary and secondary as well multiple tumours if any (DGN3,DGN2,DGN4,DGN6,DGN5,DGN8,DGN1)<br> 
# 2. PRE4: Forced vital capacity - FVC (numeric)<br> 
# 3. PRE5: Volume that has been exhaled at the end of the first second of forced expiration - FEV1 (numeric)<br> 
# 4. PRE6: Performance status - Zubrod scale (PRZ2,PRZ1,PRZ0) <br>
# 5. PRE7: Pain before surgery (T,F) <br>
# 6. PRE8: Haemoptysis before surgery (T,F) <br> 
# 7. PRE9: Dyspnoea before surgery (T,F) <br>
# 8. PRE10: Cough before surgery (T,F) <br>
# 9. PRE11: Weakness before surgery (T,F) <br>
# 10. PRE14: T in clinical TNM - size of the original tumour, from OC11 (smallest) to OC14 (largest) (OC11,OC14,OC12,OC13) <br> 
# 11. PRE17: Type 2 DM - diabetes mellitus (T,F) <br>
# 12. PRE19: MI up to 6 months (T,F) <br>
# 13. PRE25: PAD - peripheral arterial diseases (T,F) <br>
# 14. PRE30: Smoking (T,F) <br>
# 15. PRE32: Asthma (T,F) <br>
# 16. AGE: Age at surgery (numeric) <br>
# 17. Risk1Y: 1 year survival period - (T)rue value if died (T,F)</p>

# <h1> Converting data into a numeric form to apply algorithms on it </h1>

# In[4]:

data_dict = frame.T.to_dict().values()
vec = DictVectorizer()
data = vec.fit_transform(data_dict).toarray()
X = data[:,0:len(data[0])-1]
y = data[:,len(data[0])-1]
print X
print y


# <h1> Splitting dataset into training and testing data </h1>
# 

# In[5]:

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# <h1> Applying SVM Classifier </h1>
# <h3> Fitting training data using SVM </h3>

# In[8]:

from sklearn import svm
model_svm = svm.SVC()
model_svm = model_svm.fit(X_train,y_train)


# <h3> Calculating accuracy of our model for train and test data </h3>

# In[10]:

print "Train Accuracy for svm model",model_svm.score(X_train,y_train)*100
print "Test Accuracy for svm model",model_svm.score(X_test,y_test)*100


# <h3> Finding Precision and Recall </h3>
# <p> <b>precision = true positives /(true positives + false positives)</b> <br>
#         Which means precision tells us out of the all the positives predicted by our model (i.e in our case Risk after the surgery), how many are actually positives. More the precision lesser the number of false negatives. Maximum value of precision is 1 <br>
#     <b>recall = true positives/(true positives + false negatives) </b> <br></p>
#     Which means recall tells us out of the the total positives in our data set how many of them are correctly predicted by our model. More the recall, More is the the number of correctly classified positives by our model. Maximum value of recall is also 1.</p>

# In[13]:

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "Precision based on SVM model",precision_score(y,model_svm.predict(X))*100
print "Recall based on SVM model",recall_score(y,model_svm.predict(X))*100


# <h1> Applying Decision Tree Classifier </h1>
# <h3> Fitting Training Data </h3>

# In[14]:

from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier(random_state = 42)
model_tree = model_tree.fit(X_train,y_train)


# <h3> Calculating accuracy of our model for train and test data </h3>

# In[16]:

print "Train Accuracy for Decision tree model1",model_tree.score(X_train,y_train)*100
print "Test Accuracy for Decision tree model1",model_tree.score(X_test,y_test)*100


# <h3> Finding Precision and Recall </h3>

# In[19]:

print "Precision based on Decision tree model",precision_score(y,model_tree.predict(X))*100
print "Recall based on Decision tree model",recall_score(y,model_tree.predict(X))*100


# <h2> Hence we can see that Decision Tree performs excellent on our dataset </h2>

# <h1> Visualising Dataset </h1>

# In[28]:

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
normalizer = Normalizer(copy=False)
data_norm = normalizer.fit_transform(X_train)
data_reduced = pca.fit_transform(data_norm)


colors = [i for i in y_train]
plt.scatter(data_reduced[:,0],data_reduced[:,1], s = 50 ,c = colors,label = "Blue: Safe, Red: Risk")
plt.title('Data Visualization')
plt.legend()
plt.show()


# In[ ]:



