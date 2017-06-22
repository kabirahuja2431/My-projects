import arff
import numpy as np
#reading the data into a numpy array dataset
dataset=arff.load(open("ThoraricSurgery.arff",'rb'))
data=np.array(dataset['data'])
#coverting each value of the array to a string
data=data.astype(str)

data1=np.zeros((470,28))
data1=data1.astype(str)
#Manipulating data to a suitable form to do operations on it.
for i in xrange(470):
    k=14
    for j in xrange(17):
        if data[i][j]=='DGN1':
            data1[i][0]=1
        elif data[i][j]=='DGN2':
            data1[i][1]=1
        elif data[i][j]=='DGN3':
            data1[i][2]=1
        elif data[i][j]=='DGN4':
            data1[i][3]=1
        elif data[i][j]=='DGN5':
            data1[i][4]=1
        elif data[i][j]=='DGN6':
            data1[i][5]=1
        elif data[i][j]=='DGN8':
            data1[i][6]=1
        elif data[i][j]=='PRZ0':
            data1[i][7]=1
        elif data[i][j]=='PRZ1':
            data1[i][8]=1
        elif data[i][j]=='PRZ2':
            data1[i][9]=1
        elif data[i][j]=='OC11':
            data1[i][10]=1
        elif data[i][j]=='OC12':
            data1[i][11]=1
        elif data[i][j]=='OC13':
            data1[i][12]=1
        elif data[i][j]=='OC14':
            data1[i][13]=1
        else:
            data1[i][k]=data[i][j]
            k+=1

for i in xrange(470):
    for j in xrange(28):
        if data1[i][j]=='F':
            data1[i][j]=0
        elif data1[i][j]=='T':
            data1[i][j]=1
            
data1=data1.astype(float)
#loading the dataset in a pandas dataframe
import pandas as pd
df = pd.DataFrame(data1)
print df
#changing the name of the colums
new_header = ["DGN1","DGN2","DGN3","DGN4","DGN5","DGN6","DGN8","PRZ0","PRZ1","PRZ2","OC11",
              "OC12","OC13","OC14","FVC","FEV1","PBS","HBS","DBS","CBS","WBS","DM","MI",
              "PAD","Smoking","Asthma","Age","Risk"]
df.columns = new_header
print df
from sklearn.cross_validation import train_test_split
X = data1[:,0:27]
y = data1[:,27]
#splitting the data into train(80%) and split(20%)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#Applying decision tree algorithm on our data(without setting minimum depth)
from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()
model_tree = model_tree.fit(X_train,y_train)
print "Train Accuracy for Decision tree model1",model_tree.score(X_train,y_train)
print "Test Accuracy for Decision tree model1",model_tree.score(X_test,y_test)
#Applying decision tree algorithm on our data with maximum depth of tree being 7
model_tree1 = DecisionTreeClassifier(max_depth=7)
model_tree1 = model_tree1.fit(X_train,y_train)
print "Train Accuracy for Decision tree model2",model_tree1.score(X_train,y_train)
print "Test Accuracy for Decision tree model2",model_tree1.score(X_test,y_test)
#Applying SVM on our data
from sklearn import svm
model_svm = svm.SVC()
model_svm = model_svm.fit(X_train,y_train)
print "Train Accuracy for svm model",model_svm.score(X_train,y_train)
print "Test Accuracy for svm model",model_svm.score(X_test,y_test)
#Applying Bernoulli Naive bayes algorithm on our data
from sklearn.naive_bayes import BernoulliNB
model_bnb = BernoulliNB()
model_bnb = model_bnb.fit(X_train,y_train)
print "Train Accuracy for bernoulli naive bayes model",model_bnb.score(X_train,y_train)
print "Test Accuracy for bernoulli naive bayes model",model_bnb.score(X_test,y_test)
#Applying Random Forest Algorithm on our data with maximum depth of each tree being 7
from sklearn.ensemble import RandomForestClassifier
model_forest = RandomForestClassifier(max_depth=7,random_state=42)
model_forest = model_forest.fit(X_train,y_train)
print "Train Accuracy for random forest model",model_forest.score(X_train,y_train)
print "Test Accuracy for random forest model",model_forest.score(X_test,y_test)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#Calculating precision,recall and F score of the decision tree model(one with the max depth 7) 
print "Precision based on Decision tree model2",precision_score(y_test,model_tree1.predict(X_test))
print "Recall based on Decision tree model2",recall_score(y_test,model_tree1.predict(X_test))
print "Fscore based on Decision tree model2",f1_score(y_test,model_tree1.predict(X_test))
#Calculating precision,recall and F score of the random forest model(one with the max depth 7) 
print "Precision based on RandomForestClassifier",precision_score(y_test,model_forest.predict(X_test))
print "Recall based on RandomForestClassifier",recall_score(y_test,model_forest.predict(X_test))
print "Fscore based on RandomForestClassifier",f1_score(y_test,model_forest.predict(X_test))