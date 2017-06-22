import numpy as np
#loading the data to numpy array
data = np.loadtxt("haberman1.txt")
X = data[:,0:3]
y = data[:,3]
print X
print y
from sklearn.cross_validation import train_test_split
#splitting the data into training and test data
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state=40)
#Applying Logistic Regression on our data
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression()
model_log = model_log.fit(X_train,y_train)
print "Train Accuracy for Logistic Regression Model",model_log.score(X_train,y_train)
print "Test Accuracy for Logistic Regression Model",model_log.score(X_test,y_test)
#Applying SVM on our data
from sklearn import svm
model_svm = svm.SVC()
model_svm = model_svm.fit(X_train,y_train)
print "Train Accuracy for SVM Model",model_svm.score(X_train,y_train)
print "Test Accuracy for SVM Model",model_svm.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()
model_tree = model_tree.fit(X_train,y_train)
print "Train Accuracy for Decision Tree Model",model_tree.score(X_train,y_train)
print "Test Accuracy for Decison Tree Model",model_tree.score(X_test,y_test)
model_tree2 = DecisionTreeClassifier(max_depth=3)
model_tree2 = model_tree2.fit(X_train,y_train)
print "Train Accuracy for Decision Tree2 Model",model_tree2.score(X_train,y_train)
print "Test Accuracy for Decison Tree2 Model",model_tree2.score(X_test,y_test)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print "precision of log Model", precision_score(y_test,model_log.predict(X_test))
print "recall of log Model", recall_score(y_test,model_log.predict(X_test))
print "f1score of log Model", f1_score(y_test,model_log.predict(X_test))
print "precision of svm Model", precision_score(y_test,model_svm.predict(X_test))
print "recall of svm Model", recall_score(y_test,model_svm.predict(X_test))
print "f1score of svm Model", f1_score(y_test,model_svm.predict(X_test))
print "precision of tree Model", precision_score(y_test,model_tree.predict(X_test))
print "recall of tree Model", recall_score(y_test,model_tree.predict(X_test))
print "f1score of tree Model", f1_score(y_test,model_tree.predict(X_test))
print "precision of tree2 Model", precision_score(y_test,model_tree2.predict(X_test))
print "recall of tree2 Model", recall_score(y_test,model_tree2.predict(X_test))
print "f1score of tree2 Model", f1_score(y_test,model_tree2.predict(X_test))
print "Best model for our data is LogisticRegression"