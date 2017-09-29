#Reading the data
set.seed(0)
X1 <- read.csv('data/train_FNC.csv')
X1 <- as.matrix(X1)
m = dim(X1)[1]
n = dim(X1)[2]-1
X1 = X1[1:m,2:(n+1)]
X2 <- read.csv('data/train_SBM.csv')
X2 <- as.matrix(X2)
m = dim(X2)[1]
n = dim(X2)[2]-1
X2 = X2[1:m,2:(n+1)]
X <- cbind(X1,X2)
m = dim(X)[1]
n = dim(X)[2]
colnames(X) <- NULL
y = read.csv('data/train_labels.csv')
y = array(y[1:86,2],c(86,1))
#train_test split
sample = sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = F)
X_train = X[sample,]
X_test = X[-sample,]
y_train = y[sample,]
y_test = y[-sample,]
#Mean Normalization
means = colMeans(X_train)
X_train = sweep(X_train,2,means)
X_test = sweep(X_test,2,means)

m = dim(X_train)[1]
                              #Performing PCA
sigma = (t(X_train) %*% X_train)/m
a = svd(sigma)
d = unlist(a[1])
U = matrix(unlist(a[2]),n)

#visualizing data
k = 2
x_train = X_train %*% U[1:n,1:k]
frame = as.data.frame(cbind(x_train,y_train))
frame$y_train = factor(frame$y_train)

plot(frame$V1, frame$V2,
     bg=c("red","blue")[unclass(frame$y_train)],pch=21, 
     main="Visualizing First 2 principle components")

#Finding The Variance Retained by the principle components

total_var = sum(d)
vars = d/total_var
dev.new()
plot(vars,type='l')

#training the model by using 20-60 principle componenets and comparing the performance on test data
train_accuracies = c()
test_accuracies = c()
for (k in 10:60){
  x_train = X_train %*% U[1:n,1:k]
  x_test = X_test %*% U[1:n,1:k]
  # Initializing Weights and bia
  W = array(0,c(k,1))
  b = 0
  
  #Sigmoid Function
  sigmoid <- function(z){
    1/(1+exp(-z))
  }
  
  #Loss Function
  loss_func <- function(y,h,W,lamda,m=60){
    (-1/m)*(sum(y*log(h) + (1-y)*log(1-h))) + (lamda/(2*m))*sum(W^2)
  }
  
  #Gradient Descent
  lr = 0.01
  lamda = 0.5
  for (i in 1:1000){
    z = x_train %*% W + b
    h = sigmoid(z)
    W = W - (lr/60)*((t(x_train) %*% (h - y_train)) + lamda*W)
    b = b - (lr/60)*sum(h - y_train)
  }
  #accuracy on train and test sets
  preds_train = sigmoid(x_train%*%W+b)
  preds_train[preds_train>0.5] = 1
  preds_train[preds_train<=0.5] = 0
  train_accuracy = sum(preds_train==y_train)/m
  preds_test = sigmoid(x_test%*%W+b)
  preds_test[preds_test>0.5] = 1
  preds_test[preds_test<=0.5] = 0
  test_accuracy = sum(preds_test==y_test)/length(y_test)
  train_accuracies = c(train_accuracies,train_accuracy)
  test_accuracies = c(test_accuracies,test_accuracy)
  print(paste("For first",k,"Principle Components:"))
  print(paste("Train Accuracy",train_accuracy))
  print(paste("Test Accuracy",test_accuracy))
  
}
#Selecting the number of components which gave max accuracy on test set
k = which.max(test_accuracies) + 10 -1
