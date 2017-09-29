#Reading the data
set.seed(0)

                          #Building Logistic Regression Model

#Sigmoid Function
sigmoid <- function(z){
  1/(1+exp(-z))
}

#Loss Function
loss_func <- function(y,h,W,lamda,m=60){
  (-1/m)*(sum(y*log(h) + (1-y)*log(1-h))) + (lamda/(2*m))*sum(W^2)
}
logistic_model <- function(X,y){
  sample = sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = F)
  X_train = X[sample,]
  X_test = X[-sample,]
  m = dim(X)[1]
  n = dim(X)[2]
  # Initializing Weights and bia
  W = array(0,c(n,1))
  b = 0
  
  #Gradient Descent
  lr = 0.01
  lamda = 0.5
  losses = c()
  accuracies = c()
  for (i in 1:1000){
    z = X_train %*% W + b
    h = sigmoid(z)
    W = W - (lr/m)*((t(X_train) %*% (h - y_train)) + lamda*W)
    b = b - (lr/m)*sum(h - y_train)
    if (i%%10 == 0){
      loss = loss_func(y_train,h,W,lamda)
      losses = c(losses,loss)
    }
  }
  plot(losses,typ='l')
  preds = sigmoid(X_test %*% W + b)
  preds[preds > 0.5] = 1
  preds[preds <= 0.5] = 0
  test_accuracy = sum(preds==y_test)/length(y_test)
  print(test_accuracy)
}
#Reading the data

#SBM Data
X1 <- read.csv('data/train_SBM.csv')
X1 <- as.matrix(X1)
m = dim(X1)[1]
n = dim(X1)[2]-1
X1 = X1[1:m,2:(n+1)]
colnames(X1) <- NULL
y = read.csv('data/train_labels.csv')
y = array(y[1:m,2],c(m,1))

#FNC Data
X2 <- read.csv('data/train_FNC.csv')
X2 <- as.matrix(X2)
m = dim(X2)[1]
n = dim(X2)[2]-1
X2 = X2[1:m,2:(n+1)]
colnames(X2) <- NULL

#Taking both SBM and FNC data
X_full <- cbind(X1,X2)
m = dim(X)[1]
n = dim(X)[2]
colnames(X) <- NULL

logistic_model(X1,y)
logistic_model(X2,y)
logistic_model(X_full,y)
