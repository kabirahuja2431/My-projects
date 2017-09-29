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
m = dim(X_train)[1]
#Mean Normalization
means = colMeans(X_train)
X_train = sweep(X_train,2,means)
X_test = sweep(X_test,2,means)

                          #Building Logistic Regression Model

# Initializing Weights and bia
W = array(0,c(n,1))
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
losses = c()
accuracies = c()
for (i in 1:1000){
  z = X_train %*% W + b
  h = sigmoid(z)
  W = W - (lr/60)*((t(X_train) %*% (h - y_train)) + lamda*W)
  b = b - (lr/60)*sum(h - y_train)
  if (i%%10 == 0){
    loss = loss_func(y_train,h,W,lamda)
    losses = c(losses,loss)
    preds = sigmoid(X_test %*% W + b)
    preds[preds > 0.5] = 1
    preds[preds <=0.5] = 0
    accuracies = c(accuracies,sum(preds==y_test)/length(y_test))
  }
}
par(mfrow=c(1,2))
plot(losses,typ='l')

plot(accuracies,typ='l')

                                                      
