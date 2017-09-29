library('caret')
set.seed(0)

#Function for the SVM model
#Takes the Data Frame as the arguement
svm_model <- function (data_fr,method){
  sample = sample.int(n = nrow(data_fr), size = floor(.6*nrow(data_fr)), replace = F)
  train_data = data_fr[sample,]
  test_data = data_fr[-sample,]
  print(dim(train_data))
  print(dim(test_data))
  train_data[,ncol(train_data)] = factor(train_data[,ncol(train_data)])
  #Building SVM Model
  ctrl = trainControl(method="repeatedcv",number=3,repeats=1)
  grid = expand.grid(C = c(0.01,0.05,0.1,0.25,0.4,0.45,0.5,0.55,0.6,0.75,0.9,1,1.25,1.5,1.75,2,5))
  linear_svm = train(x=train_data[,1:(ncol(train_data)-1)],y=train_data[,ncol(train_data)],method=method,
                     trControl=ctrl,preProcess = c("center", "scale"),
                     #tuneGrid = grid,
                     tuneLength = 10)
  test_preds = predict(linear_svm, newdata=test_data[,1:(ncol(test_data)-1)])
  print(test_preds)
  conf_mat = confusionMatrix(test_preds,test_data[,ncol(test_data)])
  
  print(linear_svm)
  print(conf_mat)
  
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
data_fr1 = cbind(X1,y)
data_fr1 = data.frame(data_fr1)

#FNC Data
X2 <- read.csv('data/train_FNC.csv')
X2 <- as.matrix(X2)
m = dim(X2)[1]
n = dim(X2)[2]-1
X2 = X2[1:m,2:(n+1)]
colnames(X2) <- NULL
data_fr2 = cbind(X2,y)
data_fr2 = data.frame(data_fr2)

#Taking both SBM and FNC data
X <- cbind(X1,X2)
m = dim(X)[1]
n = dim(X)[2]
colnames(X) <- NULL
data_frr = cbind(X,y)
data_frr = data.frame(data_frr)

print("Taking SBM features only")
svm_model(data_fr1,"svmPoly")

print("Taking FNC features only")
svm_model(data_fr2,"svmPoly")

print("Taking both features")
svm_model(data_frr,"svmPoly")
