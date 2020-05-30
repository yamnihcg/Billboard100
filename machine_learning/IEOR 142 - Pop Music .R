library(NLP)
library(tm)
library(SnowballC)
library(wordcloud)
library(MASS)
library(caTools)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(corpus)
library(qdapRegex)
library(gbm)
library(boot)
library(dplyr)
library(ggplot2)
library(gplots)
library(GGally)
library(caTools)
library(car)
library(ROCR)
library(pROC)
library(mlbench)
library(caretEnsemble)
library(glmnet)

# Loading Data

songs = read.csv("/Users/yamnihcg/AllPopSongsFinal.csv", stringsAsFactors=FALSE)

all_lyrics = songs$lyrics
top100 = songs$Billboard100

songs$lyrics = NULL
songs$Billboard100 = NULL

songs$keyType = as.factor(songs$Key)
songs$modeType = as.factor(songs$Mode)
songs$timeSignature = as.factor(songs$Time_Signature)
songs$Explicit = as.factor(songs$explicit)
songs$genre = as.factor(songs$Genre)

songs$Danceability = scale(songs$Danceability)
songs$Energy = scale(songs$Energy)
songs$Loudness = scale(songs$Loudness)
songs$Speechiness = scale(songs$Speechiness)
songs$Acousticness = scale(songs$Acousticness)
songs$Instrumentalness = scale(songs$Instrumentalness)
songs$Liveness = scale(songs$Liveness)
songs$Valence = scale(songs$Valence)
songs$Tempo = scale(songs$Tempo)
songs$Duration = scale(songs$Duration)

songs$ID = NULL
songs$Song = NULL
songs$Artist = NULL
songs$Billboard100 = NULL
songs$URI = NULL
songs$Key = NULL
songs$Mode = NULL
songs$Time_Signature = NULL
songs$explicit = NULL
songs$Genre = NULL

# Convert every categorical variable to numerical using dummy variable
dmy <- dummyVars(" ~ .", data=songs, fullRank = TRUE)
songs_transformed <- data.frame(predict(dmy, newdata=songs))

# Add back lyrics and dependent variable
songs_transformed$Lyrics = all_lyrics
songs_transformed$Top100 = as.factor((top100))


# Distribution of Billboard and Non-Billboard Songs in dataset

table(songs_transformed$Top100)

# Number of Top 100 Billboard songs - 2117
# Number of not Top 100 Billboard songs - 5901


# NLP: Bag of Words

write.csv(songs_w_nlp, 'all_data_w_features_nlp_2010decade.csv')


# Building ML Models

# helper functions to calculate accuracy, TPR, and FPR

tableAccuracy <- function(test, pred) {
  t = table(test, pred)
  a = sum(diag(t))/length(test)
  return(a)
}

tableTPR <- function(label, pred) {
  t = table(label, pred)
  return(t[2,2]/(t[2,1] + t[2,2]))
}

tableFPR <- function(label, pred) {
  t = table(label, pred)
  return(t[1,2]/(t[1,1] + t[1,2]))
}


# Train/Test Split

spl = sample.split(songs_w_nlp$Top100, SplitRatio = 0.70)
train = songs_w_nlp %>% filter(spl == TRUE)
test = songs_w_nlp %>% filter(spl == FALSE)

levels(train$Top100) <- make.names(levels(factor(train$Top100)))
levels(test$Top100) <- make.names(levels(factor(test$Top100)))

# make model matrices
train.mm = as.data.frame(model.matrix(Top100 ~ . + 0, data=train))
test.mm = as.data.frame(model.matrix(Top100 ~ . + 0, data=test)) 


# Model 0: Baseline Model

table(train$Top100) # most frequent is "Not Top 100"
table(test$Top100) # baseline: predict always "Not Top 100"
baseline_accuracy = 1770/(1770+635) # 0.7359667

# Model 1: Logistic Regression

set.seed(123)

log_model = glm(Top100 ~ ., data = train, family = "binomial")
vif(log_model) # ALL VIF scores are low and <= 5 (meaning no multicollinearity) 
summary(log_model)

predict_log = predict(log_model, newdata = test, type = "response")
table(test$Top100, predict_log > 0.5)
tableAccuracy(test$Top100, predict_log > 0.5) 
tableTPR(test$Top100, predict_log > 0.5) 
tableFPR(test$Top100, predict_log > 0.5)

# Calculate ROC curve
rocr.log.pred <- prediction(predict_log, test$Top100)
logPerformance <- performance(rocr.log.pred, "tpr", "fpr")
plot(logPerformance, colorize = TRUE)
abline(0, 1)

as.numeric(performance(rocr.log.pred, "auc")@y.values) # AUC = 0.8542106


# Model 2: LDA

set.seed(123)

lda_model = lda(Top100 ~ ., data = train, family = "binomial")

predict_lda = predict(lda_model, newdata=test)
predict_lda$class[1:10]
predict_lda$posterior[1:10, ] 
predict_lda_probs <- predict_lda$posterior[,2]

table(test$Top100, predict_lda_probs > 0.5)
tableAccuracy(test$Top100, predict_lda_probs > 0.5) # 0.8024948
tableTPR(test$Top100, predict_lda_probs > 0.5) # 0.4897638
tableFPR(test$Top100, predict_lda_probs > 0.5) # 0.08531073

# Calculate ROC curve
rocr.lda.pred <- prediction(predict_lda_probs, test$Top100)
ldaPerformance <- performance(rocr.lda.pred, "tpr", "fpr")
plot(ldaPerformance, colorize = TRUE)
abline(0, 1)

as.numeric(performance(rocr.lda.pred, "auc")@y.values) # AUC = 0.8525477


# Model 3: 10-CV CART

set.seed(123)

train_cart = train(Top100 ~ .,
                   data = train,
                   method = "rpart",
                   metric = 'Accuracy',
                   tuneGrid = data.frame(cp=seq(0, 0.2, 0.002)),
                   trControl = trainControl(method="cv", number=10))

cart_model = train_cart$finalModel
prp(cart_model) # CART visualization

predict_cart = predict(cart_model, newdata = test, type='class')
table(test$Top100, predict_cart)
tableAccuracy(test$Top100, predict_cart) 
tableTPR(test$Top100, predict_cart) 
tableFPR(test$Top100, predict_cart) 

#Calculate ROC curve
cart_probs = predict(cart_model, newdata = test, type='prob')
rocCurve.cart <- roc(test$Top100, cart_probs[,"X1"])
#plot the ROC curve
plot(rocCurve.cart, col=c(4))
auc(rocCurve.cart) # 0.8043


# Model 3.5 10-CV CART Tuned for ROC

set.seed(123)

control <- trainControl(method="cv", number=10, classProbs=TRUE, summaryFunction=twoClassSummary)

train_cart_ROC = train(Top100 ~ .,
                       data = train,
                       method = 'rpart',
                       metric = 'ROC',
                       trControl = control
)

cart_ROC_model = train_cart_ROC$finalModel
prp(cart_ROC_model)

predict_cart_ROC = predict(cart_ROC_model, newdata = test, type='class')
table(test$Top100, predict_cart_ROC)
tableAccuracy(test$Top100, predict_cart_ROC) 
tableTPR(test$Top100, predict_cart_ROC) 
tableFPR(test$Top100, predict_cart_ROC) 

#Calculate ROC curve
cartROC_probs = predict(cart_ROC_model, newdata = test, type='prob')
rocCurve.cartROC <- roc(test$Top100, cartROC_probs[,"X1"])
#plot the ROC curve
plot(rocCurve.cartROC, col=c(4))
auc(rocCurve.cartROC) # 0.7883


# Model 4: Random Forest (default)

set.seed(123)

# (Takes about 3 min to run)
rf_model = randomForest(Top100 ~ ., data = train)  

predict_rf = predict(rf_model, newdata = test)
table(test$Top100, predict_rf)
tableAccuracy(test$Top100, predict_rf) # 0.8141372
tableTPR(test$Top100, predict_rf) # 0.4976378
tableFPR(test$Top100, predict_rf) # 0.07231638

#Calculate ROC curve
rf_probs = predict(rf_model, newdata = test, type='prob')
rocCurve.rf <- roc(test$Top100, rf_probs[,"X1"])
#plot the ROC curve
plot(rocCurve.rf, col=c(4))
auc(rocCurve.rf) # 0.8539


# Model 5: Bagging

set.seed(123)

bag_model <- randomForest(x = train.mm, y = train$Top100, mtry = 5, nodesize = 5, ntree = 500)

predict_bag <- predict(bag_model, newdata = test.mm)
table(test$Top100, predict_bag)
tableAccuracy(test$Top100, predict_bag) # 0.8199584
tableTPR(test$Top100, predict_bag) # 0.5637795
tableFPR(test$Top100, predict_bag) # 0.08813559

#Calculate ROC curve
bagging_probs = predict(bag_model, newdata = test, type='prob')
rocCurve.bagging <- roc(test$Top100, bagging_probs[,"X1"])
#plot the ROC curve
plot(rocCurve.bagging, col=c(4))
auc(rocCurve.bagging) # 0.8658


# Model 6: 10-CV Boosting Tuned for ROC

set.seed(123)

objControl <- trainControl(method="cv", number=10, classProbs=TRUE, summaryFunction=twoClassSummary)

train_boost <- train(Top100 ~ .,
                     data = train,
                     method = "gbm",
                     trControl = objControl,
                     metric = "ROC",
                     distribution = "bernoulli")

boost_model = train_boost$finalModel
predict_boost = predict(boost_model, newdata = test.mm, n.trees = boost_model$n.trees, type = "response")
table(test$Top100, predict_boost < 0.5) # Note: probabilities are flipped in gbm
tableAccuracy(test$Top100, predict_boost < 0.5) # 0.8261954
tableTPR(test$Top100, predict_boost < 0.5) # 0.5511811
tableFPR(test$Top100, predict_boost < 0.5) # 0.07514124

rocr.boost.pred <- prediction(predict_boost, test$Top100)
boostPerformance <- performance(rocr.boost.pred, "fpr", "tpr")
plot(boostPerformance, colorize = TRUE)
abline(0, 1)

as.numeric(performance(rocr.boost.pred, "auc")@y.values) # AUC = 1-0.1097891 = 0.8902109

# NOTE: Re-label axis titles

# Model 7: 10-CV KNN 

set.seed(123)

objControl <- trainControl(method="cv", number=10, classProbs=TRUE, summaryFunction=twoClassSummary)

train_knn <- train(Top100 ~ .,
                   data = train,
                   method = "knn",
                   trControl = objControl, 
                   tuneLength = 20)

knnPredict <- predict(train_knn, newdata = test)
table(test$Top100, knnPredict)
tableAccuracy(test$Top100, knnPredict) # 0.789605
tableTPR(test$Top100, knnPredict) # 0.2787402
tableFPR(test$Top100, knnPredict) # 0.02711864

# Calculate ROC curve
knn_probs = predict(train_knn, newdata = test, type='prob')
rocCurve.knn <- roc(test$Top100, knn_probs[,"X1"])
#plot the ROC curve
plot(rocCurve.knn, col=c(4))
auc(rocCurve.knn) # 0.855

# Model 8: Stacking: Logistic + CART + LDA

set.seed(123)

# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm')

models <- caretList(Top100 ~ ., data=train, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

# correlation between results
modelCor(results)
splom(results)

# stack using glm
set.seed(123)
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

predict_stack = predict(stack.glm, newdata = test, type = "prob")
table(test$Top100, predict_stack > 0.5)
tableAccuracy(test$Top100, predict_stack > 0.5) # 0.816632
tableTPR(test$Top100, predict_stack > 0.5) # 0.5228346
tableFPR(test$Top100, predict_stack > 0.5) # 0.0779661

# Calculate ROC curve
rocr.stack.pred <- prediction(predict_stack, test$Top100)
stackPerformance <- performance(rocr.stack.pred, "tpr", "fpr")
plot(logPerformance, colorize = TRUE)
abline(0, 1)

as.numeric(performance(rocr.stack.pred, "auc")@y.values) # AUC = 0.8609823


# Model 9: Ridge Regression

set.seed(123)

# convert training data to matrix format
x <- model.matrix(Top100 ~ ., train)
# convert class to numerical variable
y <- ifelse(train$Top100=='X1',1,0)

# perform grid search to find optimal value of lambda
# family = binomial => logistic regression, alpha = 0 => ridge
# check docs to explore other type.measure options
cv.out <- cv.glmnet(x, y, alpha = 0, family = 'binomial', type.measure = 'mse')
#plot result
plot(cv.out)

# min value of lambda
lambda_min <- cv.out$lambda.min
# best value of lambda
lambda_1se <- cv.out$lambda.1se
# regression coefficients
coef(cv.out,s=lambda_1se)

# get test data
x_test <- model.matrix(Top100 ~ ., test)
# predict class, type=”class”
ridge_prob <- predict(cv.out, newx = x_test, s = lambda_1se, type = 'response')
#translate probabilities to predictions
ridge_predict <- rep('X0', nrow(test))
ridge_predict[ridge_prob>.5] <- 'X1'

#confusion matrix
table(pred=ridge_predict,true=test$Top100)
tableAccuracy(test$Top100, ridge_predict) # 0.7987526
tableTPR(test$Top100, ridge_predict) # 0.4
tableFPR(test$Top100, ridge_predict) # 0.05819209

# Model 10: Lasso Regression

set.seed(123)

# convert training data to matrix format
x <- model.matrix(Top100 ~ ., train)
# convert class to numerical variable
y <- ifelse(train$Top100=='X1',1,0)

# perform grid search to find optimal value of lambda
# family = binomial => logistic regression, alpha = 1 => lasso
# check docs to explore other type.measure options
cv.out <- cv.glmnet(x, y, alpha = 1, family = 'binomial', type.measure = 'mse')
#plot result
plot(cv.out)
# min value of lambda
lambda_min <- cv.out$lambda.min
# best value of lambda
lambda_1se <- cv.out$lambda.1se
# regression coefficients
coef(cv.out,s=lambda_1se)

# get test data
x_test <- model.matrix(Top100 ~ ., test)
# predict class, type=”class”
lasso_prob <- predict(cv.out, newx = x_test, s = lambda_1se, type = 'response')
#translate probabilities to predictions
lasso_predict <- rep('neg', nrow(test))
lasso_predict[lasso_prob>.5] <- 'X1'

#confusion matrix
table(pred=lasso_predict,true=test$Top100)
tableAccuracy(test$Top100, lasso_predict) # 0.7950104
tableTPR(test$Top100, lasso_predict) # 0.4125984
tableFPR(test$Top100, lasso_predict) # 0.06779661

