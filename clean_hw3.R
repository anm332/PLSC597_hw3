# clean hw 3

#Load packages
library(ggplot2)
library(tidyverse)
library(corrplot)
library(apaTables)
library(psych)
library(tidyr)
library(RCurl)
library(mlr)
library(tidyverse)


# Read in the data ############################################################
clean_df <- getURL("https://raw.githubusercontent.com/anm332/thesisdata_anm332/main/clean_thesis_data_2021.csv")
clean_df <- read_csv(clean_df)
clean_df <- na.omit(clean_df)

clean_df <- clean_df[c(4,7,19,20,22,11,12)]
names(clean_df)


###########################################################################################
# 1) Find a replication archive (e.g., via a journal’s Dataverse site) for a recently published 
# article in which the authors estimate a linear regression model 
# (ordinary least squares or median regression). Confirm that you can reproduce the 
# results from the article, estimating the same coefficient values for the corresponding 
# variables in the model. Note, standard error estimates vary based on the software used 
# and settings selected, but you should be able to reproduce the coefficients. 
# Write up a brief introduction to the article, variables, and model, 
# and present the replicated results in your report. You can reuse the data 
# from Homework 1 if you'd like.
###########################################################################################


## Relationship between motive to aggress and seeing Neutral faces as Negative
names(clean_df) # x=totagg, y=Seeing neutral eyes as negative, data=clean_df

model1 <- lm(RMEneutralItemNegInacc_total ~ totagg + totalExplicitAggressionScore + totagg*totalExplicitAggressionScore, data=clean_df)
summary(model1)
# OG thesis results:
## 1) implicit ag --> neutral as neg: r=0.58, p=0.08 (marg)
# 2) explicit ag--> eutral as neg = r=0.19, p=0.002

library(mlr)
library(tidyverse)

# 9.1 load, explore data
dfTib <- as_tibble(clean_df)
dfTib

train70tib <- as_tibble(train70)
test30tib <- as_tibble(test30)

# make sure all vars are numeric
# then pipe result into filter() function to remove any NAs in dv
dfClean <- mutate_all(dfTib, as.numeric) %>%
  filter(is.na(RMEneutralItemNegInacc_total) == FALSE)

dfClean

# (do the same thing for test30 and train70, will need later)
train70 <- mutate_all(train70tib, as.numeric) %>%
  filter(is.na(RMEneutralItemNegInacc_total) ==FALSE)
test30 <- mutate_all(test30tib, as.numeric) %>%
  filter(is.na(RMEneutralItemNegInacc_total) ==FALSE)




# 9.3 plot each of the (important) predictor vars agains neutralNeg: 
dfUntidy <- gather(dfClean, key = "Variable",
                   value = "Value", -RMEneutralItemNegInacc_total)

ggplot(dfUntidy, aes(Value, RMEneutralItemNegInacc_total)) +
  facet_wrap(~ Variable, scale = "free_x") +
  geom_point() +
  geom_smooth() +
  geom_smooth(method = "lm", col = "red") +
  theme_bw()

# red lines are straight (linear)
# blue lines are curvy (GAM)


# 9.4 use rpart to impute any missing values:
imputeMethod <- imputeLearner("regr.rpart")

dfImp <- impute(as.data.frame(dfClean),
                classes = list(numeric = imputeMethod))


# 9.5 define the task and learner 
lrTask <- makeRegrTask(data = dfImp$data, target = "RMEneutralItemNegInacc_total")

lin <- makeLearner("regr.lm")


# 9.6 Filter method for feature selection
filterVals <- generateFilterValuesData(lrTask,
                                       method="linear.correlation")
filterVals$data
plotFilterValues(filterVals) + theme_bw()
# totalAcc is most important, but of course total accuracy on RME test is predictive of another score variable on RME
# Explicit Ag is second most important - use this
# totagg third most important - use this
# then age, then num siblings, then sex(basically 0)


# 9.8 create filter wrapper
filterWrapper = makeFilterWrapper(learner = lin,
                                  fw.method = "linear.correlation")

# 9.9 tune number of predictors to retain
lmParamSpace <- makeParamSet(
  makeIntegerParam("fw.abs", lower = 1, upper = 12)
)

gridSearch <- makeTuneControlGrid()

kFold <- makeResampleDesc("CV", iters = 10)

tunedFeats <- tuneParams(filterWrapper, task = lrTask, resampling = kFold,
                         par.set = lmParamSpace, control = gridSearch) 

tunedFeats
# Tune result:
# Op. pars: fw.abs=2
# mse.test.mean=1.6210742


# 9.10 train model with filtered features
filteredTask <- filterFeatures(lrTask, fval = filterVals,
                               abs = unlist(tunedFeats$x))

filteredModel <- train(lin, filteredTask)

# 9.11 Using wrapper method for feature selection
featSelControl <- makeFeatSelControlSequential(method = "sfbs")

selFeats <- selectFeatures(learner = lin, task = lrTask,
                           resampling = kFold, control = featSelControl)

selFeats
# FeatSel result:
# Features (0): 
#  mse.test.mean=2.3011733


# 9.12 use wrapper method for feature selection
dfImp$data
dfSelFeat <- dfImp$data[, c("RMEneutralItemNegInacc_total", selFeats$x)]

dfSelFeat

dfSelFeatTask <- makeRegrTask(data = dfSelFeat, target = "RMEneutralItemNegInacc_total")

wrapperModel <- train(lin, dfSelFeatTask)




# 9.13 combining imputation and feat select wrappers
imputeMethod <- imputeLearner("regr.rpart")

imputeWrapper <- makeImputeWrapper(lin,
                                   classes = list(numeric = imputeMethod))

featSelWrapper <- makeFeatSelWrapper(learner = imputeWrapper,
                                     resampling = kFold,
                                     control = featSelControl)


# 9.14 cross validate model building process
library(parallel)
library(parallelMap)

lrTaskWithNAs <- makeRegrTask(data = dfClean, target = "RMEneutralItemNegInacc_total")

kFold3 <- makeResampleDesc("CV", iters = 3)

parallelStartSocket(cpus = detectCores())

lmCV <- resample(featSelWrapper, lrTaskWithNAs, resampling = kFold3)

parallelStop()

lmCV
# Resample Result
# Task: dfClean
# Learner: regr.lm.imputed.featsel
# Aggr perf: mse.test.mean=2.3532972
# Runtime: 2.58595


# 9.15 interpret the model
wrapperModelData <- getLearnerModel(wrapperModel)

summary(wrapperModelData)






###########################################################################################
# 2) Split the data into 70% training and 30% test sets. 
# Use the training set alone for this part of the exercise. 
# Evaluating performance via cross-validation, identify a generalized additive model, 
# and a random forest, using the same independent variables, or a subset thereof, 
# used in the model replicated for Question 1.
###########################################################################################

# train and test split . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
smp_size <- floor(0.7 * nrow(clean_df))
set.seed(1234)
train_ind <- sample(seq_len(nrow(clean_df)), size = smp_size)
train70 <- clean_df[train_ind, ] # 70% training df 
test30 <- clean_df[-train_ind, ] # 30% testing df 



# On train70, do a generalized additive model (GAM) and cross-validate . . . . . . . . . . . .
library(mgcv)
mod_gam2 <- gam(RMEneutralItemNegInacc_total ~ 
                  s(totagg)  + 
                  s(totalExplicitAggressionScore), 
                data=train70)
summary(mod_gam2)


#fits=predict(mod_gam2, newdata=test30, type='response', se=T)
#predicts=data.frame(test30, fits)



# On train70, do a random forest and cross-validate (Rhys) . . . . . . . . . . . . . . . . . . . . . . . 
forest <- makeLearner("classif.randomForest")

# 8.1

# make task (see 7.1-7.3 of last chapter)
# 7.1 
train70tib <- as_tibble(train70)
train70tib <- mutate_all(train70tib, as.numeric)
names(train70tib)

# 7.2 
train70tib$RMEneutralItemNegInacc_total <- as.factor(train70tib$RMEneutralItemNegInacc_total)

# 7.3 create task 
rfTask <- makeClassifTask(data=train70tib, target="RMEneutralItemNegInacc_total")

forestParamSpace <- makeParamSet(     makeIntegerParam("ntree", lower = 300, upper = 300),
                                      makeIntegerParam("mtry", lower = 6, upper = 12),
                                      makeIntegerParam("nodesize", lower = 1, upper = 5),
                                      makeIntegerParam("maxnodes", lower = 5, upper = 20))
randSearch <- makeTuneControlRandom(maxit = 100) 

cvForTuning <- makeResampleDesc("CV", iters = 5)          

parallelStartSocket(cpus = detectCores())

tunedForestPars <- tuneParams(forest, task = rfTask,     
                              resampling = cvForTuning,     
                              par.set = forestParamSpace,   
                              control = randSearch)         

parallelStop()

tunedForestPars  

tunedForest <- setHyperPars(forest, par.vals = tunedForestPars$x)

tunedForestModel <- mlr::train(tunedForest, rfTask)



###########################################################################################
# Compare the predictive performance original regression model and the 
# other machine learning methods using the test data. Be sure not to estimate 
# any model using the test data. Comment on which model fits better on the test data.
###########################################################################################


# FOR LIN REG . . . . . . . . .
# Use Absolute Mean Error of predictions from test sets to compare performance:
library(Metrics)

test30 <- na.omit(test30)

predLinReg <- data.frame(predict(filteredModel, newdata = test30))
predLinReg
mae_lr <- sum(abs(predLinReg$response - test30$RMEneutralItemNegInacc_total))/110 # sample size here
mae_lr # MAE of linear regression model on test30 is 0.26


# FOR GAM . . . . . . . . .
predGam <- predict(mod_gam2, newdata=test30, type='response')
predGam
mae_GAM <- sum(abs(predGam)/110) # sample size here
mae_GAM # MAE of linear regression model on test30 is 0.37



# FOR RANDOM FOREST . . . . . . . . .
predRF <- data.frame(predict(tunedForestModel, newdata = test30))

predRF$response <-  as.numeric(predRF$response)
test30$RMEneutralItemNegInacc_total <- as.numeric(test30$RMEneutralItemNegInacc_total)

mae_rf <- sum(abs(predRF$response - test30$RMEneutralItemNegInacc_total))/110
mae_rf # MAE of random forest on test30 is 0.35



### linear regression model does the best, has the lowest absolute mean error (0.26)
### GAM and RF perform nearly equally well (0.37, 0.35, respectively)






# Now using the full data, calculate variable importance using each method (linear regression, GAM, RF), 
# and comment on whether the importance of variables differs across the three models. 
# Plot the effects of the GAM, and comment on whether you find any notable nonlinear relationships 
# that were not reflected in the linear model.


# linear regression variable importance .............................................
filterVals_lin <- generateFilterValuesData(lrTask,
                                           method="linear.correlation")
filterVals_lin$data
plotFilterValues(filterVals_lin) + theme_bw()
# totalAcc is most important, but of course total accuracy on RME test is predictive of another score variable on RME
# Explicit Ag is second most important - use this
# totagg third most important - use this
# then age, then num siblings, then sex(basically 0)


# GAM variable importance ...........................................................
filterVals_gam <- generateFilterValuesData(gamTask,
                                           method="linear.correlation")
filterVals_gam$data
plotFilterValues(filterVals_gam) + theme_bw()
# totAcc, explicit aggression, totagg, num siblings, sex, age
# same order as lin reg (first three predictors)
# demographics still least predictive



# RF variable importance ...........................................................

#install.packages("randomForestSRC")
library(randomForestSRC)
filtervalues_GAM <- generateFilterValuesData(rfTask,
                                             method="randomForestSRC_importance")


plotFilterValues(filtervalues_GAM) + theme_bw()
## importance in order: total acc, totagg, explicit agg, sex, siblings, age
# totacc still super important like in lin reg, imp and exp agg still tied for second/third, with demographics last
## so same pattern, slightly dif order than lin reg model AND GAM


# Plot GAM: Are there any notable nonlinear relationships not in the linear regression model? ..................


