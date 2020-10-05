#XGBoost Multi Classification Model 
#Code by Hailey Tucker
#Spring 2020 Research 

#Input Type: Sparse Matrix (Will this handle sparse issues with goals realized being so low?)

#Installing Packages
#From Github for latest version...
#install.packages("drat", repos="https://cran.rstudio.com") 
#drat:::addRepo("dmlc") 
#install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")

library(dplyr)
# install.packages("xgboost")
library(xgboost)
#install.packages("DMwR")
library(DMwR)
# install.packages("Matrix")
library(Matrix)
# install.packages("DiagrammeR")
library(DiagrammeR)
#install.packages('caret')
library(caret)
#install.packages('e1071')
library(e1071)

# Data Preparation -----------------
# Loading in Dataset (full csv)


final_df <- read_csv("~/Documents/Research Spring 2020/final_df.csv")
final_df<- final_df[-1]

# Cleaning column names 
names(final_df)[6] <- "average_shot_quality"
names(final_df)[10] <- "minutes_played"
names(final_df)[13] <- "attacking_challenges"
names(final_df)[15] <- "challenges_in_attack_won_pct"

final_df <- select(final_df, -date)  # removing date 

final_df$BinGoal <- ifelse(final_df$Goals > 0, 1, 0)  # Target Variable , Goal Yes or No 
final_df$BinGoal <- as.factor(final_df$BinGoal)  # Making target variable BinGoal into a factor
#final_df$Position <- as.factor(final_df$Position) # Position to factor
#final_df$Team <- as.factor(final_df$Team) # Team to factor
#final_df$OppTeam <- as.factor(final_df$OppTeam) # Opposing team to factor

str(final_df) # checking var types

# One hot encoding 
for(unique_value in unique(final_df$Position)){ # Position
  final_df[paste("position", unique_value, sep = ".")] <- ifelse(final_df$Position == unique_value, 1, 0)
}
for(unique_value in unique(final_df$Team)){  # Team 
  final_df[paste("team", unique_value, sep = ".")] <- ifelse(final_df$Team == unique_value, 1, 0)
}
for(unique_value in unique(final_df$OppTeam)){ # Opposing Team 
  final_df[paste("oppTeam", unique_value, sep = ".")] <- ifelse(final_df$OppTeam == unique_value, 1, 0)
}

final_df <- select(final_df, -c("Position", "Team", "OppTeam", "Player")) # Remove categorical variables 
final_df <- select(final_df, -Goals)


# Split data into train and Test Data ---------
set.seed(42920)
sample <- sample(nrow(final_df), 0.75*nrow(final_df)) # sample for 75/25% training data
df_train <- final_df[sample,] # creating training set
#df_train <- select(df_train, -Goals)  # Commented out as goals are taken out above
table(df_train$BinGoal)

#df_train <- as.matrix(as.data.frame(df_train)) # makes it into a dataframe, not sure if this is what we want

# Convert df_train from tibble to data frame
df_train <- data.frame(df_train)

balanced_train <- SMOTE(BinGoal ~ ., df_train, perc.over = 600, perc.under=100) # Creating balanced training set table(balanced_train$BinGoal)

label <- balanced_train$BinGoal #  Target Variable
df_t <- select(balanced_train, -BinGoal)
df_t <- as.matrix(df_t) # run this code
#df_matrix <- as.matrix(sapply(df_train, as.numeric)) 

df_test <- final_df[-sample,] # Creating Training Set
table(df_test$BinGoal)

tslabel <- df_test$BinGoal # Seperating Test Target Variable 
df_test <- as.matrix(select(df_test, -BinGoal))

dftrain <- xgb.DMatrix(data = df_t, label = as.numeric(label) - 1)
dftest <- xgb.DMatrix(data = df_test, label = as.numeric(tslabel) - 1)

param <- list(max_depth = 2, eta = 1, nthread = 10, objective = "binary:logistic")
test <- xgb.train(param, dftrain, nrounds = 5) # maybe switch to xgboost framework 
test$pred <- predict(test, dftest)
table(test$pred,tslabel)

xgb <- xgb.train(param, dftrain, nrounds = 5) # Use this model for confusion matrix

xgb_plot<-xgb.plot.tree(model = xgb, trees = 0:1, render = FALSE)

xgb_plot

# confusion matrix for output
xgbpred <- predict(xgb,df_test)
xgbpred <- ifelse(xgbpred>0.5,1,0)
u <- union(xgbpred,tslabel)
tbl <- table(factor(xgbpred,u),factor(tslabel,u))
tbl <- confusionMatrix(tbl)
print("XGB Model")
print(tbl)

mat<-xgb.importance(feature_names = colnames(dftrain), model = xgb) # Measuring variable importance 
modeldf <- final_df[,names(final_df)%in%c("one", mat$Feature[1:nrow(mat)*.66])]

modeldf # average player index, average chances per min, average shots per minute, and if they are a forward yes or no
        # seem to be the most imporant variables 

# ggplot for relationships, x = predicted goals, y = actual BinGoals, gmsmooth 


