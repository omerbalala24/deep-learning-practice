rm(list=ls())

library(dplyr)       # Data manipulation (0.8.0.1)
library(readr)       # Reading csv files (1.3.1)
library(tidyr)       # Database operations (0.8.3)
library(tibble)      # Modern alternative to data frames (2.1.1)
library(ggplot2)     # general plotting tool (3.1.0)
library(estimatr)    # simple interface for OLS estimation w/ robust std errors ()

#install.packages("tensorflow")
library(reticulate)
library(tensorflow)
#install_tensorflow(envname = "r-reticulate")
#install_keras(envname = "r-reticulate")
library(tensorflow)
#tf$constant("Hello Tensorflow!")

install.packages("tensorflow")
library(tensorflow)
install_tensorflow()

#install.packages("keras") #You'll need to uncomment this if you're running Keras for the first time
library(keras)
install_keras()  #You'll need to uncomment this if you're running Keras for the first time

#Set your working directory to where you have saved the data. This is mine:

df<-read.csv("/Users/omarbalala/Desktop/Machine Learning in Economics/tutorial 2/welfarelabel.csv") #Read in the data as a dataframe

#In this dataset y is an indicator for whether a survey respondent states that 
#“too much” is being spent on either welfare or assistance to the poor.
#w is an indicator for whether the question asked about "assistance to the poor"
#or "welfare". The paper can be found here: https://raw.githubusercontent.com/gsbDBI/ExperimentData/master/Welfare/Green%20and%20Kern%20BART.pdf
outcome_variable_name<-'y'
treatment_variable_name<-'w'

#We need to make the categorical variables into binary indicators. This is one way to do it:
df$wrkslf <- factor(df$wrkslf)
df$wrkstat <- factor(df$wrkstat)
df$evwork <- factor(df$evwork)
df$wrkgovt <- factor(df$wrkgovt)
df$marital <- factor(df$marital)

#We'll save the variable names and also make a matrix of the regressors (treatments and features)
covariate_names<-c('w','year','hrs1','hrs2','agewed','occ','evwork','wrkslf','wrkstat','wrkgovt','prestige','marital')
X<-model.matrix(~w + year + hrs1 +hrs2+ agewed + occ + evwork +wrkslf +wrkstat+wrkgovt+ prestige +marital , df)



# Combine all names of variables we wish to keep
all_variables_names <- c(outcome_variable_name, treatment_variable_name, covariate_names)
df <- df %>% select(one_of(all_variables_names)) #Drop all other variables

#Let's get a simple difference in means estimate of the ATE:
ATE<-mean(df$y[df$w==1])-mean(df$y[df$w==0])

train_fraction <- 0.80  # Use train_fraction % of the dataset to train our models the rest is a test/validation set
n <- dim(df)[1]
train_idx <- sample.int(n, replace=F, size=floor(n*train_fraction))
df_train <- df[train_idx,]
df_test <- df[-train_idx,]

X_train <-X[train_idx,]
X_test <-X[-train_idx,]


#Let's create a neural network with two hidden layers, 16 units in each layer, each using
#ReLU activiations and using drop-out
model <- keras_model_sequential()

#The code below defines the architecture of our model.
model %>%
  
  # Adds a densely-connected layer with 16 units to the model:
  layer_dense(units = 16, activation = 'relu',kernel_initializer=initializer_random_uniform(minval = -0.001, maxval = 0.001, seed = 2)) %>%
 #In the line above, you can change activation to say `sigmoid', or `linear'.
  #The `units' variable controls the number of units (a.k.a nodes/neurons) in that layer.
  #The `kernel initializer' argument tells it how to choose random starting values for the weights.` 
  
  #If you want to try using dropout for the layer above, un-comment the below
  # layer_dropout(rate = 0.5) %>% 
#Now let's add a second hidden layer:
  layer_dense(units = 16, activation = 'relu',kernel_initializer=initializer_random_uniform(minval = -0.001, maxval = 0.001, seed = 2)) %>%
 # layer_dropout(rate = 0.5) %>% #un-comment this line to add dropout for this layer
  
  #Final output layer will be a sigmoid function applied to a linear combination of the outputs from the 
  #hidden layer. 
  layer_dense(units = 1, activation = 'sigmoid',kernel_initializer=initializer_random_uniform(minval = -0.001, maxval = 0.001, seed = 2))

#If you want to add l1 and l2 penalties for the weights in the output layer replace the above
#with the line below:
#layer_dense(units = 1, activation = 'sigmoid',kernel_regularizer = regularizer_l1_l2(l1=0.0001,l2=0.0001),kernel_initializer=initializer_random_uniform(minval = -0.001, maxval = 0.001, seed = 2))
#If you want to add penalties to the other layers you can add the text below as an argument in the layer_dense function:
#kernel_regularizer = regularizer_l1_l2(l1=0.0001,l2=0.0001)


#Now we compile the model, telling tensorflow which optimization method to use and loss
#function (in our case the MSE). We will use the adagrad algorithm discussed in class.
model %>% compile(
  optimizer = optimizer_adagrad(learning_rate=0.01), #We'll optimize using the adagrad algorithm discussed in class
  loss = 'mse'
)

#Now we actually train the model on the test data. We can see how it does on the
#test sample as the model trains.
model %>% fit(
  X_train,
  df_train[,outcome_variable_name],
  epochs = 500,    #This controls how many times the optimizer goes through the whole dataset before it finishes training
  batch_size = 32,   #This controls the size of the batches of data used to update the weights by gradient descent
  validation_data=list(X_test,  df_test[,outcome_variable_name])  #This tells the optimizer to apply the model to our test data and show us how it performs as we optimize.
)

#Get test sample predictions:
predictions<-model %>% predict(X_test, batch_size = 32)
#Let's evaluate the MSE in the test sample (Keras should have already told us this in the console):
ANN_error<-mean((predictions-df_test[,outcome_variable_name])^2)


#Let's compare to OLS
ols_model <- lm(y~w + year + hrs1 +hrs2+ agewed + occ + evwork +wrkslf +wrkstat+wrkgovt+ prestige +marital, data = df_train)
predicted_y<-predict(ols_model, df_test[,c(treatment_variable_name,covariate_names)])
#And get the test sample MSE for OLS:
ols_error<-mean((predicted_y-df_test[,outcome_variable_name])^2)

