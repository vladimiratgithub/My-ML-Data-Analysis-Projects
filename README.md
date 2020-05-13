# Project in R
Actual files: AR_Modeling.R and Yield.dat
This project analyses input time series with chemical reaction data, identifies a right process to model it and estimates its parameters

# Project in Python/Pandas/sci-kit
Actual files: SVM.py
This medical project is devoted to analysis of the  set of 34 labels measured on clinic patients numbered as pid. For 1 patient maybe several measurements, i.e. several lines in the input (train_f.csv), which also contains a lot of NaNs. 
The labels from (train_labs.csv) correspond to 25 columns, have values 0 or 1 and tell whether a certain type of test with the label will be needed for patient in the future. Key goals are: a) Impute missing data; b) Predict for new measurements from test_f.csv whether the similar test as for train_labs will be needed.
Here, I provide only the code file that utilizes kNN, PCA and SVM algorithms to impute missing data (first two) and to predict a possible need of tests for patients (SVM and SVR for continous labels)

# Project in Python (regression)
Actual files: RidgeRegression.py <br>
This project fits a model on the train data using CV that can be later used for prediction using the test data 
