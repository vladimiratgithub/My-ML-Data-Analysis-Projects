from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import math
import model_Alexandre
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


from sklearn import linear_model

df= pd.read_csv("train.csv")
y= df['y'].to_numpy()
df = df.drop(['Id'], axis=1)
df = df.drop(['y'], axis=1)

# podschet intercepta ne vliyayet na resulataty i coeff_ = 0 sootv
df['intercept']= np.ones(len(df.index))
df = df[['intercept', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','x9','x10','x11','x12','x13']]

#len(df.index) - number of strings in adataframe without accounting for labels
#len(df.columns) - number of columns in a dataframe

X= df.to_numpy()

Nsplits= 10
lambda_arr= np.array([0.01, 0.1, 1.0, 10.0, 100.0])
mean_RMSE= np.empty(lambda_arr.size)
for j in range(lambda_arr.size):
    kf = KFold(n_splits= 10, random_state=j*10, shuffle=True)  # Shuffles indices of the array X before splitting
    regr = linear_model.Ridge(lambda_arr[j], fit_intercept= True, normalize= False)  # fits intercept, no normalization as aked in Task_1a
    RMSEs= cross_val_score(regr, X, y, scoring= 'neg_root_mean_squared_error', cv=kf)
    mean_RMSE[j]= (-1)*np.mean(RMSEs)

np.savetxt("sample.csv", mean_RMSE, delimiter="\n", fmt="%.20f")