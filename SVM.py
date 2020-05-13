import pandas as pd
import numpy as np
import gzip

from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA



def dfImputer(flag, kNN_flag):  
    # flag == 1 corresponds to the shorter version for testing; kNN_flag = # of nearest neihghbours in kNN
    df_train_X= pd.read_csv("train_f.csv") # with pid columns
    df_test_X= pd.read_csv("test_f.csv")  # with pid's as I calculate labels for them

    # train_f.csv IS A SET OF 35 FEATURES containing the results of tests for each patient
    # Each line corresponds to a patient. For 1 patient may be several lines as they correspond to different moments ot time
    # train_labs.csv test data contains 0 and 1 for LABLES_A, while medical LABELS_B contain real numbers as a result of measurements
    # test_f.csv is similar to train_f.csv and used by the model for predictions
    # test_labs.csv is being predicted by the model
    
    if (flag == 1):  # smaller array for  
        df_train_X= df_train_X.iloc[0:10000]
        df_test_X= df_train_X.iloc[0:10000]

    pid_col_train= df_train_X['pid']  # unique for train and test sets <-> patient id
    pid_col_test= df_test_X['pid']  # unique for train and test sets

    df_train_X= df_train_X.drop(columns= ['pid'])
    df_test_X= df_test_X.drop(columns= ['pid'])

    ALL_LABELS= df_train_X.columns  # universal for both train and test sets

    train_X= preprocessing.scale(df_train_X)
    test_X= preprocessing.scale(df_test_X)

    df_train_X= pd.DataFrame(data= train_X, columns= ALL_LABELS)
    df_test_X= pd.DataFrame(data= test_X, columns= ALL_LABELS)

    df_train_X.insert(0, 'pid', pid_col_train)         
    df_test_X.insert(0, 'pid', pid_col_test)         

    # Delete redundant features
    df_train_X= df_train_X.drop(columns=['Hgb'])
    df_test_X= df_test_X.drop(columns=['Hgb'])
    
    df_train_X= df_train_X.groupby(['pid']).mean()
    df_test_X= df_test_X.groupby(['pid']).mean()

    df_train_X= df_train_X.reset_index(drop= False)  # as previous step sets pid as a default index
    df_test_X= df_test_X.reset_index(drop= False)  # False is to keep a pid coloumn there
    
    pid_col_train= df_train_X['pid']  # because of taking mean over pid in line before
    pid_col_test= df_test_X['pid']  # because of taking mean over pid in line before

    df_train_X= df_train_X.drop(columns= ['pid'])
    df_test_X= df_test_X.drop(columns= ['pid'])

    ALL_LABELS= df_train_X.columns  # deleted patient id; universal for both train and test sets

    for label in (['train', 'test']):
        if (label == 'train'):
            df_unfilled= df_train_X 
            pid_col= pid_col_train
        if (label == 'test'):
            df_unfilled= df_test_X 
            pid_col= pid_col_test

        imputer= KNNImputer(n_neighbors= kNN_flag)  #
        df_X= imputer.fit_transform(df_unfilled)
        df_X= preprocessing.scale(df_X) # standardize after kNN
        df_X= pd.DataFrame(data= df_X, columns= df_train_X.columns)
    
        df_X.insert(0, 'pid', pid_col) # pid_col['train'] or pid_col['test']   
        df_X[['pid']]= df_X[['pid']].astype(int)  # imputed group
        print('df_', label, '_X is imputed')
    
        # Prepare df_X for classification
        df_X= df_X.sort_values(by=['pid'])
        df_X= df_X.drop(columns=['pid'])  # no pid in train output
        df_X= df_X.reset_index(drop= True)
        if (label == 'train'):
            df_train_X= df_X
        if (label == 'test'):
            df_test_X= df_X

    # Preparaing label data
    df_train_labels= pd.read_csv("train_labs.csv")  #with pid by default

    # Fit pid_column of input train labels to that [since test labels are not provided - we calculate them]
    # I can just copy the number of unque pid rows from df_test_X to test_label 

    ind=[]  
    aa= pid_col_train.to_list()
    for i, j in df_train_labels['pid'].items():  # from all pid in train_labels taekeonly that, which arepresented in train_X
        if (j in aa):
            ind.append(i)
    df_train_labels= df_train_labels.loc[ind]  
    df_train_labels= df_train_labels.sort_values(by=['pid'])  # pid can remain as I will anyway extract columns from here
    df_train_labels= df_train_labels.reset_index(drop= True)

    # Prepare test_labels
    df_test_labels= pd.DataFrame(columns= df_train_labels.columns)  # LABEL columns are the same
    df_test_labels['pid']= pid_col_test  # take all the pids from the test
    df_test_labels= df_test_labels.sort_values(by=['pid'])
    df_test_labels= df_test_labels.reset_index(drop= True)

    # OUTPUT: train_X test_X - with feature labels, but no pids; sorted
    #train_labels, test_labels: both with pids (individual for two arrays) and features (the same for two arrays);#Vladimir Korenev,11.05.20

    df_train_X.to_csv("df_X_train.csv", index= False, header= True) # with pid columns
    df_test_X.to_csv("df_X_test.csv", index= False, header= True) # with pid columns
    df_train_labels.to_csv("df_X_train_labels.csv", index= False, header= True) # with pid columns
    df_test_labels.to_csv("df_X_test_labels.csv", index= False, header= True) # with pid columns

    print('Data vorbereitung: DONE!')
    
    return 'Done'



def sigmoid(X):  # to predict probabilities between [0, 1]
    return 1/(1 + np.exp(-X))  # X is a vector



def get_score(df_train_X, df_train_labels, df_test_X, df_test_labels, params, write_all): 
    # Calculates kernelized SVM prediction and its score
    ALL_LABELS= df_train_labels.columns.drop('pid').tolist()
    
    LABELS_B= ['LABEL_B1', 'LABEL_B2', 'LABEL_B3', 'LABEL_B4']
    LABELS_A= ['LABEL_A1', 'LABEL_A2', 'LABEL_A3', 'LABEL_A4', 'LABEL_A5',
             'LABEL_A6', 'LABEL_A7', 'LABEL_A8',
             'LABEL_A9', 'LABEL_A10']
    df_scores= pd.DataFrame(index=LABELS_A, columns=['score'])

    counter= 0
    for CurLabelName in ALL_LABELS:
        counter= counter + 1
        print('Prediction for LABEL #', counter, 'out of ', len(ALL_LABELS), 'labels')
        if (CurLabelName in LABELS_A):
            flag= 'lab_a'
            clf= SVC(C=params.loc[flag, 'C'].values[0], kernel=params.loc[flag, 'ker'].values[0], 
                     degree=params.loc[flag, 'deg'].values[0], class_weight='balanced', gamma='scale', tol=0.001)
            clf.fit(df_train_X, df_train_labels[CurLabelName])
            
            cv= StratifiedKFold(n_splits= 10, shuffle= True, random_state= 1)
            scores= cross_val_score(clf, df_train_X, df_train_labels[CurLabelName], scoring='roc_auc', cv=cv, n_jobs=-1)
            df_scores.loc[CurLabelName, 'score']= np.mean(scores)
            df_test_labels[CurLabelName]= sigmoid(clf.decision_function(df_test_X)).tolist() # .toarray().tolist()
            
        if (CurLabelName in LABELS_B):
            flag= 'lab_b'
            clf= SVR(C=params.loc[flag, 'C'].values[0], kernel=params.loc[flag, 'ker'].values[0], 
                     degree=params.loc[flag, 'deg'].values[0], gamma='scale', tol=0.001)
            clf.fit(df_train_X, df_train_labels[CurLabelName])
            
            cv= KFold(n_splits= 10, shuffle= True, random_state= 3)
            scores= cross_val_score(clf, df_train_X, df_train_labels[CurLabelName], scoring='r2', cv=cv, n_jobs=-1)
            df_scores.loc[CurLabelName, 'score']= np.maximum(0, np.mean(scores))
            df_test_labels[CurLabelName]= clf.predict(df_test_X).tolist() # .toarray().tolist()

    if (write_all == True):
        df_test_labels.to_csv(r'test_labs.csv', float_format= '%.5f', index = False, header=True)

    # Calculate Scores:
    task1 = np.mean(df_scores.loc['LABEL_B1':'LABEL_B4', 'score'].tolist())
    task2 = np.mean(df_scores.loc['LABEL_A1':'LABEL_A10', 'score'].tolist())
    score_list= [task1, task2]
    
    return score_list  # score_list[] should be an output for the massive optimization



def Optimizer(df_train_X, df_train_labels, df_test_X, df_test_labels):
    # Calculate scores for different parameter combinations of C, kernel type and degree for polynomial kernel

    C_arr= [0.01, 0.1, 1.0, 5.0]
    TaskList= ['lab_a', 'lab_b']
    params= pd.DataFrame(index= [TaskList], columns= ['C', 'ker', 'deg'])
    tasks= pd.Series(index= TaskList)
    dfScores= pd.DataFrame(columns=['group', 'C', 'ker', 'deg', 'score']) # e.g. [lab_a, 1.0,rbf,3.0, 11.5]
    
    for ker in (['rbf']):
        for CurC in (C_arr):
            print('CurC: ', CurC)
            for label in TaskList:
                params.loc[label]= [CurC, ker, 3]      
            tasks['lab_a':'lab_b']= get_score(df_train_X, df_train_labels, df_test_X, df_test_labels, params, False)
            for label in TaskList:
                CurID= len(dfScores)
                dfScores.loc[CurID]= [label, CurC, ker, 3]+[tasks[label]]  # drugoy variant - prosto na osnove [..] sdelat df and append

    gkk= dfScores.groupby(['group'])['score'].max()  # AOC should be max

    test= []
    for label in (['lab_a', 'lab_b']):
        i= dfScores.index[(dfScores['score'] == gkk[label]) & (dfScores['group'] == label)].values[0]
        params.loc[label]= dfScores.loc[i, 'C':'deg'].tolist()
        test.append(dfScores.loc[i, 'score'])
        
    ResScore= np.mean(test)
    print('Optimal Score: ', ResScore)
    print('Optimal [test1, test2]: ', test)
    print('Total Scores: ', dfScores)    

    return params



# MAIN CODE

dfImputer(1, 5)  # impute the data; 0=default, all the data is takein into account, 1 - first 10k points

df_train_X= pd.read_csv("df_X_train.csv") # with pid columns
df_test_X= pd.read_csv("df_X_test.csv") # with pid columns
df_train_labels= pd.read_csv("df_X_train_labs.csv") # with pid columns
df_test_labels= pd.read_csv("df_X_test_labs.csv") # with pid columns


TaskList= ['lab_a', 'lab_b']
params= pd.DataFrame(index= [TaskList], columns= ['C', 'ker', 'deg'])
params.loc['lab_a']= [0.1, 'rbf', 3]
params.loc['lab_b']= [5.0, 'rbf', 3]

params= Optimizer(df_train_X, df_train_labels, df_test_X, df_test_labels)  # Finds Optimized parameters
[test1, test2]= get_score(df_train_X, df_train_labels, df_test_X, df_test_labels, params, True)


score= np.mean([test1, test2])
print(params)
print('Score: ', score)