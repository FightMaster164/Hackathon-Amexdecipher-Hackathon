# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:59:56 2019

@author: Sailaja Raman
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler

import lightgbm as lgb
from sklearn import metrics, preprocessing, model_selection

# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LassoCV
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline


import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

# Classifier Libraries
from sklearn.metrics import mean_squared_log_error


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test_9K3DBWQ.csv')
print(df_train.shape, df_test.shape)


#label encoder the categorical features
labelencoder = LabelEncoder()
df_train['gender'] = labelencoder.fit_transform(df_train['gender'])

labelencoder = LabelEncoder()
df_test['gender'] = labelencoder.fit_transform(df_test['gender'])

labelencoder = LabelEncoder()
df_train['account_type'] = labelencoder.fit_transform(df_train['account_type'])

labelencoder = LabelEncoder()
df_test['account_type'] = labelencoder.fit_transform(df_test['account_type'])

df_train['loan_enq'].isnull()
#update the NaN values to 'N' indicating that they havent enquired for loan
df_train['loan_enq'].fillna('N', inplace = True)

df_test['loan_enq'].isnull()
#update the NaN values to 'N' indicating that they havent enquired for loan
df_test['loan_enq'].fillna('N', inplace = True)


labelencoder = LabelEncoder()
df_train['loan_enq'] = labelencoder.fit_transform(df_train['loan_enq'])

labelencoder = LabelEncoder()
df_test['loan_enq'] = labelencoder.fit_transform(df_test['loan_enq'])

#Check all the columns which have null values
null_columns = df_train.columns[df_train.isnull().any()]
df_train[null_columns].isnull().sum()


#amount of credit and debit card transactions
#Safe to replace with 0. These need not be missing values. 
#But instead '0' spend on debit card
df_train['dc_cons_apr'].fillna(0, inplace = True)
df_train['dc_cons_may'].fillna(0, inplace = True)
df_train['dc_cons_jun'].fillna(0, inplace = True)
df_train['cc_count_apr'].fillna(0, inplace = True)
df_train['cc_count_may'].fillna(0, inplace = True)
df_train['cc_count_jun'].fillna(0, inplace = True)
df_train['dc_count_apr'].fillna(0, inplace = True)
df_train['dc_count_may'].fillna(0, inplace = True)
df_train['dc_count_jun'].fillna(0, inplace = True)

df_test['dc_cons_apr'].fillna(0, inplace = True)
df_test['dc_cons_may'].fillna(0, inplace = True)
df_test['dc_cons_jun'].fillna(0, inplace = True)
df_test['cc_count_apr'].fillna(0, inplace = True)
df_test['cc_count_may'].fillna(0, inplace = True)
df_test['cc_count_jun'].fillna(0, inplace = True)
df_test['dc_count_apr'].fillna(0, inplace = True)
df_test['dc_count_may'].fillna(0, inplace = True)
df_test['dc_count_jun'].fillna(0, inplace = True)

#Can update to the maximum credit card transaction value across all the months
df_train[(df_train['card_lim'].isnull())] 
df_train.loc[df_train['card_lim'].isnull(), 'card_lim'] = df_train[['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun', ]].max(axis=1)

df_test[(df_test['card_lim'].isnull())] 
df_test.loc[df_test['card_lim'].isnull(), 'card_lim'] = df_test[['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun', ]].max(axis=1)

#Also there are few records where 'card_lim' is 0 and 1. They can be updated too.
df_train.loc[df_train['card_lim'] == 0, 'card_lim'] = df_train[['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun', ]].max(axis=1)
df_train.loc[df_train['card_lim'] == 1, 'card_lim'] = df_train[['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun', ]].max(axis=1)

df_test.loc[df_test['card_lim'] == 0, 'card_lim'] = df_test[['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun', ]].max(axis=1)
df_test.loc[df_test['card_lim'] == 1, 'card_lim'] = df_test[['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun', ]].max(axis=1)

#if there aren't any active loans mark them as 0.
#'1' indicates there are active loans going on currently.
df_train['personal_loan_active'].fillna(0, inplace = True)
df_train['vehicle_loan_active'].fillna(0, inplace = True)

df_test['personal_loan_active'].fillna(0, inplace = True)
df_test['vehicle_loan_active'].fillna(0, inplace = True)

#if the loans havent been closed in last 12 months, mark them 0
df_train['personal_loan_closed'].fillna(0, inplace = True)
df_train['vehicle_loan_closed'].fillna(0, inplace = True)

df_test['personal_loan_closed'].fillna(0, inplace = True)
df_test['vehicle_loan_closed'].fillna(0, inplace = True)

#if there aren't any investments made in Jun
df_train['investment_1'].fillna(0, inplace = True)
df_train['investment_2'].fillna(0, inplace = True)
df_train['investment_3'].fillna(0, inplace = True)
df_train['investment_4'].fillna(0, inplace = True)

df_test['investment_1'].fillna(0, inplace = True)
df_test['investment_2'].fillna(0, inplace = True)
df_test['investment_3'].fillna(0, inplace = True)
df_test['investment_4'].fillna(0, inplace = True)

#if there isn't any debit, credit amount in a month mark as 0
df_train['debit_amount_apr'].fillna(0, inplace = True)
df_train['credit_amount_apr'].fillna(0, inplace = True)
df_train['debit_amount_may'].fillna(0, inplace = True)
df_train['credit_amount_may'].fillna(0, inplace = True)
df_train['debit_amount_jun'].fillna(0, inplace = True)
df_train['credit_amount_jun'].fillna(0, inplace = True)

df_test['debit_amount_apr'].fillna(0, inplace = True)
df_test['credit_amount_apr'].fillna(0, inplace = True)
df_test['debit_amount_may'].fillna(0, inplace = True)
df_test['credit_amount_may'].fillna(0, inplace = True)
df_test['debit_amount_jun'].fillna(0, inplace = True)
df_test['credit_amount_jun'].fillna(0, inplace = True)

#null debit_count and credit_count indicates 0 transactions
df_train['debit_count_apr'].fillna(0, inplace = True)
df_train['credit_count_apr'].fillna(0, inplace = True)
df_train['debit_count_may'].fillna(0, inplace = True)
df_train['credit_count_may'].fillna(0, inplace = True)
df_train['debit_count_jun'].fillna(0, inplace = True)
df_train['credit_count_jun'].fillna(0, inplace = True)

df_test['debit_count_apr'].fillna(0, inplace = True)
df_test['credit_count_apr'].fillna(0, inplace = True)
df_test['debit_count_may'].fillna(0, inplace = True)
df_test['credit_count_may'].fillna(0, inplace = True)
df_test['debit_count_jun'].fillna(0, inplace = True)
df_test['credit_count_jun'].fillna(0, inplace = True)


#nunll max_credit_amount indicates, 0
df_train['max_credit_amount_apr'].fillna(0, inplace = True)
df_train['max_credit_amount_may'].fillna(0, inplace = True)
df_train['max_credit_amount_jun'].fillna(0, inplace = True)

df_test['max_credit_amount_apr'].fillna(0, inplace = True)
df_test['max_credit_amount_may'].fillna(0, inplace = True)
df_test['max_credit_amount_jun'].fillna(0, inplace = True)


#we have 2 options ither to Impute these records from our model or update the age with mean value.
#df_test[df_test['age'] >100]
df_train.loc[df_train['age'] > 100, 'age'] = df_train[df_train['age' ]<100]['age'].mean()

df_test.loc[df_test['age'] > 100, 'age'] = df_test[df_test['age' ]<100]['age'].mean()

#Lets do some feature engineering.
#Lets introduce a new feature for investment. and remove the individual investments..
df_train['investments'] = df_train['investment_1'] + df_train['investment_2'] + df_train['investment_3'] + df_train['investment_4'] 

#New feature for active loans, instead of 2 seperate features
df_train['loan_active'] = df_train['personal_loan_active'] + df_train['vehicle_loan_active']

#New feature for any recently closed loans, instead of 2 seperate features
df_train['loan_closed'] = df_train['personal_loan_closed'] + df_train['vehicle_loan_closed']

#max_credit_amount_apr is highly correlated to credit_amount_apr. So drop these columns

df_train['debit_amount_last_three_months'] = df_train['debit_amount_apr'] + df_train['debit_amount_may'] + df_train['debit_amount_jun']
df_train['credit_amount_last_three_months'] = df_train['credit_amount_apr'] + df_train['credit_amount_may'] + df_train['credit_amount_jun']
df_train['debit_count_last_three_months'] = df_train['debit_count_apr'] + df_train['debit_count_may'] + df_train['debit_count_jun']
df_train['credit_count_last_three_months'] = df_train['credit_count_apr'] + df_train['credit_count_may'] + df_train['credit_count_jun']
    
df_train['cc_per_trans_apr'] = df_train['cc_cons_apr'] / df_train['cc_count_apr']
df_train['cc_per_trans_may'] = df_train['cc_cons_may'] / df_train['cc_count_may']
df_train['cc_per_trans_jun'] = df_train['cc_cons_jun'] / df_train['cc_count_jun']

df_train['dc_per_trans_apr'] = df_train['dc_cons_apr'] / df_train['dc_count_apr']
df_train['dc_per_trans_may'] = df_train['dc_cons_may'] / df_train['dc_count_may']
df_train['dc_per_trans_jun'] = df_train['dc_cons_jun'] / df_train['dc_count_jun']


df_train.info()

df_train.drop(['investment_1', 'investment_2', 'investment_3', 'investment_4'], axis=1, inplace=True)
df_train.drop(['personal_loan_active', 'vehicle_loan_active'], axis=1, inplace = True)
df_train.drop(['personal_loan_closed', 'vehicle_loan_closed'], axis=1, inplace=True)
df_train.drop(['max_credit_amount_apr', 'max_credit_amount_may', 'max_credit_amount_jun'], axis=1, inplace=True)
df_train.drop(['debit_amount_apr', 'debit_amount_may', 'debit_amount_jun', 
         'credit_amount_apr', 'credit_amount_may', 'credit_amount_jun',
         'debit_count_apr', 'debit_count_may', 'debit_count_jun',
         'credit_count_apr', 'credit_count_may', 'credit_count_jun'], axis=1, inplace=True)
df_train.drop(['cc_cons_apr', 'cc_count_apr', 'cc_cons_may', 'cc_count_may', 'cc_cons_jun', 'cc_count_jun',
         'dc_cons_apr', 'dc_count_apr', 'dc_cons_may', 'dc_count_may', 'dc_cons_jun', 'dc_count_jun'], axis=1, inplace=True)

    
df_test['investments'] = df_test['investment_1'] + df_test['investment_2'] + df_test['investment_3'] + df_test['investment_4'] 

#New feature for active loans, instead of 2 seperate features
df_test['loan_active'] = df_test['personal_loan_active'] + df_test['vehicle_loan_active']

#New feature for any recently closed loans, instead of 2 seperate features
df_test['loan_closed'] = df_test['personal_loan_closed'] + df_test['vehicle_loan_closed']

#max_credit_amount_apr is highly correlated to credit_amount_apr. So drop these columns

df_test['debit_amount_last_three_months'] = df_test['debit_amount_apr'] + df_test['debit_amount_may'] + df_test['debit_amount_jun']
df_test['credit_amount_last_three_months'] = df_test['credit_amount_apr'] + df_test['credit_amount_may'] + df_test['credit_amount_jun']
df_test['debit_count_last_three_months'] = df_test['debit_count_apr'] + df_test['debit_count_may'] + df_test['debit_count_jun']
df_test['credit_count_last_three_months'] = df_test['credit_count_apr'] + df_test['credit_count_may'] + df_test['credit_count_jun']
    
df_test['cc_per_trans_apr'] = df_test['cc_cons_apr'] / df_test['cc_count_apr']
df_test['cc_per_trans_may'] = df_test['cc_cons_may'] / df_test['cc_count_may']
df_test['cc_per_trans_jun'] = df_test['cc_cons_jun'] / df_test['cc_count_jun']

df_test['dc_per_trans_apr'] = df_test['dc_cons_apr'] / df_test['dc_count_apr']
df_test['dc_per_trans_may'] = df_test['dc_cons_may'] / df_test['dc_count_may']
df_test['dc_per_trans_jun'] = df_test['dc_cons_jun'] / df_test['dc_count_jun']    
    
df_test.drop(['investment_1', 'investment_2', 'investment_3', 'investment_4'], axis=1, inplace=True)
df_test.drop(['personal_loan_active', 'vehicle_loan_active'], axis=1, inplace = True)
df_test.drop(['personal_loan_closed', 'vehicle_loan_closed'], axis=1, inplace=True)
df_test.drop(['max_credit_amount_apr', 'max_credit_amount_may', 'max_credit_amount_jun'], axis=1, inplace=True)
df_test.drop(['debit_amount_apr', 'debit_amount_may', 'debit_amount_jun', 
         'credit_amount_apr', 'credit_amount_may', 'credit_amount_jun',
         'debit_count_apr', 'debit_count_may', 'debit_count_jun',
         'credit_count_apr', 'credit_count_may', 'credit_count_jun'], axis=1, inplace=True)
df_test.drop(['cc_cons_apr', 'cc_count_apr', 'cc_cons_may', 'cc_count_may', 'cc_cons_jun', 'cc_count_jun',
         'dc_cons_apr', 'dc_count_apr', 'dc_cons_may', 'dc_count_may', 'dc_cons_jun', 'dc_count_jun'], axis=1, inplace=True)


df_train.info()
df_test.info()
print(df_train.shape, df_test.shape)


def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, dep=8, seed=0, data_leaf=50, rounds=20000):
    params = {}
    params["objective"] = "regression"
    params['metric'] = 'rmse'
    params["max_depth"] = dep
    params["num_leaves"] = 50
    params["min_data_in_leaf"] = data_leaf
#     params["min_sum_hessian_in_leaf"] = 50
    params["learning_rate"] = 0.01
    params["bagging_fraction"] = 0.8
    params["feature_fraction"] = 0.2
    params["feature_fraction_seed"] = seed
    params["bagging_freq"] = 1
    params["bagging_seed"] = seed
    params["lambda_l2"] = 3
    params["lambda_l1"] = 3
    params["verbosity"] = -1
    num_rounds = rounds

    plst = list(params.items())
    lgtrain = lgb.Dataset(train_X, label=train_y)

    if test_y is not None:
        lgtest = lgb.Dataset(test_X, label=test_y)
        model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest], early_stopping_rounds=300, verbose_eval=500)
    else:
        lgtest = lgb.DMatrix(test_X)
        model = lgb.train(params, lgtrain, num_rounds)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    #imps = model.feature_importance()
    #names = model.feature_name()
    #for fi, fn in enumerate(names):
    #    print(fn, imps[fi])

    loss = 0
    if test_y is not None:
        loss = np.sqrt(metrics.mean_squared_error(test_y, pred_test_y))
        print(loss)
        return model, loss, pred_test_y, pred_test_y2
    else:
        return model, loss, pred_test_y, pred_test_y2
    
    
##########################################################

train_y = df_train['cc_cons']
df_train.drop(['cc_cons'], axis=1, inplace=True)
#df_test.drop(['id'], axis=1, inplace=True)

print("Building model..")
cv_scores = []
pred_test_full = 0
pred_train = np.zeros(df_train.shape[0])
n_splits = 5
#kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=7988)
gkf = model_selection.GroupKFold(n_splits=n_splits)
model_name = "lgb"
for dev_index, val_index in gkf.split(df_train, train_y, df_train["region_code"].values):
    dev_X, val_X = df_train.iloc[dev_index,:], df_train.iloc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val = 0
    pred_test = 0
    n_models = 0.

    model, loss, pred_v, pred_t = runLGB(dev_X, dev_y, val_X, val_y, df_test, dep=6, data_leaf=300, seed=3568)
    pred_val += pred_v
    pred_test += pred_t
    n_models += 1
    
    model, loss, pred_v, pred_t = runLGB(dev_X, dev_y, val_X, val_y, df_test, dep=7, data_leaf=230, seed=6875)
    pred_val += pred_v
    pred_test += pred_t
    n_models += 1

    pred_val /= n_models
    pred_test /= n_models
    
    #loss = np.sqrt(metrics.mean_squared_error(val_y, pred_val))
     
    pred_val[np.where(pred_val < 0)] = 0
    RMSEL_metric_Value = mean_squared_log_error(val_y, pred_val)
        
    pred_train[val_index] = pred_val
    pred_test_full += pred_test / n_splits
    
    
    
    cv_scores.append(RMSEL_metric_Value)
    print(cv_scores)
#     break
print(np.mean(cv_scores))

pred_test_full[np.where(pred_test_full < 0)] = 0
submit_df = pd.DataFrame(df_test["id"])
submit_df["cc_cons"] = pred_test_full
submit_df.to_csv("submit.csv", index=False)





