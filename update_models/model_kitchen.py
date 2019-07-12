import pickle
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

#provide the file path to input
df = pd.read_csv('../input_csv/input_csv.csv')
#removing very high floor plan sizes( Only 1 sample)
df=df[df['Floorplan Size']<=5000]
#Binning by quantiles
bins = [287,1055,1233,1585,4813]
df['binned'] = pd.cut(df['Floorplan Size'],bins=bins,labels=[1,2,3,4])
df_new = pd.DataFrame()
df_new['Property Config']= df['Property Config']
df_new['Kitchen Shape'] = df['Kitchen Shape']
df_new['City'] =df['City']
df_new['Floorplan Size']= df['Floorplan Size']
df_new['Wall A']=df['Wall A']
df_new['Wall B']=df['Wall B']
df_new['Wall C']=df['Wall C']
df_new['binned']=df['binned']
#df_new = df_new[df_new['City']!='Kolkata']
#df_new = df_new[df_new['City']!='Pune']
#df_new = df_new[df_new['Property Config']!='4 BHK']
df_new.loc[df['City']=='Thane',"City"]='Mumbai'
df_new.loc[df['City']=='Thane',"City"]='Mumbai'
df_new.loc[df['City'].isin(['Gurgaon', 'Ghaziabad','New Delhi','Noida']),"City"]='NCR'
df_new.loc[df['City']=='Chennai',"City"]='Chennai'
df_new.loc[df['City']=='Hyderabad',"City"]='Hyderabad'

#model definition for wall A
features = pd.get_dummies(df_new[['Property Config','binned','City']])
output=df_new[['Wall A','Wall B','Wall C']]
print(features.columns)


def linear_regressionA(x_train, x_test, y_train, y_test):
    rf = Ridge(normalize=True,alpha=3)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train error",mean_squared_error(rf.predict(x_train),y_train))
    print("test error",mean_squared_error(predictions,y_test))
    return rf,rf.predict(x_train).reshape(-1),predictions.reshape(-1)


def linear_regressionB(x_train, x_test, y_train, y_test):
    rf = Ridge(alpha=2)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train error",mean_squared_error(rf.predict(x_train),y_train))
    print("test error",mean_squared_error(predictions,y_test))
    return rf,rf.predict(x_train).reshape(-1),predictions.reshape(-1)


def linear_regressionC(x_train, x_test, y_train, y_test):
    rf = Ridge(alpha=2)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train error",mean_squared_error(rf.predict(x_train),y_train))
    print("test error",mean_squared_error(predictions,y_test))
    return rf,rf.predict(x_train).reshape(-1),predictions.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(features,output, test_size=.1)

# removing na tuples of output
X_trainA = X_train[y_train['Wall A'].notna()]
y_trainA = y_train[y_train['Wall A'].notna()]['Wall A']
X_testA = X_test[y_test['Wall A'].notna()]
y_testA = y_test[y_test['Wall A'].notna()]['Wall A']

X_trainB = X_train[y_train['Wall B'].notna()]
y_trainB = y_train[y_train['Wall B'].notna()]['Wall B']
X_testB= X_test[y_test['Wall B'].notna()]
y_testB = y_test[y_test['Wall B'].notna()]['Wall B']

X_trainC = X_train[y_train['Wall C'].notna()]
y_trainC = y_train[y_train['Wall C'].notna()]['Wall C']
X_testC = X_test[y_test['Wall C'].notna()]
y_testC = y_test[y_test['Wall C'].notna()]['Wall C']


lrA, train_predA, test_predA = linear_regressionA(X_trainA, X_testA, y_trainA, y_testA)
lrB, train_predA, test_predA = linear_regressionB(X_trainB, X_testB, y_trainB, y_testB)
lrC, train_predC, test_predC = linear_regressionC(X_trainC, X_testC, y_trainC, y_testC)


dump(lrA, '../models/lrA.joblib')
dump(lrB, '../models/lrB.joblib')
dump(lrC, '../models/lrC.joblib')
