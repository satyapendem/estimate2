import pickle
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('./Estimator 2.0 Data Analysis - visualize_kitchen_shape.csv')
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
df_new['Bedroom 1']=df['Bedroom 1']
df_new['Bedroom 2']=df['Bedroom 2']
df_new['Bedroom 3']=df['Bedroom 3']
df_new['binned']=df['binned']
#df_new = df_new[df_new['City']!='Kolkata']
#df_new = df_new[df_new['City']!='Pune']
#df_new = df_new[df_new['Property Config']!='4 BHK']
df_new.loc[df['City']=='Thane',"City"]='Mumbai'
df_new.loc[df['City']=='Thane',"City"]='Mumbai'
df_new.loc[df['City'].isin(['Gurgaon', 'Ghaziabad','New Delhi','Noida']),"City"]='NCR'
df_new.loc[df['City']=='Chennai',"City"]='Chennai'
df_new.loc[df['City']=='Hyderabad',"City"]='Hyderabad'



def random_forest_bedroom1(x_train, x_test, y_train, y_test):
    rf = RandomForestRegressor(n_estimators= 23,min_samples_split= 5,min_samples_leaf= 1,max_features= 'sqrt',max_depth= 1,bootstrap= True)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train error",mean_squared_error(rf.predict(x_train),y_train))
    print("test error",mean_squared_error(predictions,y_test))
    return rf, rf.predict(x_train),predictions

def linear_regression_bedroom1(x_train, x_test, y_train, y_test):
    rf = Ridge(normalize=True,alpha=1)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train error",mean_squared_error(rf.predict(x_train),y_train))
    print("test error",mean_squared_error(predictions,y_test))
    return rf,rf.predict(x_train).reshape(-1),predictions.reshape(-1)


#model definition for bedroom 2
def linear_regression_bedroom2(x_train, x_test, y_train, y_test):
    rf = Ridge(normalize=True,alpha=1)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train error",mean_squared_error(rf.predict(x_train),y_train))
    print("test error",mean_squared_error(predictions,y_test))
    return rf,rf.predict(x_train).reshape(-1),predictions.reshape(-1)

def random_forest_bedroom2(x_train, x_test, y_train, y_test):
    rf = RandomForestRegressor(n_estimators= 23,min_samples_split= 5,min_samples_leaf= 1,max_features= 'sqrt',max_depth= 1,bootstrap= True)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train error",mean_squared_error(rf.predict(x_train),y_train))
    print("test error",mean_squared_error(predictions,y_test))
    return rf,rf.predict(x_train),predictions


#model definition for bedroom 3
def linear_regression_bedroom3(x_train, x_test, y_train, y_test):
    rf = Ridge(alpha=4)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train error",mean_squared_error(rf.predict(x_train),y_train))
    print("test error",mean_squared_error(predictions,y_test))
    return rf,rf.predict(x_train).reshape(-1),predictions.reshape(-1)

def random_forest_bedroom3(x_train, x_test, y_train, y_test):
    rf = RandomForestRegressor(n_estimators= 12,min_samples_split= 5,min_samples_leaf= 4,max_features= 'sqrt',max_depth= 100,bootstrap= True)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train error",mean_squared_error(rf.predict(x_train),y_train))
    print("test error",mean_squared_error(predictions,y_test))
    return rf,rf.predict(x_train),predictions


#Removing nan values
df_b1 = df_new[df_new['Bedroom 1'] >=2]
df_b2 = df_new[df_new['Bedroom 2'] >=2]
df_b3 = df_new[df_new['Bedroom 3'] >=2]

features_b1 = pd.get_dummies(df_b1[['Property Config','binned','City']])
output_b1 =df_b1[['Bedroom 1']]

features_b2 = pd.get_dummies(df_b2[['Property Config','binned','City']])
output_b2 =df_b2[['Bedroom 2']]

features_b3 = pd.get_dummies(df_b3[['Property Config','binned','City']])
output_b3 =df_b3[['Bedroom 3']]


print("For Bedroom 1")
X_train, X_test, y_train, y_test = train_test_split(features_b1,output_b1, test_size=.1)
print("Random Forest")
rfA, train_predA, test_predA = random_forest_bedroom1(X_train, X_test, y_train, y_test)
print('Linear Regression')
lrA, train_predA, test_predA = linear_regression_bedroom1(X_train, X_test, y_train, y_test)


print("For Bedroom 2")
X_train, X_test, y_train, y_test = train_test_split(features_b2,output_b2, test_size=.1)
print("Random Forest")
rfB, train_predA,test_predA = random_forest_bedroom2(X_train, X_test, y_train, y_test)
print("Linear Regression")
lrB, train_predA, test_predA = linear_regression_bedroom2(X_train, X_test, y_train, y_test)


print("For Bedroom 3")
X_train, X_test, y_train, y_test = train_test_split(features_b3,output_b3, test_size=.1)
print("Random Forest")
rfC, train_predC, test_predC = random_forest_bedroom3(X_train, X_test, y_train, y_test)
print("Linear Regression")
lrC, train_predC, test_predC = linear_regression_bedroom3(X_train, X_test, y_train, y_test)

dump(rfA, './models/rf_bedroom1.joblib')
dump(rfB, './models/rf_bedroom2.joblib')
dump(rfC, './models/rf_bedroom3.joblib')

dump(lrA, './models/lr_bedroom1.joblib')
dump(lrB, './models/lr_bedroom2.joblib')
dump(lrC, './models/lr_bedroom3.joblib')


print(features_b1.columns)
