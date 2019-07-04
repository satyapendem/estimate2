import pandas as pd
import numpy as np
import pickle
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
df_new['binned']=df['binned']
df_new.loc[df['City']=='Thane',"City"]='Mumbai'
df_new.loc[df['City']=='Thane',"City"]='Mumbai'
df_new.loc[df['City'].isin(['Gurgaon', 'Ghaziabad','New Delhi','Noida']),"City"]='NCR'
df_new.loc[df['City']=='Chennai',"City"]='Chennai'
df_new.loc[df['City']=='Hyderabad',"City"]='Hyderabad'

features = pd.get_dummies(df_new[['Property Config','binned','City']])
output=df_new[['Kitchen Shape']]
output.loc[output['Kitchen Shape']=='l-shaped','Kitchen Shape']=0
output.loc[output['Kitchen Shape']=='parallel','Kitchen Shape']=1
output.loc[output['Kitchen Shape']=='straight','Kitchen Shape']=2
output.loc[output['Kitchen Shape']=='u-shaped','Kitchen Shape']=3



def random_forest(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators= 12,min_samples_split= 10,min_samples_leaf= 4,bootstrap= True)
    rf.fit(x_train,y_train)
    predictions  =  rf.predict(x_test)
    print("train accuracy",accuracy_score(rf.predict(x_train),y_train),"%")
    print("test accuracy",accuracy_score(predictions,y_test),"%")
    return rf,rf.predict(x_train),predictions

X_train, X_test, y_train, y_test = train_test_split(features,output, test_size=.1)
rf, train_pred, test_pred = random_forest(X_train, X_test, y_train, y_test)

print(test_pred)
print(X_train.columns)

print(rf.feature_importances_)

dump(rf,"./models/model_kitchen_shape.joblib")
