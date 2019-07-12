import pandas as pd
import numpy as np

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

binned=np.unique(df_new['binned'])
city=np.unique(df_new['City'])
property_config = np.unique(df_new['Property Config'])

list_entries=[]
for b in binned:
    for c in city:
        for p in property_config:
            df_temp = df_new.loc[df_new['Property Config']==p]
            df_temp = df_temp.loc[df_temp['City']==c]
            df_temp= df_temp.loc[df_temp['binned']==b]
            mode= df_temp['Kitchen Shape'].mode()
            if(len(mode)==0):
                mode='l-shaped'
            else:
                mode=mode[0]
            d_temp = {'property_config':p,'city':c,'bin':b,'kitchen_shape':mode}
            list_entries.append(d_temp)
df_entries = pd.DataFrame(list_entries)
df_entries.to_csv('../models/kitchen_shape.csv')
