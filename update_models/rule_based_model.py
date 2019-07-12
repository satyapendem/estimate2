#!/usr/bin/env python
# coding: utf-8

# In[15]:



import pandas as pd
import numpy as np
import math


# In[16]:


#provide the file path to input
df = pd.read_csv('../input_csv/input_csv.csv')
rules = pd.read_csv('../util_csv/all_rules.csv',header=0)

#removing very high floor plan sizes( Only 1 sample)
df=df[df['Floorplan Size']<=5000]


# In[17]:


#Binning by quantiles
bins = [287,1055,1233,1585,4813]
df['binned'] = pd.cut(df['Floorplan Size'],bins=bins,labels=[1,2,3,4])
df_new = pd.DataFrame()
df_new['Property Config']= df['Property Config']
#df_new['Kitchen Shape'] = df['Kitchen Shape']
df_new['City'] =df['City']
df_new['Floorplan Size']= df['Floorplan Size']
df_new['Bedroom 1']=df['Bedroom 1']
df_new['Bedroom 2']=df['Bedroom 2']
df_new['Bedroom 3']=df['Bedroom 3']
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


# In[ ]:





# In[20]:


def get_result(df_new, target_tuple):
    df = df_new[df_new['Property Config']==target_tuple['Property Config']]
    #df= df[df['Kitchen Shape']== target_tuple['Kitchen Shape']]
    df= df[df['City']== target_tuple['City']]
    df= df[df['binned']== target_tuple['binned']]
    bedroom1 =df['Bedroom 1'].dropna().mean()
    bedroom2 =df['Bedroom 2'].dropna().mean()
    bedroom3 =df['Bedroom 3'].dropna().mean()

    wall_a =  df['Wall A'].dropna().mean()
    wall_b =  df['Wall B'].dropna().mean()
    wall_c =  df['Wall C'].dropna().mean()

    if(math.isnan(bedroom1)):
        bedroom1=df_new['Bedroom 1'].mean()
    if(math.isnan(bedroom2)):
        bedroom2=df_new['Bedroom 2'].mean()
    if(math.isnan(bedroom3)):
        bedroom3=df_new['Bedroom 3'].mean()

    if(target_tuple['Property Config']=='1 BHK'):
        bedroom2=0
        bedroom3=0
    if(target_tuple['Property Config']=='2 BHK'):
        bedroom3=0

    if(math.isnan(wall_a)):
        wall_a=df_new['Wall A'].mean()
    if(math.isnan(wall_b)):
        wall_b=df_new['Wall B'].mean()
    if(math.isnan(wall_c)):
        wall_c=df_new['Wall C'].mean()

    #print( "wallB predicted",wallB)
    return  bedroom1, bedroom2, bedroom3, wall_a, wall_b, wall_c


# In[7]:





# In[21]:


bed1=[]
bed2=[]
bed3=[]
wall_a=[]
wall_b=[]
wall_c=[]
for i in range(len(rules)):
    b1,b2,b3, wa, wb, wc = get_result(df_new, rules.iloc[i])
    bed1.append(b1)
    bed2.append(b2)
    bed3.append(b3)
    wall_a.append(wa)
    wall_b.append(wb)
    wall_c.append(wc)


# In[23]:


res_df= pd.DataFrame()
res_df['property_config']=rules['Property Config']
res_df['city'] = rules['City']
res_df['bin']= rules['binned']
res_df['bedroom_1']=bed1
res_df['bedroom_2']=bed2
res_df['bedroom_3']=bed3
res_df['wall_a']=wall_a
res_df['wall_b']=wall_b
res_df['wall_c']=wall_c


# In[27]:


res_df.to_csv('../models/rule_based_classifier.csv')


# In[ ]:
