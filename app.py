from flask import Flask, jsonify, request
import pickle
from joblib import dump, load
import numpy as np
import timeit
import pandas as pd

app = Flask(__name__)

#Setting up model pre-requisites
columns = ['Property Config_1 BHK', 'Property Config_2 BHK',
       'Property Config_3 BHK', 'Property Config_4 BHK', 'binned_1',
       'binned_2', 'binned_3', 'binned_4', 'City_Bengaluru', 'City_Chennai',
       'City_Hyderabad', 'City_Kolkata', 'City_Mumbai', 'City_NCR',
       'City_Pune']
feature_index = {}
ncr_cities= set(['Gurgaon', 'Ghaziabad','New Delhi','Noida'])
for i in range(len(columns)):
    feature_index[columns[i].split("_")[-1]]=i

print(feature_index)
print("Loading models")
start= timeit.timeit()
rfA = load('./models/rfA.joblib')
rfB = load('./models/rfB.joblib')
rfC = load('./models/rfC.joblib')

lrA = load('./models/lrA.joblib')
lrB = load('./models/lrB.joblib')
lrC = load('./models/lrC.joblib')
end= timeit.timeit()


#Loading Models for bedroom
columns_bedroom =['Property Config_1 BHK', 'Property Config_2 BHK',
       'Property Config_3 BHK', 'Property Config_4 BHK', 'binned_1',
       'binned_2', 'binned_3', 'binned_4', 'City_Bengaluru', 'City_Chennai',
       'City_Hyderabad', 'City_Kolkata', 'City_Mumbai', 'City_NCR',
       'City_Pune']
feature_index_bedroom={}
for i in range(len(columns_bedroom)):
    feature_index_bedroom[columns_bedroom[i].split("_")[-1]]=i

start =timeit.timeit()
rf_bedroom1 = load('./models/rf_bedroom1.joblib')
rf_bedroom2 = load('./models/rf_bedroom2.joblib')
rf_bedroom3= load('./models/rf_bedroom3.joblib')

lr_bedroom1 = load('./models/lr_bedroom1.joblib')
lr_bedroom2 = load('./models/lr_bedroom2.joblib')
lr_bedroom3= load('./models/lr_bedroom3.joblib')
end=timeit.timeit()

## Loading csv to compute kitchen shape
kitchen_shape_entries = pd.read_csv('./kitchen_shape.csv')



#Function to return bin based on floorplan size
def get_bin(floorplan_size):
    bins = [287,1055,1233,1585,4813]
    if(floorplan_size<bins[1]):
        return 1
    elif(floorplan_size>=bins[1] and floorplan_size<bins[2]):
        return 2
    elif(floorplan_size>=bins[2] and floorplan_size<bins[3]):
        return 3
    else:
        return 4


def get_city(data):
    city=data['city']
    ncr=set(['Gurgaon', 'Ghaziabad','New Delhi','Noida'])
    mumbai = set(['Thane','Mumbai'])
    if(city in ncr):
        return "NCR"
    elif (city in mumbai):
        return "Mumbai"
    else:
        return city

def get_kitchen_shape(data):
    city=get_city(data)
    bin=get_bin(data['floorplan_size'])
    property_config= data['property_config']

    temp = kitchen_shape_entries.loc[(kitchen_shape_entries['property_config']==property_config) & (kitchen_shape_entries['bin']==bin) & (kitchen_shape_entries['city']==city)]
    return temp['kitchen_shape'].iloc[0]

#Function to return input vector for model
def get_input_vector_kitchen(data):
    model_input = np.zeros(len(columns))
    city = get_city(data)
    bin=get_bin(data['floorplan_size'])
    property_config = data['property_config']
    if(property_config=='1 BHK'):
        model_input[feature_index[property_config]]=1
    elif(property_config=='2 BHK'):
        model_input[feature_index[property_config]]=1
    elif(property_config=='3 BHK'):
        model_input[feature_index[property_config]]=1
    else:
        model_input[feature_index['4 BHK']]=1
    #Setting input field for city
    if(city in ncr_cities):
        model_input[feature_index['NCR']]=1
    else:
        model_input[feature_index[city]]=1
    #Setting bin
    model_input[feature_index[str(bin)]]=1
    return model_input

def get_input_vector_bedroom(data):
    model_input = np.zeros(len(columns_bedroom))
    city = get_city(data)
    bin=get_bin(data['floorplan_size'])
    property_config = data['property_config']
    if(property_config=='1 BHK'):
        model_input[feature_index_bedroom[property_config]]=1
    elif(property_config=='2 BHK'):
        model_input[feature_index_bedroom[property_config]]=1
    elif(property_config=='3 BHK'):
        model_input[feature_index_bedroom[property_config]]=1
    else:
        model_input[feature_index_bedroom['4 BHK']]=1
    #Setting input field for city
    if(city in ncr_cities):
        model_input[feature_index_bedroom['NCR']]=1
    else:
        model_input[feature_index_bedroom[city]]=1
    #Setting bin
    model_input[feature_index_bedroom[str(bin)]]=1
    return model_input

def predict_kitchen(data):
    list_attributes=['city','floorplan_size','property_config']
    for i in list_attributes:
        if(i not in data):
            return "Mandatory Params missing"

    kitchen_shape = get_kitchen_shape(data)

    model_input = get_input_vector_kitchen(data)
    wallA = lrA.predict(model_input.reshape(1,-1))
    wallA=wallA[0]
    wallB = lrB.predict(model_input.reshape(1,-1))
    wallB=wallB[0]
    wallC = lrC.predict(model_input.reshape(1,-1))
    wallC=wallC[0]
    return {'wall_a':wallA,'wall_b':wallB,'wall_c':wallC,'kitchen_shape':kitchen_shape}

def predict_bedroom(data):
    list_attributes=['city','floorplan_size','property_config']
    for i in list_attributes:
        if(i not in data):
            return "Mandatory Params missing"


    property_config = data['property_config']

    model_input = get_input_vector_bedroom(data)
    wallA=0
    wallB=0
    wallC=0
    wallA = lr_bedroom1.predict(model_input.reshape(1,-1))
    wallA=wallA[0][0]
    if(property_config!='1 BHK'):
        wallB = lr_bedroom2.predict(model_input.reshape(1,-1))
        wallB=wallB[0][0]
    if(property_config!='1 BHK' and property_config!='2 BHK'):
        wallC = lr_bedroom3.predict(model_input.reshape(1,-1))
        wallC=wallC[0][0]
    return {'bedroom_1':wallA,'bedroom_2':wallB,'bedroom_3':wallC}


@app.route("/predict",methods=['POST'])
def predict():
    try:
        data  = request.get_json()
        list_attributes=['city','floorplan_size','property_config']
        for i in list_attributes:
            if(i not in data):
                return jsonify({'msg':"Mandatory Params missing"})
        kitchen_output= predict_kitchen(data)
        bedroom_output= predict_bedroom(data)


        return jsonify({'kitchen':kitchen_output,'bedroom':bedroom_output})
    except:
        return jsonify({'msg':"Unexpected Input or Internal Server error"})








if(__name__=='__main__'):
    app.run(debug=True,threaded=True)
