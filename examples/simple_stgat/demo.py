import streamlit as st
import pandas as pd
import os
import datetime as dt
import time
import numpy as np


meta_path = os.path.split(os.path.realpath(__file__))[0]+r'\data\pems-stgcn\raw\station_meta_50.csv'
pre_V_path = os.path.split(os.path.realpath(__file__))[0]+r'\data\pems-stgcn\stgcn_prediction.csv'
tru_V_path = os.path.split(os.path.realpath(__file__))[0]+r'\data\pems-stgcn\stgcn_label.csv'
meta = pd.read_csv(meta_path)
pre_V = pd.read_csv(pre_V_path)
tru_V = pd.read_csv(tru_V_path)
def process_meta(df_meta):
    id2Location = {}
    ID = df_meta["ID"]
    lat = df_meta["Latitude"]
    lon = df_meta["Longitude"]
    for index, i in enumerate(ID):
        id2Location[i] = {'lat':lat[index] , 'lon':lon[index]}
    return id2Location
id2Location = process_meta(meta)
def get_location(ID):
    return [id2Location[ID]['lat'], id2Location[ID]['lon']]
time_timestamp = list(tru_V['timestamp'])
demo_data_ID = []
demo_data_time = []
demo_data_lat = []
demo_data_lon = []
demo_data_pre_speed = []
demo_data_tru_speed = []
demo_data_delta = []

for i in meta["ID"]:
    for index, j in enumerate(pre_V[str(i)]):
        demo_data_ID.append(i)
        s = time_timestamp[index]
        a = dt.datetime.strptime(s, '%m/%d/%Y %H:%M:%S')
        b = dt.datetime.strftime(a, '%Y-%m-%d %H:%M:%S')
        demo_data_time.append(b)
        demo_data_lat.append(get_location(i)[0])
        demo_data_lon.append(get_location(i)[1])
        demo_data_pre_speed.append(j)
        demo_data_tru_speed.append(tru_V[str(i)][index])
        demo_data_delta.append(abs(j-tru_V[str(i)][index]))
demo_data = pd.DataFrame()
demo_data["ID"] = demo_data_ID
demo_data["timestamp"] = demo_data_time
demo_data["Latitude"] = demo_data_lat
demo_data["Longitude"] = demo_data_lon
demo_data["pre_speed"] = demo_data_pre_speed
demo_data["tru_speed"] = demo_data_tru_speed
demo_data["delta_speed"] = demo_data_delta
demo_data.to_csv('./data/pems-stgcn/demo_data.csv',index=False)




# meta = pd.read_csv(meta_path)
# columns = ['ID','Latitude','Longitude']
# meta_V = pd.read_csv(meta_path)
# id = meta_V[columns[0]]
# lat = meta_V[columns[1]]
# lon = meta_V[columns[2]]
# map_df = pd.DataFrame()
# map_df['lat'] = lat
# map_df['lon'] = lon
# map_df['lat'].fillna(30, inplace=True)
# map_df['lon'].fillna(-118, inplace=True)
# st.map(map_df)