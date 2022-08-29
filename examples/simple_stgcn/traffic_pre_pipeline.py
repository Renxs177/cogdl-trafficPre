import sys
sys.path.append("./")

import sys, os
add_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(add_path)

sys.path.append(r'C:\Users\user\Desktop\Aotu_cogdl\cogdl-trafficPre-master')

import sys, os
add_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(add_path)

from cogdl import utils
from examples import experiment
import os.path as osp
import os
import numpy as np
import gzip
import time
import requests
from bs4 import BeautifulSoup
import re
from lxml import etree
import json
import shutil
import pandas as pd
import os
import datetime as dt
import numpy as np
import datetime
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import streamlit as st
import streamlit.components.v1 as components
import flickrapi
import random
from dotenv import load_dotenv
import os
import urllib
import numpy as np
import pandas as pd
import folium


def get_timestamp(date):
    return datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp()


def get_location(ID):
    return [id2Location[ID]['lat'], id2Location[ID]['lon']]


def process_meta(df_meta):
    id2Location = {}
    ID = df_meta["ID"]
    lat = df_meta["Latitude"]
    lon = df_meta["Longitude"]
    for index, i in enumerate(ID):
        id2Location[i] = {'lat': lat[index], 'lon': lon[index]}
    return id2Location


def un_gz(file_name):
    # 获取文件的名称，去掉后缀名
    f_name = file_name.replace(".gz", "")
    # 开始解压
    g_file = gzip.GzipFile(file_name)
    # 读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


def files_exist(files):
    # 返回一个布尔变量
    return all([osp.exists(f) for f in files])


if __name__ == "__main__":


    bbb = 0
    kwargs = {"epochs": 1,
              "kernel_size": 3,
              "n_his": 20,
              "n_pred": 1,
              "channel_size_list": np.array([[1, 16, 64], [64, 64, 64], [64, 16, 64]]),
              "num_layers": 3,
              "num_nodes": 288,
              "train_prop": 0.8,
              "val_prop": 0.1,
              "test_prop": 0.1,
              "pred_length": 288,
              }


    meta_path = os.path.split(os.path.realpath(__file__))[0] + r'\data\pems-stgcn\raw\station_meta_288.csv'
    print(os.path.abspath(__file__))
    raw_path = os.path.split(os.path.realpath(__file__))[0] + r'\data\pems-stgcn\raw'
    process_path = os.path.split(os.path.realpath(__file__))[0] + r'\data\pems-stgcn\processed'
    print(raw_path)

    #
    # latest_data_name = "None"
    # s = requests.Session()
    # username = "Renxs177@gmail.com"
    # password = "!2P#&~3bookr"
    # login_url = "https://pems.dot.ca.gov"
    # # search_url= "https://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=7&submit=Submit"
    # search_url = "https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=station_5min&returnformat=text"
    # url_meta = "https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=meta&returnformat=text"
    # result = s.post(url=login_url, data={"username": username, "password": password, "commit": "login"})
    # print(result.status_code)
    # print(result.cookies)
    # page = s.get(search_url)
    # soup = BeautifulSoup(page.content, "lxml")
    # print(soup)
    # list_url = soup.text
    # print(list_url)
    #
    #
    # result_json = json.loads(list_url)
    # print(result_json)
    # data_month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    #               "November", "December"]
    # detail = "detail"
    # """
    # {"file_name":"d07_text_station_5min_2021_01_01.txt.gz","file_id":"409509","bytes":"29,673,928","url":"\/?download=409509&dnode=Clearinghouse"}
    # "detail":{"district":"7","month":12,"year":2021,"date":"2021","data_set":"station_5min"}}
    # """
    #
    # """
    # 7天预测3天
    # """
    #
    # ### 下载最新一天的数据
    # latest_data_url = ""
    # for month in data_month[::-1]:
    #     if month in result_json['data']:
    #         latest_data_url = 'https://pems.dot.ca.gov' + result_json['data'][month][-1]["url"]
    #         latest_data_name = result_json['data'][month][-1]["file_name"]
    #         latest_data = result_json['data'][month][-1]["file_name"].split('.')[0].split('5min_')[1]
    #         break
    # print(latest_data_url)
    # print(latest_data_name)
    # print(latest_data)

    # """
    #
    # https://pems.dot.ca.gov/?download=430913&amp;dnode;=Clearinghouse
    # https://pems.dot.ca.gov/   +   ?download=442594&dnode=Clearinghouse
    #
    # https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=station_5min&returnformat=text
    #
    # https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=station_hour&returnformat=text
    #
    # https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=station_day&returnformat=text
    #
    # https://pems.dot.ca.gov/?srq=clearinghouse&district_id=7&geotag=null&yy=2022&type=meta&returnformat=text
    #
    # """


    latest_data_url = "https://down.wss.show/mt36wnf/8/yo/8yoxmt36wnf?cdn_sign=1659625360-66-0-35778ab0c7692c94bf2459add6c866a5&exp=1200&response-content-disposition=attachment%3B%20filename%3D%22d07_text_station_5min_2022_07_31.txt.gz%22%3B%20filename%2A%3Dutf-8%27%27d07_text_station_5min_2022_07_31.txt.gz"
    meta_data_url = "https://down.wss.show/tftoq18/8/yq/8yqrtftoq18?cdn_sign=1659626090-80-0-ed58eb2d57251c3ce7102fddce0583d7&exp=240&response-content-disposition=attachment%3B%20filename%3D%22d07_text_meta_2022_03_12.txt%22%3B%20filename%2A%3Dutf-8%27%27d07_text_meta_2022_03_12.txt"
    latest_data_name = "hahah"
    if osp.exists(latest_data_name):
        print("没有更新")
        time.sleep(1000)
    else:
        pass
        latest_data_name = raw_path + "/d07_text_station_5min_2022_08_10.txt.gz"
        # un_gz('./d07_text_station_5min_2022_07_31.txt.gz')
        # experiment(dataset="pems-50", model="stgcn", resume_training=True, **kwargs)

        # try:
        #     # if osp.exists(raw_path):
        #     #     shutil.rmtree(raw_path)
        #     #     makedirs(raw_path)
        #     # else:
        #     #     makedirs(raw_path)
        #     # if osp.exists(process_path):
        #     #     shutil.rmtree(process_path)
        #     # download_url(latest_data_url, raw_path, name="d07_text_station_5min_2022_07_29.txt.gz")
        #     # download_url(meta_data_url, raw_path, name="d07_text_meta.txt")
        #     latest_data_name = raw_path + "/d07_text_station_5min_2022_07_09.txt.gz"
        #     # un_gz('./d07_text_station_5min_2022_07_31.txt.gz')
        #     experiment(dataset="pems-stgcn", model="stgcn", resume_training=True, **kwargs)
        #     experiment(dataset="pems-stgat", model="stgat", resume_training=True, **kwargs)
        # except:
        #     pass


    # pre_V_path = os.path.split(os.path.realpath(__file__))[0] + r'\data\pems-stgcn\stgcn_prediction.csv'
    pre_V_path = os.path.split(os.path.realpath(__file__))[0] + r'\data\pems-stgcn\20220802_truth.csv'
    meta_path = os.path.split(os.path.realpath(__file__))[0] + r'\data\pems-stgcn\station_meta_288.csv'
    meta = pd.read_csv(meta_path)
    pre_V = pd.read_csv(pre_V_path)


    id2Location = process_meta(meta)


    time_timestamp = list(pre_V['timestamp'])
    demo_data_ID = []
    demo_data_time = []
    demo_data_lat = []
    demo_data_lon = []
    demo_data_pre_speed = []

    for i in meta["ID"]:
        for index, j in enumerate(pre_V[str(i)]):
            demo_data_ID.append(i)
            s = time_timestamp[index]
            a = dt.datetime.strptime(s, '%Y/%m/%d %H:%M')
            b = dt.datetime.strftime(a, '%Y-%m-%d %H:%M:%S')
            demo_data_time.append(b)
            demo_data_lat.append(get_location(i)[0])
            demo_data_lon.append(get_location(i)[1])
            demo_data_pre_speed.append(j)
    demo_data = pd.DataFrame()
    demo_data["ID"] = demo_data_ID
    demo_data["timestamp"] = demo_data_time
    demo_data["Latitude"] = demo_data_lat
    demo_data["Longitude"] = demo_data_lon
    maxpp = max(demo_data_pre_speed)
    minpp = min(demo_data_pre_speed)
    for i in range(len(demo_data_pre_speed)):
        xxx = (demo_data_pre_speed[i]-minpp)/(maxpp-minpp)
        # demo_data_pre_speed[i] = (xxx**(1/10))*1000
    demo_data["predict_speed"] = demo_data_pre_speed
    demo_data.to_csv(os.path.split(os.path.realpath(__file__))[0] + r'\data\pems-stgcn\demo_data.csv', index=False)



    # travel_log = pd.read_csv(os.path.split(os.path.realpath(__file__))[0] + r'\data\pems-stgcn\demo_data.csv')
    travel_log = demo_data
    timestamp = travel_log['timestamp']
    timestamp_set = set(timestamp)
    new_timeatamp = list(timestamp_set)

    new_sort_timeatamp = sorted(new_timeatamp, key=lambda date: get_timestamp(date))

    data_move = []
    for i in new_sort_timeatamp:
        df = travel_log[(travel_log['timestamp'] == i)]
        num = df.shape[0]
        # 获取纬度
        lat = np.array(df["Latitude"][0:num])
        # 获取经度
        lon = np.array(df["Longitude"][0:num])
        # 获取PM2.5，转化为numpy浮点型
        speed = np.array(df["predict_speed"][0:num], dtype=float)
        # 将数据制作成[lats, lons, weights]的形式
        data1 = [[lat[i], lon[i], speed[i]] for i in range(num)]
        data_move.append(data1)

    load_dotenv()

    # set page layout
    dsds = ['eqw','eqweqw','eqweqwe']
    st.set_page_config(
        page_title="PeMS traffic predict by CogDl",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    @st.cache
    def load_data():
        """ Load the cleaned data with latitudes, longitudes & timestamps """
        travel_log = pd.read_csv(
            r'C:\Users\user\Desktop\Aotu_cogdl\cogdl-trafficPre-master\examples\simple_stgcn\data\pems-stgcn\demo_data.csv')
        # travel_log = pd.read_csv("/home/xiangsheng/zp/cogdl-trafficPre/examples/simple_stgcn/data/pems-stgcn/demo_streamlit.csv")
        # travel_log = pd.read_csv("./data/pems-stgcn/demo_streamlit.csv")
        travel_log["date"] = pd.to_datetime(travel_log["timestamp"])
        return travel_log


    st.title("🌍 PeMS traffic predict by CogDl"  +random.choice(dsds))

    travel_data = load_data()
    # 绘制Map，中心经纬度[32, 120],开始缩放程度是5倍
    # tiles用于指示地图风格 主要用到的风格：Stamen Toner(黑白)，默认(OSM)，Stamen Terrain(地形图)
    bbb += 1
    tii = ["Stamen Toner", "Stamen Terrain"]
    print(bbb)

    map_osm = folium.Map(location=[34, -118], zoom_start=10, tiles="Stamen Toner", control_scale=True)

    # map_osm.add_child(folium.ClickForMarker())  # 标记图标

    # 将热力图添加到前面建立的map里
    # 可设置热力图的颜色，0.4表示的数据的40%分位数。radius=5可设置热力斑块的大小。

    """
        def __init__(self, data, index=None, name=None, radius=15, min_opacity=0,
                     max_opacity=0.6, scale_radius=False, gradient=None,
                     use_local_extrema=False, auto_play=False,
                     display_index=True, index_steps=1, min_speed=0.1,
                     max_speed=10, speed_step=0.1, position='bottomleft',
                     overlay=True, control=True, show=True):
    """

    # scale_radius=False
    # min_opacity max_opacity 热图的最小、大不透明度

    # gradient  将点密度值与颜色匹配。颜色可以是名称（“红色”），RGB值（'RGB（255,0,0）'）或十六进制数（'FF0000'）。
    # 可设置热力图的颜色，0.4表示的数据的40%分位数。radius=5可设置热力斑块的大小。

    # use_local_extrema 定义热图是否使用从输入数据中找到的全局极值集或局部极值（当前显示视图的最大值和最小值）。
    # auto_play 跨时间 自动播放动画
    # display_index 在时间控件中显示索引（通常为时间）。与 index 对应
    # index_steps 播放速度的单位步长
    # position 时间滑块的位置字符串。格式：“下/上”+“左/右”。
    # overlay 将层添加为可选覆盖（True）或基础层（False）。
    # control 图层是否包含在图层控制中。
    # show 是否在开口处显示该层（仅适用于覆盖层）。

    hm = folium.plugins.HeatMapWithTime(data_move, index=new_sort_timeatamp, name="Speed Map", radius=15,
                                        min_opacity=0.5,
                                        max_opacity=0.6, scale_radius=False,
                                        gradient={.0001: 'blue', .2: 'lime', .3: 'red', .4: 'black', .5: 'pink',
                                                  .6: 'green', .7: 'blue', .8:'black',.9:'red',1: 'yellow'},
                                        use_local_extrema=True, auto_play=False,
                                        display_index=True, index_steps=1, min_speed=1,
                                        max_speed=100, speed_step=1, position='bottomleft',
                                        overlay=True, control=True, show=True)
    hm.add_to(map_osm)
    st.subheader("Heatmap")
    fig = folium.Figure().add_child(map_osm)
    components.html(fig.render(), height=800, width=1600)
    file_path = r"./demo_traffic.html"
    hm.save(file_path)
    time.sleep(120)

    st.experimental_rerun()