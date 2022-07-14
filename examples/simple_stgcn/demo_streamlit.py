import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import flickrapi
import random
from dotenv import load_dotenv
import os
import urllib
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap



# travel_log = pd.read_csv("F:/CogDL/cogdl-for-trafficPre/examples/simple_stgcn/data/pems-stgcn/demo_streamlit.csv")
# travel_log = pd.read_csv("/home/xiangsheng/zp/cogdl-trafficPre/examples/simple_stgcn/data/pems-stgcn/demo_streamlit.csv")
travel_log = pd.read_csv("./data/pems-stgcn/demo_streamlit.csv")
timestamp = travel_log['timestamp']
timestamp_set = set(timestamp)
new_timeatamp = list(timestamp_set)


def get_timestamp(date):
    return datetime.strptime(date,"%Y-%m-%d %H:%M:%S").timestamp()
new_sort_timeatamp=sorted(new_timeatamp,key=lambda date: get_timestamp(date))


data_move = []
for i in new_sort_timeatamp:
    df = travel_log[(travel_log['timestamp'] == i)]
    num = df.shape[0]
    lat = np.array(df["latitude"][0:num])
    lon = np.array(df["longitude"][0:num])
    speed = np.array(df["predict_speed"][0:num], dtype=float)
    data1 = [[lat[i], lon[i], speed[i]] for i in range(num)]
    data_move.append(data1)

load_dotenv()

# set page layout
st.set_page_config(
    page_title="Travel Exploration",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache
def load_data():
    """ Load the cleaned data with latitudes, longitudes & timestamps """
    # travel_log = pd.read_csv("F:/CogDL/cogdl-for-trafficPre/examples/simple_stgcn/data/pems-stgcn/demo_streamlit.csv")
    # travel_log = pd.read_csv("/home/xiangsheng/zp/cogdl-trafficPre/examples/simple_stgcn/data/pems-stgcn/demo_streamlit.csv")
    travel_log = pd.read_csv("./data/pems-stgcn/demo_streamlit.csv")
    travel_log["date"] = pd.to_datetime(travel_log["timestamp"])
    return travel_log


st.title("🌍 Travels Exploration")

travel_data = load_data()

# # Calculate the timerange for the slider
# min_ts = datetime.strptime(min(travel_data["timestamp"]), '%Y-%m-%d %H:%M:%S')
# max_ts = datetime.strptime(max(travel_data["timestamp"]), '%Y-%m-%d %H:%M:%S')
#
# st.sidebar.subheader("Inputs")
# min_selection, max_selection = st.sidebar.slider(
#     "Timeline", min_value=min_ts, max_value=max_ts, value=[min_ts, max_ts]
# )
#
# # Toggles for the feature selection in sidebar
# show_heatmap = st.sidebar.checkbox("Show Heatmap")
#
#
# # Filter Data based on selection
# st.write(f"Filtering between {min_selection.date()} & {max_selection.date()}")
# travel_data = travel_data[
#     (travel_data["date"] >= min_selection) & (travel_data["date"] <= max_selection)
#     ]
# st.write(f"Data Points: {len(travel_data)}")
# # Plot the GPS coordinates on the map
# st.map(travel_data, zoom=6, use_container_width = True)



# 绘制Map，中心经纬度[32, 120],开始缩放程度是5倍
# tiles用于指示地图风格 主要用到的风格：Stamen Toner(黑白)，默认(OSM)，Stamen Terrain(地形图)
map_osm = folium.Map(location=[34, -118], zoom_start=10,tiles='Stamen Toner',control_scale=True)

map_osm.add_child(folium.ClickForMarker()) # 标记图标



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

hm = folium.plugins.HeatMapWithTime(data_move, index=new_sort_timeatamp, name="Speed Map", radius=15, min_opacity=0.5,
                 max_opacity=0.6, scale_radius=False, gradient={.1: 'blue', .2: 'lime', .3:'red',.4:'black',.5:'pink',.6:'green', 1: 'yellow'},
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



