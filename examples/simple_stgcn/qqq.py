import streamlit as st
import pandas as pd
import numpy as np

# 安装完成，没出什么大毛病

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz')


# 缓存机制，再一个函数前使用@st.cache，用户在第二次使用时，可以直接读取。当然，前提是传进去的参数要是一样的。
@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# 创建一个文本框，让用户知道数据正在加载
data_load_state = st.text('Loading data...')
# 展示数据的前1000行
data = load_data(1000)
# 告诉用户数据已经加载成功
data_load_state.text('Done! (using st.cache)')

# 添加一个单选框，是否要显示表格里面的内容
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
# 绘制柱状图
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)
