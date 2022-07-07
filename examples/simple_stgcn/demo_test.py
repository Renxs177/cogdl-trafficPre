import copy
import itertools
import os
from collections import defaultdict, namedtuple
import torch
import torch.nn as nn
from tabulate import tabulate
from cogdl.utils import set_random_seed, tabulate_results
from cogdl.models import build_model
from cogdl.datasets import build_dataset
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper
from cogdl.options import get_default_args
from cogdl.trainer import Trainer
import numpy as np


def output_results(results_dict, tablefmt="github"):
    variant = list(results_dict.keys())[0]
    col_names = ["Variant"] + list(results_dict[variant][-1].keys())
    tab_data = tabulate_results(results_dict)

    print(tabulate(tab_data, headers=col_names, tablefmt=tablefmt))


# 传入参数，加载模型
def train(args):  # noqa: C901
    if isinstance(args.dataset, list):
        args.dataset = args.dataset[0]
    if isinstance(args.model, list):
        args.model = args.model[0]
    if isinstance(args.seed, list):
        args.seed = args.seed[0]
    if isinstance(args.split, list):
        args.split = args.split[0]
    # dataset='cora', model='gcn', seed=1, split=0
    # 设置随机种子
    set_random_seed(args.seed)

    # 要求字符串
    model_name = args.model if isinstance(args.model, str) else args.model.model_name
    dw_name = args.dw if isinstance(args.dw, str) else args.dw.__name__
    mw_name = args.mw if isinstance(args.mw, str) else args.mw.__name__


    # 打印训练关键信息
    print(
        f""" 
|-------------------------------------{'-' * (len(str(args.dataset)) + len(model_name) + len(dw_name) + len(mw_name))}|
    *** Running (`{args.dataset}`, `{model_name}`, `{dw_name}`, `{mw_name}`)
|-------------------------------------{'-' * (
                    len(str(args.dataset)) + len(model_name) + len(dw_name) + len(mw_name))}|"""
    )

    # 按照指定格式 构建数据集
    """
        aa = Namespace(activation='relu', actnn=False, checkpoint_path='./checkpoints/model.pt', cpu=False, cpu_inference=False, 
    dataset=['cora'], devices=[0], distributed=False, dropout=0.5, dw='node_classification_dw', epochs=500, eval_step=1, 
    fp16=False, hidden_size=64, load_emb_path=None, local_rank=0, log_path='.', logger=None, lr=0.01, master_addr='localhost', 
    master_port=13425, max_epoch=None, model=['gcn'], mw='node_classification_mw', n_trials=3, n_warmup_steps=0, no_test=False, 
    norm=None, nstage=1, num_classes=None, num_features=None, num_layers=2, patience=100, progress_bar='epoch', 
    project='cogdl-exp', residual=False, resume_training=False, rp_ratio=1, save_emb_path=None, seed=[1], split=[0],
    unsup=False, use_best_config=False, weight_decay=0)
    """


    # 接受数据集类的实例化
    dataset = build_dataset(args)
    # cogdl.datasets.planetoid_data.CoraDataset
    # 并且 args 已经根据 数据集发生改变了

    # 获取模型封装，数据封装的类
    mw_class = fetch_model_wrapper(args.mw)
    dw_class = fetch_data_wrapper(args.dw)  # node_classification_dw

    if mw_class is None:
        raise NotImplementedError("`model wrapper(--mw)` must be specified.")

    if dw_class is None:
        raise NotImplementedError("`data wrapper(--dw)` must be specified.")

    # 定义 封装的参数
    data_wrapper_args = dict()
    model_wrapper_args = dict()


    # setup data_wrapper
    # 根据数据名称获取类的实例化
    data_wrapper_args['batch_size'] = args.batch_size
    data_wrapper_args['n_his'] = args.n_his
    data_wrapper_args['n_pred'] = args.n_pred
    data_wrapper_args['train_prop'] = args.train_prop
    data_wrapper_args['val_prop'] = args.val_prop
    data_wrapper_args['test_prop'] = args.test_prop
    data_wrapper_args['pred_length'] = args.pred_length

    dataset_wrapper = dw_class(dataset, **data_wrapper_args)
    # cogdl.wrappers.data_wrapper.node_classification.node_classification_dw.FullBatchNodeClfDataWrapper

    args.num_features = dataset.num_features
    if hasattr(dataset, "num_nodes"):
        args.num_nodes = dataset.num_nodes
    if hasattr(dataset, "num_edges"):
        args.num_edges = dataset.num_edges
    if hasattr(dataset, "num_edge"):
        args.num_edge = dataset.num_edge
    if hasattr(dataset, "max_graph_size"):
        args.max_graph_size = dataset.max_graph_size
    if hasattr(dataset, "edge_attr_size"):
        args.edge_attr_size = dataset.edge_attr_size
    else:
        args.edge_attr_size = [0]
    if hasattr(args, "unsup") and args.unsup:
        args.num_classes = args.hidden_size
    else:
        args.num_classes = dataset.num_classes
    if hasattr(dataset.data, "edge_attr") and dataset.data.edge_attr is not None:
        args.num_entities = len(torch.unique(torch.stack(dataset.data.edge_index)))
        args.num_rels = len(torch.unique(dataset.data.edge_attr))


    # setup model
    if isinstance(args.model, nn.Module):
        model = args.model
    else:
        model = build_model(args)

    # specify configs for optimizer
    optimizer_cfg = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_warmup_steps=args.n_warmup_steps,
        epochs=args.epochs,
        batch_size=args.batch_size if hasattr(args, "batch_size") else 0,
    )

    if hasattr(args, "hidden_size"):
        optimizer_cfg["hidden_size"] = args.hidden_size

    # setup model_wrapper
    # 根据模型名称获取类的实例化
    # renxs
    model_wrapper_args['edge_index'] = dataset.data.edge_index
    model_wrapper_args['edge_weight'] = dataset.data.edge_weight
    model_wrapper_args['scaler'] = dataset_wrapper.scaler
    model_wrapper_args['node_ids'] = dataset.data.node_ids
    model_wrapper_args['pred_timestamp'] = dataset_wrapper.get_pre_timestamp()
    if isinstance(args.mw, str) and "embedding" in args.mw:
        model_wrapper = mw_class(model, **model_wrapper_args)
    else:
        model_wrapper = mw_class(model, optimizer_cfg, **model_wrapper_args)

    os.makedirs("./checkpoints", exist_ok=True)

    # setup controller
    trainer = Trainer(
        epochs=args.epochs,
        device_ids=args.devices,
        cpu=args.cpu,
        save_emb_path=args.save_emb_path,
        load_emb_path=args.load_emb_path,
        cpu_inference=args.cpu_inference,
        progress_bar=args.progress_bar,
        distributed_training=args.distributed,
        checkpoint_path=args.checkpoint_path,
        resume_training=args.resume_training,
        patience=args.patience,
        eval_step=args.eval_step,
        logger=args.logger,
        log_path=args.log_path,
        project=args.project,
        no_test=args.no_test,
        nstage=args.nstage,
        actnn=args.actnn,
        fp16=args.fp16,
    )

    # Go!!!
    # 开始训练，传入 模型参数封装和数据集封装
    result = trainer.run(model_wrapper, dataset_wrapper)

    return result


def variant_args_generator(args, variants):
    """Form variants as group with size of num_workers"""
    for idx, variant in enumerate(variants):
        args.dataset, args.model, args.seed, args.split = variant
        yield copy.deepcopy(args)


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    return itertools.starmap(Variant, itertools.product(*items.values()))


def raw_experiment(args):
    # 产生变量 [Variant(dataset='cora', model='gcn', seed=1, split=0)]
    variants = list(gen_variants(dataset=args.dataset, model=args.model, seed=args.seed, split=args.split))

    # defaultdict(<class 'list'>, {})
    results_dict = defaultdict(list)

    # train
    results = []
    for aa in variant_args_generator(args, variants):
        """
        aa = Namespace(activation='relu', actnn=False, checkpoint_path='./checkpoints/model.pt', cpu=False, cpu_inference=False, 
    dataset=['cora'], devices=[0], distributed=False, dropout=0.5, dw='node_classification_dw', epochs=500, eval_step=1, 
    fp16=False, hidden_size=64, load_emb_path=None, local_rank=0, log_path='.', logger=None, lr=0.01, master_addr='localhost', 
    master_port=13425, max_epoch=None, model=['gcn'], mw='node_classification_mw', n_trials=3, n_warmup_steps=0, no_test=False, 
    norm=None, nstage=1, num_classes=None, num_features=None, num_layers=2, patience=100, progress_bar='epoch', 
    project='cogdl-exp', residual=False, resume_training=False, rp_ratio=1, save_emb_path=None, seed=[1], split=[0],
    unsup=False, use_best_config=False, weight_decay=0)
        """
        results.append(train(aa))

    # results = [train(args) for args in variant_args_generator(args, variants)] # [ {'test_acc': 0.794, 'val_acc': 0.788} ]
    # list(zip(variants, results)) [(Variant(dataset='cora', model='gcn', seed=1, split=0), {'test_acc': 0.794, 'val_acc': 0.788})]
    for variant, result in zip(variants, results):
        # 取('cora', 'gcn') ：[ {'test_acc': 0.794, 'val_acc': 0.788} ]
        results_dict[variant[:-2]].append(result)

    tablefmt = "github"
    output_results(results_dict, tablefmt)

    return results_dict


def experiment(dataset, model=None, **kwargs):
    dataset = [dataset]
    model = [model]
    #  获取 Namespace 参数空间
    # 从 wrappers 获取初始化的参数, 继承基础参数设置，并 根据需要添加新的参数
    args = get_default_args(dataset=[str(x) for x in dataset], model=[str(x) for x in model], **kwargs)
    args.dataset = dataset
    args.model = model
    return raw_experiment(args)





"""
Namespace(K=3, actnn=False, batch_size=30, channel_size_list=array([[ 1, 16, 64],[64, 16, 64]]), checkpoint_path='./checkpoints/model.pt', 
cpu=False, cpu_inference=False, dataset=['pems-50'], devices=[0], distributed=False, dw='traffic_prediction_dw', 
epochs=100, eval_step=1, fp16=False, kernel_size=3, load_emb_path=None, local_rank=0, log_path='.', 
logger=None, lr=0.01, master_addr='localhost', master_port=13425, max_epoch=None, model=['stgcn'], 
mw='traffic_prediction_mw', n_his=20, n_pred=5, n_trials=3, n_warmup_steps=0, no_test=False, normalization='sym', 
nstage=1, num_layers=2, num_nodes=50, patience=100, pred_length=288, progress_bar='epoch', project='cogdl-exp', 
resume_training=False, rp_ratio=1, save_emb_path=None, seed=[1], split=[0], test_prop=0.01, train_prop=0.03, 
unsup=False, use_best_config=False, val_prop=0.02, weight_decay=0)
"""

kwargs = {"epochs":100,
          "kernel_size":3,
          "n_his":20,
          "n_pred":5,
          "channel_size_list":np.array([[ 1, 16, 64 ],[64, 16, 64]]),
          "num_layers":2,
          "num_nodes":288,
          "train_prop": 0.7,
          "val_prop": 0.05,
          "test_prop": 0.05,
          "pred_length":288,
          }
# experiment(dataset="pems-50", model="stgcn", **kwargs)



import streamlit as st
import pandas as pd
import os
import datetime as dt
import time
import numpy as np

meta_path = os.path.split(os.path.realpath(__file__))[0] + '/data/pems-stgcn/raw/station_meta_288.csv'
pre_V_path = os.path.split(os.path.realpath(__file__))[0] + '/data/pems-stgcn/stgcn_prediction.csv'
tru_V_path = os.path.split(os.path.realpath(__file__))[0] + '/data/pems-stgcn/stgcn_label.csv'
meta = pd.read_csv(meta_path)
pre_V = pd.read_csv(pre_V_path)
tru_V = pd.read_csv(tru_V_path)


def process_meta(df_meta):
    id2Location = {}
    ID = df_meta["ID"]
    lat = df_meta["Latitude"]
    lon = df_meta["Longitude"]
    for index, i in enumerate(ID):
        id2Location[i] = {'lat': lat[index], 'lon': lon[index]}
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
        demo_data_delta.append(abs(j - tru_V[str(i)][index]))

demo_data = pd.DataFrame()
demo_data["ID"] = demo_data_ID
demo_data["timestamp"] = demo_data_time
demo_data["Latitude"] = demo_data_lat
demo_data["Longitude"] = demo_data_lon
demo_data["pre_speed"] = demo_data_pre_speed
demo_data["tru_speed"] = demo_data_tru_speed
demo_data["delta_speed"] = demo_data_delta
demo_data.to_csv('./data/pems-stgcn/demo_data.csv', index=False)

demo_streamlit = pd.DataFrame()
demo_streamlit["ID"] = demo_data_ID
demo_streamlit["timestamp"] = demo_data_time
demo_streamlit["latitude"] = demo_data_lat
demo_streamlit["longitude"] = demo_data_lon
demo_streamlit["predict_speed"] = demo_data_pre_speed
demo_streamlit.to_csv('./data/pems-stgcn/demo_streamlit.csv', index=False)

meta = pd.read_csv('./data/pems-stgcn/demo_streamlit.csv')
columns = ['ID', 'timestamp', 'latitude', 'longitude', 'predict_speed']


meta_V = pd.read_csv('./data/pems-stgcn/demo_streamlit.csv')
id = meta_V[columns[0]]
timestamp = meta_V[columns[1]]
lat = meta_V[columns[2]]
lon = meta_V[columns[3]]
speed = meta_V[columns[4]]




import streamlit as st
import pandas as pd
from datetime import datetime
from folium.plugins import HeatMap
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import flickrapi
import random
from dotenv import load_dotenv
import os
import urllib

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
    travel_log = pd.read_csv("./data/pems-stgcn/demo_streamlit.csv")
    travel_log["date"] = pd.to_datetime(travel_log["timestamp"])
    return travel_log


def get_pics_from_location(locations_df, size=10):
    """ Get images from flickr using the gps coordinates"""
    api_key = os.getenv("FLICKR_API_KEY")
    api_secret = os.getenv("FLICKR_API_SECRET")
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format="parsed-json")
    urls = []

    for index, row in locations_df.iterrows():
        try:
            photos = flickr.photos.search(
                lat=row["latitude"], lon=row["longitude"], per_page=10, pages=1
            )
            # Get a random image from the set of images
            choice_max = min(size - 1, int(photos["photos"]["total"]))
            selection = random.randint(0, choice_max)
            selected_photo = photos["photos"]["photo"][selection]

            # Compute the url for the image
            url = f"https://live.staticflickr.com/{selected_photo['server']}/{selected_photo['id']}_{selected_photo['secret']}_w.jpg"
            urls.append(url)
        except Exception as e:
            print(e)
            continue
    return urls


@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    """ Download a single file and make its content available as a string"""
    url = (
            "https://raw.githubusercontent.com/nithishr/streamlit-data-viz-demo/main/"
            + path
    )
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


st.title("🌍 Travels Exploration")

travel_data = load_data()

# Calculate the timerange for the slider
min_ts = datetime.strptime(min(travel_data["timestamp"]), '%Y-%m-%d %H:%M:%S')
max_ts = datetime.strptime(max(travel_data["timestamp"]), '%Y-%m-%d %H:%M:%S')

st.sidebar.subheader("Inputs")
min_selection, max_selection = st.sidebar.slider(
    "Timeline", min_value=min_ts, max_value=max_ts, value=[min_ts, max_ts]
)

# Toggles for the feature selection in sidebar
show_heatmap = st.sidebar.checkbox("Show Heatmap")
show_histograms = st.sidebar.checkbox("Show Histograms")
show_images = st.sidebar.checkbox("Show Images")
images_count = st.sidebar.number_input("Images to Show", value=10)
show_detailed_months = st.sidebar.checkbox("Show Detailed Split per Year")
show_code = st.sidebar.checkbox("Show Code")

# Filter Data based on selection
st.write(f"Filtering between {min_selection.date()} & {max_selection.date()}")
travel_data = travel_data[
    (travel_data["date"] >= min_selection) & (travel_data["date"] <= max_selection)
    ]
st.write(f"Data Points: {len(travel_data)}")

# Plot the GPS coordinates on the map
st.map(travel_data)


if show_histograms:
    # Plot the histograms based on the dates of data points
    years = travel_data.groupby(travel_data["date"].dt.year).count().plot(kind="bar")
    years.set_xlabel("Year of Data Points")
    hist_years = years.get_figure()
    st.subheader("Data Split by Year")
    st.pyplot(hist_years)

    months = travel_data.groupby(travel_data["date"].dt.month).count().plot(kind="bar")
    months.set_xlabel("Month of Data Points")
    hist_months = months.get_figure()
    st.subheader("Data Split by Months")
    st.pyplot(hist_months)

    hours = travel_data.groupby(travel_data["data"].dt.hour).count().plot(kind="bar")
    hours.set_xlabel("Hour of Data Points")
    hist_hours = hours.get_figure()
    st.subheader("Data Split by Hours of Day")
    st.pyplot(hist_hours)

if show_detailed_months:
    month_year = (
        travel_data.groupby([travel_data["date"].dt.year, travel_data["date"].dt.month])
            .count()
            .plot(kind="bar")
    )
    month_year.set_xlabel("Month, Year of Data Points")
    hist_month_year = month_year.get_figure()
    st.subheader("Data Split by Month, Year")
    st.pyplot(hist_month_year)

if show_heatmap:
    # Plot the heatmap using folium. It is resource intensive!
    # Set the map to center around Munich, Germany (48.1351, 11.5820)
    map_heatmap = folium.Map(location=[48.1351, 11.5820], zoom_start=11)

    # Filter the DF for columns, then remove NaNs
    heat_df = travel_data[["latitude", "longitude"]]
    heat_df = heat_df.dropna(axis=0, subset=["latitude", "longitude"])

    # List comprehension to make list of lists
    heat_data = [
        [row["latitude"], row["longitude"]] for index, row in heat_df.iterrows()
    ]

    # Plot it on the map
    HeatMap(heat_data).add_to(map_heatmap)

    # Display the map using the community component
    st.subheader("Heatmap")
    folium_static(map_heatmap)

if show_images:
    # Show the images from Flickr's public images
    st.subheader("Image Highlights")
    sample_data = travel_data.sample(n=images_count)
    urls = get_pics_from_location(sample_data, images_count)
    st.image(urls, width=200)

if show_code:
    st.code(get_file_content_as_string("travel_viz.py"))

