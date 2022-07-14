![CogDL](./docs/source/_static/cogdl-logo.png)
===

[![PyPI Latest Release](https://badge.fury.io/py/cogdl.svg)](https://pypi.org/project/cogdl/)
[![Build Status](https://travis-ci.org/THUDM/cogdl.svg?branch=master)](https://travis-ci.org/THUDM/cogdl)
[![Documentation Status](https://readthedocs.org/projects/cogdl/badge/?version=latest)](https://cogdl.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/cogdl)](https://pepy.tech/project/cogdl)
[![Coverage Status](https://coveralls.io/repos/github/THUDM/cogdl/badge.svg?branch=master)](https://coveralls.io/github/THUDM/cogdl?branch=master)
[![License](https://img.shields.io/github/license/thudm/cogdl)](https://github.com/THUDM/cogdl/blob/master/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

**[Homepage](https://cogdl.ai)** | **[Paper](https://arxiv.org/abs/2103.00959)** | **[100 GNN papers](./gnn_papers.md)** | **[Leaderboards](./results.md)** | **[Documentation](https://cogdl.readthedocs.io)** | **[Datasets](./cogdl/datasets/README.md)** | **[Join our Slack](https://join.slack.com/t/cogdl/shared_invite/zt-b9b4a49j-2aMB035qZKxvjV4vqf0hEg)** | **[中文](./README_CN.md)**

CogDL is a graph deep learning toolkit that allows researchers and developers to easily train and compare baseline or customized models for node classification, graph classification, and other important tasks in the graph domain. 

We summarize the contributions of CogDL as follows:

- **Efficiency**: CogDL utilizes well-optimized operators to speed up training and save GPU memory of GNN models.
- **Ease of Use**: CogDL provides easy-to-use APIs for running experiments with the given models and datasets using hyper-parameter search.
- **Extensibility**: The design of CogDL makes it easy to apply GNN models to new scenarios based on our framework.

## ❗ News

- The new **v0.5.2 release** adds a GNN example for ogbn-products and updates geom datasets. It also fixes some potential bugs including setting devices, using cpu for inference, etc.

- The new **v0.5.1 release** adds fast operators including SpMM (cpu version) and scatter_max (cuda version). It also adds lots of datasets for node classification which can be found in [this link](./cogdl/datasets/rd2cd_data.py). 🎉

- The new **v0.5.0 release** designs and implements a unified training loop for GNN. It introduces `DataWrapper` to help prepare the training/validation/test data and `ModelWrapper` to define the training/validation/test steps. 🎉

- The new **v0.4.1 release** adds the implementation of Deep GNNs and the recommendation task. It also supports new pipelines for generating embeddings and recommendation. Welcome to join our tutorial on KDD 2021 at 10:30 am - 12:00 am, Aug. 14th (Singapore Time). More details can be found in https://kdd2021graph.github.io/. 🎉

<details>
<summary>
News History
</summary>
<br/>

- The new **v0.4.0 release** refactors the data storage (from `Data` to `Graph`) and provides more fast operators to speed up GNN training. It also includes many self-supervised learning methods on graphs. BTW, we are glad to announce that we will give a tutorial on KDD 2021 in August. Please see [this link](https://kdd2021graph.github.io/) for more details. 🎉

- CogDL supports GNN models with Mixture of Experts (MoE). You can install [FastMoE](https://github.com/laekov/fastmoe) and try **[MoE GCN](./cogdl/models/nn/moe_gcn.py)** in CogDL now!

- The new **v0.3.0 release** provides a fast spmm operator to speed up GNN training. We also release the first version of **[CogDL paper](https://arxiv.org/abs/2103.00959)** in arXiv. You can join [our slack](https://join.slack.com/t/cogdl/shared_invite/zt-b9b4a49j-2aMB035qZKxvjV4vqf0hEg) for discussion. 🎉🎉🎉

- The new **v0.2.0 release** includes easy-to-use `experiment` and `pipeline` APIs for all experiments and applications. The `experiment` API supports automl features of searching hyper-parameters. This release also provides `OAGBert` API for model inference (`OAGBert` is trained on large-scale academic corpus by our lab). Some features and models are added by the open source community (thanks to all the contributors 🎉).

- The new **v0.1.2 release** includes a pre-training task, many examples, OGB datasets, some knowledge graph embedding methods, and some graph neural network models. The coverage of CogDL is increased to 80%. Some new APIs, such as `Trainer` and `Sampler`, are developed and being tested. 

- The new **v0.1.1 release** includes the knowledge link prediction task, many state-of-the-art models, and `optuna` support. We also have a [Chinese WeChat post](https://mp.weixin.qq.com/s/IUh-ctQwtSXGvdTij5eDDg) about the CogDL release.

</details>

## Getting Started

### Requirements and Installation

- Python version >= 3.7
- PyTorch version >= 1.7.1

Please follow the instructions here to install PyTorch (https://github.com/pytorch/pytorch#installation).

When PyTorch has been installed, cogdl can be installed using pip as follows:

```bash
pip install cogdl
```

Install from source via:

```bash
pip install git+https://github.com/thudm/cogdl.git
```

Or clone the repository and install with the following commands:

```bash
git clone git@github.com:THUDM/cogdl.git
cd cogdl
pip install -e .
```

## Usage

### API Usage

You can run all kinds of experiments through CogDL APIs, especially `experiment`. You can also use your own datasets and models for experiments. 
A quickstart example can be found in the [quick_start.py](https://github.com/THUDM/cogdl/tree/master/examples/quick_start.py). More examples are provided in the [examples/](https://github.com/THUDM/cogdl/tree/master/examples/).

```python
from cogdl import experiment
```

# basic usage
experiment(dataset="pems-288", model="stgcn")







## CogDL Team
CogDL is developed and maintained by [Tsinghua, ZJU, BAAI, DAMO Academy, and ZHIPU.AI](https://cogdl.ai/about/). 

The core development team can be reached at [cogdlteam@gmail.com](mailto:cogdlteam@gmail.com).

## Citing CogDL

Please cite [our paper](https://arxiv.org/abs/2103.00959) if you find our code or results useful for your research:

```
@article{cen2021cogdl,
    title={CogDL: Toolkit for Deep Learning on Graphs},
    author={Yukuo Cen and Zhenyu Hou and Yan Wang and Qibin Chen and Yizhen Luo and Xingcheng Yao and Aohan Zeng and Shiguang Guo and Peng Zhang and Guohao Dai and Yu Wang and Chang Zhou and Hongxia Yang and Jie Tang},
    journal={arXiv preprint arXiv:2103.00959},
    year={2021}
}
```
