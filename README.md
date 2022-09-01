# DEMO

[cogdl-trafficPre demo](https://renxs177.github.io/cogdl-trafficPre/demo_traffic.html)   

# Usage

## API Usage

You can run all kinds of experiments through CogDL APIs, especially `experiment`. You can also use your own datasets and models for experiments. 
A quickstart example can be found in the [STGCN_example.py](https://github.com/Renxs177/cogdl-trafficPre/blob/master/examples/simple_stgcn/STGCN_example.py). More examples are provided in the [examples/](https://github.com/Renxs177/cogdl-trafficPre/tree/master/examples).

```python
from cogdl import experiment
```

## basic usage
experiment(dataset="pems-stgcn", model="stgcn")







# CogDL Team
CogDL is developed and maintained by [Tsinghua, ZJU, BAAI, DAMO Academy, and ZHIPU.AI](https://cogdl.ai/about/). 

The core development team can be reached at [cogdlteam@gmail.com](mailto:cogdlteam@gmail.com).

# Citing CogDL

Please cite [our paper](https://arxiv.org/abs/2103.00959) if you find our code or results useful for your research:

```
@article{cen2021cogdl,
    title={CogDL: Toolkit for Deep Learning on Graphs},
    author={Yukuo Cen and Zhenyu Hou and Yan Wang and Qibin Chen and Yizhen Luo and Xingcheng Yao and Aohan Zeng and Shiguang Guo and Peng Zhang and Guohao Dai and Yu Wang and Chang Zhou and Hongxia Yang and Jie Tang},
    journal={arXiv preprint arXiv:2103.00959},
    year={2021}
}
```
