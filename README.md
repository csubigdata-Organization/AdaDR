## Requirements

```shell
tensorflow-gpu==1.13.1
Keras==2.2.4
scikit-learn==0.22.2
protobuf==3.20.1
numpy==1.21.6
pandas==1.3.5
```


## Quick Start

A quick start example is given by:
```shell
$ python reproduce.py --dataset Fdataset
or
$ python reproduce.py --dataset Cdataset
or
$ python reproduce.py --dataset LRSSL
```

An example of auto search is as follows:
```shell
$ python auto_search.py --dataset LRSSL
```
