
# Anomaly Detection via Self-Organizing Map

## Introduction
This paper is accepted by ICIP 2021.

SOMAD is a novel unsupervised anomaly detection approach based on Self-organizing Map (SOM)

For more details, please refer to our [paper](https://arxiv.org/abs/2107.09903).

## Requirements
- torch
- torchvision
- numpy
- opencv

## How to use
python somad.py --dataset mvtec

## Dataset
we use the [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), please prepare dataset like below
```
./mvtec/bottle
./mvtec/xxx
...
```
## TODO List
- [ ] Release the models trained using MVTec dataset
- [ ] Update train doc


## Citation
If you find SOMAD useful in your research, please consider citing:
```
@article{Li2021AnomalyDV,
  title={Anomaly Detection Via Self-Organizing Map},
  author={Ning Li and Kaitao Jiang and Zhiheng Ma and Xing Wei and Xiaopeng Hong and Yihong Gong},
  journal={2021 IEEE International Conference on Image Processing (ICIP)},
  year={2021},
  pages={974-978}
}
```
