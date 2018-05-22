# Nuclei Segmentation

## Model
![alt text](https://github.com/ternaus/robot-surgery-segmentation/blob/master/images/TernausNet.png)

## Prerequisites
- Pytorch
- [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/data)

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

### Nuclei Dataset
- To train the model, modify the dataset path in train.py 
```bash
file_path = 'dataset path'
```

### Train
- Train a model:
```bash
python train.py 
```

### Test
- Test the model:
```bash
python test.py
```
