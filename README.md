# Chainer Skin Lesion Detector

[![MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/shunk031/chainer-skin-lesion-detector/blob/master/LICENSE)

Skin Lesion Detector using [HAM10000](https://arxiv.org/abs/1803.10417) dataset with [Chainer](https://chainer.org/)

## Requirements

- Python 3.6
- Chainer 5.0
- ChainerCV 0.11
- cupy-cuda90
- opencv 3.4

```shell
$ pip install -r requirements.txt
```

## Setup environment using Docker

```
$ docker build -t chainer-skin-lesion-detector .
$ docker run --rm -it -v $(pwd):/chainer-skin-lesion-detector --runtime nvidia --name chainer-skin-lesion-detector-dev chainer-skin-lesion-detector /bin/bash
```

## Directory architecture
```
.
├── data
│   ├── ISIC2018_Task1-2_Training_Input
│   ├── ISIC2018_Task1_Training_GroundTruth
│   ├── preprocessed
│   │   ├── ground_truth
│   │   └── input
│   └── xml
└── src
    ├── models
    │
    ├── notebooks
    │
    ├── result
    └── util
```

## Download dataset
- Download training dataset and ground truth data from [Task 1: Training | ISIC 2018](https://challenge2018.isic-archive.com/task1/training/) to `data/` directory

## Preprocess

- Re-scale image and ground truth
- Make bounding box from ground truth of segmentation image
- Create VOC format based label to `data/xml` directory

```shell
$ python make_dataset.py --loaderjob 4
```

An example of annotation data with a bounding box from the ground truth of segmentation using ISIC2018 task1 dataset:

![](https://raw.githubusercontent.com/shunk031/chainer-skin-lesion-detector/master/.github/ground_truth_segmentation_with_bbox.png)

## Train
- You can specify model, number of batch size, number of epoch, GPU ID and number of parallel data loading process.
```
$ python main.py --model ssd300 --batchsize 32 --epoch 30 --gpu 0 --loaderjob 4
```

## Evaluation

### Example of model prediction

![](https://raw.githubusercontent.com/shunk031/chainer-skin-lesion-detector/master/.github/example_of_model_prediction.png)

## Reference

- [Tschandl, Philipp, Cliff Rosendahl, and Harald Kittler. "The HAM10000 Dataset: A Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions." arXiv preprint arXiv:1803.10417 (2018).](https://arxiv.org/abs/1803.10417)
- [Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C. "SSD: Single shot multibox detector." 14th European Conference on Computer Vision, ECCV 2016. Springer Verlag, 2016.](https://arxiv.org/abs/1512.02325)
