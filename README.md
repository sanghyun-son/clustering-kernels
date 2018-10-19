# Clustering Convolutional Kernels to Compress Deep Neural Networks
![](/figs/main.png)

This repository is an official PyTorch implementation of the Paper **"Clustering Convolutional Kernels to Compress Deep Neural Networks"** from **ECCV 2018 (poster)**.
If you find our work useful in your research of publication, please cite our work:

[1] Sanghyun Son, Seungjun Nah, Kyoung Mu Lee, **"Clustering Convolutional Kernels to Compress Deep Neural Networks,"** In ECCV 2018.
[[PDF](https://cv.snu.ac.kr/publication/conf/2018/Sanghyun_Son_Clustering_Convolutional_Kernels_ECCV_2018_paper.pdf)]
[[Poster](https://cv.snu.ac.kr/research/clustering_kernels/eccv2018_clustering_kernels_poster.pdf)]

```
@inproceedings{son2018clustering,
  author = {Son, Sanghyun and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Clustering Convolutional Kernels to Compress Deep Neural Networks},
  booktitle = {ECCV},
  month = {September},
  year = {2018}
}
```

We provide scripts to reproduce every experiment from our paper.
With some additional coding, you can compress your pre-trained model, too.

**Currently, some scripts are under revising. They will be included in ``demo.sh`` after revision.**

## What we provide
* Codes for training baseline/compressed models
* Centroids visualization
* GPU-accelerated k-means (transform invariant) clustering algorithm
* Non-official implementations of existing network quantization methods
* Functions to save/load compressed models
 (**Currently disabled due to the compatibility issue. Will be included in future.**)

## What we do not provide
* CUDA kernels to accelerate the proposed algorithm (We only report theoretical speed-up in the paper.)

## Dependencies
* Python 3.6
* PyTorch >= 0.4.1
* numpy
* matplotlib
* tqdm

## Quick start
Clone this repository into any place you want.
```bash
git clone https://github.com/thstkdgus35/clustering-kernels
cd clustering-kernels/src
```

If you would like to do some experiments about ImageNet classification, follow [this link](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to prepare the dataset.
Your dataset directory should be like below:
```bash
[some_path]/ILSVRC2012/train/[1000_many_folders]
[some_path]/ILSVRC2012/va/[1000_many_folders]
```
Then, you have to specify the directory with ``--dir_data [some_path]``.
It is also possible to fix a default argument in ``options.py`` directly.

We recommend you to download our pre-trained models (with ``--pretrained download``), but it is also possible to train them from scratch.
In that case, remove ``--pretrained download --test_only`` from ``demo.sh`` and specify a save directory with ``--save [directory_you_want]``.

After you prepare all the prerequisites, uncomment a line you want to execute and type ``sh demo.sh``!

