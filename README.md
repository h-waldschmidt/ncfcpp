# ncfcpp

Neural Collaborative Filtering with PyTorch in C++

Algorithms used in this repository are based on the following paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

You can find an implemenation by the authors of the paper in Python [here](https://github.com/hexiangnan/neural_collaborative_filtering)

## Why in C++?

It is probably better to use Python for ML/DL related tasks, because of its easy to use APIs, but integration into already existing C++ code-bases can sometimes be tricky.

## Installation of PyTorch for C++

This repository comes with a PyTorch binaries with (version 1.13.1 without CUDA). If you want to use CUDA, you can download the corresponding version on the [PyTorch Website](https://pytorch.org/get-started/locally/) and also need to install CUDA with several libraries like cuDNN.
