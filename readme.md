# Human Activity Recognition Deep Learning Model

## Table of Contents
1. [Description](#description)
2. [Quick Start](#quick-start)

## Description
This repository contains a deep learning model designed for activity recognition, leveraging time-series data from smartphone sensors. The model classifies various types of human activities by analyzing acceleration and rotation data captured by smartphones' built-in accelerometers and gyroscopes. This approach aims to provide real-time, accurate activity classification, such as walking, running, sitting, and more.

## Quick Start
There are 3 model architectures: DNN, CNN, and LSTM with the optimal parameter searched from Keras tuner. To train the models, try
```bash

# Installation
git clone git@github.com:chindanaitrakan/activity-recognition.git
cd ~/activity-recognition

# Train models
python train_dnn.py
