# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains a few models to tackle Udacity's Behavioral Cloning Project.

Besides the seminal [PilotNet](https://arxiv.org/abs/1704.07911) from NVIDIA, a couple of models pretrained on [imagenet](https://arxiv.org/abs/1409.0575) are included as well: [MoblineNet](https://arxiv.org/abs/1704.04861) and [VGG16](https://arxiv.org/abs/1409.1556).

All these models are implemented using [Keras](https://keras.io/).

Usage
---
The first step is downloading Udacity' Simulator where training examples are generated and the models validated. Click [here](https://github.com/udacity/self-driving-car-sim) for the details.

The included saved nvidia model and example videos were trained on the dataset provided by Udacity which can be downloaded from the classroom.

Training any model requires that a training dataset is present in `./data`:

```
$ python train.py {nvidia,mobilenet,vgg16}
```

Validation on the simulator is accomplished by using the `drive.py` script as follows:

```
$ python drive.py [model.h5]
```

### Dependencies
If using `pip` the dependencies can be installed by

```
$ pip install -r requirements.txt
```

This project requires:

* Keras 2.1.2
* Tensorflow 1.1+
* Flask-SocketIO
* eventlet
* numpy
* opencv
* pandas
* pydot
