# AIGO-Segmentation

This respository aims to train DeepLab v3 in PyTorch which using Mapillary Vistas datasets.


![Example image segmentation video](https://github.com/kuobrian/AIGO-Segmentation/blob/main/images/example.gif)

## Usage
### Running Inference

The easiest way to get started with inference is to clone this repository and use the `demo.py` script. if you have street images named `test.jpg`, then you can generate segmentation labels for them with the following command.

`demo.py` script incluide testing video single image and camera.

```shell
$ python demo.py
```

##### Download our weights [googledrive](https://drive.google.com/drive/folders/1fXa_6e5fpmb9nzMBws_xMX5HY_rY9WVX?usp=sharing)

download weights and change code in `demo.py` script

```shell
$ folder = params.model_id
$ ckpt_path = './logs/{}/checkpoints/best_valid.pth'.format(folder)

```

##### Test Video  [Video folder](https://drive.google.com/drive/folders/1ErgRlsQvF38M0M4OW8AzdT260FACtuMG?usp=sharing)

download video and put it in video folder


## Training Process
