# YOLOv5 ONNX Object Detection Inference

![Downloads](https://img.shields.io/github/downloads/yakhyo/yolov5-onnx-inference/total)

<video controls autoplay loop src="https://github.com/user-attachments/assets/f988c823-0638-44b3-be7e-a77fe423e275" muted="false"></video>
Link: https://youtu.be/Snyg0RqpVxY

This repository contains code and instructions for performing object detection using YOLOv5 inference with ONNX Runtime.

## Features

- Inference using ONNX Runtime with GPU (tested on Ubuntu).
- Easy-to-use Python scripts for inference.
- Supports multiple input formats: image, video, or webcam.

## Installation

#### Clone the Repository

```bash
git clone https://github.com/yourusername/yolov5-license-plate-detection.git
cd yolov5-license-plate-detection
```

#### Install Required Packages

```bash
pip install -r requirements.txt
```

## Usage

Before running inference, you need to download weights of the YOLOv5 model weights in ONNX format.

#### Download weights (Linux)

```bash
sh download.sh anpr_yolov5s
               anpr_yolov5m
```

#### Download weights from the following links

**Note:** The weights are saved in FP32.

| Model Name | ONNX Model Link                                                                                                | Number of Parameters | Model Size |
| ---------- | -------------------------------------------------------------------------------------------------------------- | -------------------- | ---------- |
| YOLOv5s    | [yolov5s.onnx](https://github.com/yakhyo/yolov5-license-plate-detection/releases/download/v0.0.1/yolov5s.onnx) | 7.2M                 | 28 MB      |
| YOLOv5m    | [yolov5m.onnx](https://github.com/yakhyo/yolov5-license-plate-detection/releases/download/v0.0.1/yolov5m.onnx) | 21.2M                | 84 MB      |

<br>

> If you have custom weights, you can convert your weights to ONNX format. Follow the instructions in the [YOLOv5 repository](https://github.com/ultralytics/yolov5) to convert your model. You can use the converted ONNX model with this repository.

#### Inference

```bash
python main.py --weights weights/yolov5s.onnx --source assets/vid_input.mp4 # video
                                              --source 0 --view-img # webcam and display
                                              --source assets/img_input.jpg # image
```

- To save results add the `--save-img` argument and results will be saved under the `runs` folder
- To display video add the `--view-img` argument

**Command Line Arguments**

```
usage: main.py [-h] [--weights WEIGHTS] [--source SOURCE] [--img-size IMG_SIZE [IMG_SIZE ...]] [--conf-thres CONF_THRES] [--iou-thres IOU_THRES]
               [--max-det MAX_DET] [--save-img] [--view-img] [--project PROJECT] [--name NAME]

options:
  -h, --help            show this help message and exit
  --weights WEIGHTS     model path
  --source SOURCE       Path to video/image/webcam
  --img-size IMG_SIZE [IMG_SIZE ...]
                        inference size h,w
  --conf-thres CONF_THRES
                        confidence threshold
  --iou-thres IOU_THRES
                        NMS IoU threshold
  --max-det MAX_DET     maximum detections per image
  --save-img            Save detected images
  --view-img            View inferenced images
  --project PROJECT     save results to project/name
  --name NAME           save results to project/name
```

## Reference

1. https://github.com/ultralytics/yolov5
