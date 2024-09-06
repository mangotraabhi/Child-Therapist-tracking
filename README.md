
# Detecting and Tracking Therapist and Child

This project implements a system for detecting and tracking therapists and children in videos using a fine-tuned YOLOv8n model and DeepSORT for multi-object tracking. The model was trained using the notebook available at [Adult-child-detection-yolov8n.ipynb](https://github.com/mangotraabhi/Projects/blob/main/Adult-child-detection-yolov8n.ipynb).

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Output](#output)
7. [Customization](#customization)
8. [Limitations](#limitations)

## Features

- Therapist and child detection using a fine-tuned YOLOv8n model (named as best.pt).
- Multi-object tracking with DeepSORT
- Batch processing of multiple video files
- Unique ID assignment for each tracked person (children: 1-99, adults: 100+)
- Color-coded bounding boxes (yellow for children, red for adults)
- Output video generation with bounding boxes and labels

## Requirements

- Google Colab notebook
- GPU runtime (recommended for faster processing)

## Setup

1. Open the Colab notebook containing the project code.

2. Install the required packages by running the following cells:

   ```python
   !pip install ultralytics
   !pip install torch torchvision torchaudio
   !pip install deep_sort_realtime
   !pip install opencv-python
   ```

3. Import the necessary libraries:

   ```python
   import ultralytics
   from ultralytics import YOLO
   import time
   import torch
   import cv2
   import torch.backends.cudnn as cudnn
   from PIL import Image
   import colorsys
   import numpy as np
   from deep_sort_realtime.deepsort_tracker import DeepSort
   import os
   ```


4. Load the fine-tuned model:

   ```python
   model = YOLO("/content/best.pt")
   ```




## Usage

1. Upload your input video files to the Colab environment in the `/content/input_videos` and best.pt model file to the `/content` directory. 
2. Run the code cells in the notebook sequentially.
3. The processed videos will be saved in the `/content/output_videos` directory.
4. You can download the output videos from the Colab file browser.


## How It Works

1. **Video Processing**: The script iterates through all `.mp4` files in the input directory.
2. **Object Detection**: The fine-tuned YOLOv8n model is used to detect therapists and children in each frame.
3. **Object Tracking**: DeepSORT algorithm tracks the detected people across frames.
4. **ID Assignment**: Unique IDs are assigned to each tracked person (1-99 for children, 100+ for adults).
5. **Visualization**: Color-coded bounding boxes are drawn around detected people (yellow for children, red for adults), along with their assigned ID and class label.
6. **Output Generation**: Processed frames are compiled into an output video file.


## Output

The script generates output videos with the following features:

- Color-coded bounding boxes (yellow for children, red for adults)
- Labels showing the tracking ID and class (Child/Therapist) for each person
- Output videos are saved in the `/content/output_videos` directory


## Customization

You can customize various parameters in the script:

- `max_age`, `n_init`, `nms_max_overlap`, `max_iou_distance` in the DeepSort initialization
- Colors for children and adults (`child_color` and `adult_color`)
- ID ranges for children and adults (`next_child_id` and `next_adult_id`)


## Limitations

**Note**: The accuracy of this model may be limited due to several factors:

1. Dataset constraints: The model's performance depends on the quality and diversity of the training dataset.
2. GPU constraints: Training and inference performance may be affected by the available computational resources.
3. Model size: The use of the YOLOv8n (nano) model, while faster, may result in lower accuracy compared to larger models.


For optimal results, consider using a larger dataset, more powerful GPU, and experimenting with larger YOLOv8 model variants if higher accuracy is required.
