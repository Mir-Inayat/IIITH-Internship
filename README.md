# Automated Detection and Classification of Skin Diseases using YOLOv8

## Overview
This project implements a deep learning-based skin disease detection and classification system using the YOLOv8 object detection framework. Leveraging a curated dataset of dermatological images, the system is trained to identify and distinguish between several types of skin conditions, offering real-time, robust analysis suitable for assisting clinical diagnosis.

## Features
- Multi-class detection and classification of skin diseases in high-resolution medical images
- Fast and accurate prediction using Ultralytics YOLOv8
- Evaluation with standard metrics: precision, recall, mAP, and confusion matrix
- Ready-to-run scripts for training, inference, and result visualization

## Dataset
- Medical image dataset of annotated skin disease examples
- Images labeled for multiple disease classes
- Train/validation split for robust testing

## Setup
### Prerequisites
- Python 3.8+
- Ultralytics' YOLOv8 (`pip install ultralytics`)
- CUDA-compatible GPU (optional but recommended for training)
- Recommended packages: `torch`, `opencv-python`, `matplotlib`, `roboflow` (if using for dataset)

## Usage

### Training
Train the YOLOv8 model on your annotated dataset:
yolo detect train data=data.yaml model=yolov8n.pt epochs=20 imgsz=640

- Results and model weights will be saved in `runs/detect/train/`.

### Inference
Make predictions on new skin disease images:
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/test_images/

- Predicted images with bounding boxes and class labels saved in `runs/detect/predict/`.

### Evaluation & Visualization
- Review `results.png` for loss curves and metric graphs
- Review confusion matrix and output images to interpret model performance

## Key Results
- Achieved high mAP and precision on validation set
- Accurate detection of multiple skin conditions in real time

## Lessons Learned
- Quality and diversity of training data are crucial
- Fine-tuning YOLOv8 hyperparameters improves detection accuracy for medical images
- Model outputs and evaluation metrics offer insights for further improvement

## Skills Demonstrated
- Python (YOLOv8, PyTorch)
- Dataset annotation and curation (labeling tools, Roboflow)
- Model training, inference, and evaluation
- Interpretation of machine learning metrics
- Technical reporting and documentation

## Project Structure
├── data/
│ ├── images/
│ └── labels/
├── train.py
├── predict.py
├── results/
├── README.md

## References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow for dataset annotation](https://roboflow.com/)

