# circledetectionproject


## Overview
This project tries to develop an ML model to detect circles in noisy images. The goal is to accurately determine the circle's center and radius in a given image using an IOU (intersection over union) calculation.

## Code Structure
slingshotai.py: The main script that initializes the training and evaluation process
data_generation.py: Contains functions for generating images with noisy circles and the corresponding circle parameters of center and radius
model.py: Defines the CNN model used for circle detection
train_and_evaluate.py: Handles the training and evaluation of the model, including dataset creation and IOU calculation

## Model Architecture
The model is a CNN with layers with increasing depth (32, 64, 128 filters) for feature extraction. Dense layers for final prediction.

## Metrics
Average IOU Score: Currently the IOU score is around 0.31, meaning there is room for improvement in the model's final evaluations.

## Packages
TensorFlow
NumPy
Matplotlib
scikit-image
