# MNIST-Digit-Classifier

Overview
This project implements a deep learning model to classify handwritten digits (0–9) from the MNIST dataset using TensorFlow/Keras. The model includes data preprocessing, a neural network with batch normalization and dropout, training with validation, and performance visualization.

Requirements
Dataset: MNIST dataset loaded via tensorflow.keras.datasets.mnist.
Libraries: TensorFlow, NumPy, Matplotlib, Seaborn.
Project Structure
Dataset Loading:
Loads MNIST dataset: 60,000 training and 10,000 test images (28x28 pixels).
Preprocessing:
Normalizes pixel values to [0, 1].
Flattens images to 784-dimensional vectors.
Converts labels to one-hot encoded format.
Model Architecture:
Sequential model with three hidden layers (256, 128, 64 units).
BatchNormalization applied before ReLU activation in each hidden layer.
Dropout (0.3, 0.3, 0.2) for regularization.
Output layer: 10 units with softmax activation.
Training:
Compiled with adam optimizer and categorical_crossentropy loss.
Trained for 15 epochs, batch size 128, with 20% validation split.
Visualization:
Plots training vs. validation accuracy and loss over epochs.
Figure size adjusted to figsize=(18, 5) for wider x-axis.
Y-axis limits set for balanced visualization (accuracy: 0–1, loss: dynamic range).
Displays five test images with predicted and true labels.
Evaluation:
Reports test accuracy and loss.
Typical test accuracy: ~97–98%.
How to Run
Open Google Colab and create a new notebook.
Copy the provided code into a cell.
Save as MNIST_Digit_Classifier.ipynb (or include your name).
Run all cells to generate outputs.
Share the notebook with “Anyone can view” permission.
Notes
BatchNormalization: Applied before ReLU activation for stable training.
Visualization: Wider figure size ensures longer x-axis for both accuracy and loss plots.
Troubleshooting: Use GPU runtime for faster training. Adjust epochs or dropout if accuracy is low.
Output
Preprocessing details, training logs, accuracy/loss plots, test accuracy/loss, and example predictions.
