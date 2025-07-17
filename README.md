# 🧠 MNIST Digit Classifier

## 🚀 Overview
Unlock the power of deep learning! This project builds a robust neural network to recognize handwritten digits (0–9) from the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using **TensorFlow/Keras**. The pipeline features thorough data preprocessing, a modern neural architecture (with Batch Normalization & Dropout), and insightful visualizations to track model performance.

---

## 📋 Requirements

- **Dataset:**  
  MNIST, loaded via `tensorflow.keras.datasets.mnist`
- **Libraries:**  
  - TensorFlow  
  - NumPy  
  - Matplotlib  
  - Seaborn  

---

## 🗂️ Project Structure

### 1️⃣ Dataset Loading
- Loads **60,000 training** and **10,000 test** grayscale images (28×28 pixels each).

### 2️⃣ Preprocessing
- **Normalization:** Scales pixel values to `[0, 1]`
- **Flattening:** Converts images to 784-dimension vectors
- **Label Encoding:** One-hot encodes digit labels

### 3️⃣ Model Architecture
- **Sequential Model** with **3 hidden layers:**
  - **Layer 1:** 256 units
  - **Layer 2:** 128 units
  - **Layer 3:** 64 units
- **Batch Normalization:** Before ReLU activation in all hidden layers
- **Dropout:** 0.3, 0.3, 0.2 (per hidden layer) for regularization
- **Output Layer:** 10 units (softmax activation) for digit classification

### 4️⃣ Training
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Epochs:** 15  
- **Batch Size:** 128  
- **Validation Split:** 20%

### 5️⃣ Visualization
- **Training vs. Validation Accuracy & Loss:**  
  - Plots with a wide aspect ratio (`figsize=(18, 5)`)
  - Y-axis: accuracy (0–1), loss (dynamic range)
- **Sample Predictions:**  
  - Displays 5 test images with predicted vs. true labels

### 6️⃣ Evaluation
- **Test Accuracy & Loss:**  
  - Typical accuracy: **~97–98%**

---

## ▶️ How to Run

1. **Open [Google Colab](https://colab.research.google.com/) and create a new notebook.**
2. **Copy the provided code** into a cell.
3. **Save as** `MNIST_Digit_Classifier.ipynb` (or personalize the filename).
4. **Run all cells** to see outputs and visualizations.
5. **Share the notebook** with “Anyone can view” permission for collaboration or review.

---

## 💡 Notes & Tips

- **Batch Normalization** is applied before ReLU activations for more stable and faster training.
- **Visualization** plots use a wider figure size for clarity.
- **Troubleshooting:**  
  - Enable **GPU runtime** in Colab for faster training.
  - Adjust **epochs** or **dropout rates** if accuracy is below expected.

---

## 📊 Output
- Data preprocessing details  
- Training logs  
- Accuracy/loss plots  
- Test set accuracy and loss  
- Example predictions with images

---

