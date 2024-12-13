# CIFAR-10 Classification: Baseline vs Data Augmentation

## Project Overview
This project evaluates the impact of data augmentation on the performance of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. A baseline model without augmentation and an augmented model incorporating real-time data transformations were trained and compared.

## Dataset
The CIFAR-10 dataset is a popular benchmark dataset for image classification tasks. It contains:
- **60,000 images** (32x32 color images) in **10 classes**, such as airplanes, automobiles, birds, cats, and dogs.
- Split into **50,000 training samples** and **10,000 testing samples**.
- Classes are balanced, with **6,000 images per class**.

Dataset source: [CIFAR-10 by Alex Krizhevsky and Geoffrey Hinton](https://www.cs.toronto.edu/~kriz/cifar.html).

## Project Objectives
1. Train and evaluate a **baseline CNN model** on the CIFAR-10 dataset.
2. Implement data augmentation techniques and train an **augmented model** to assess its impact on classification performance.
3. Compare results to understand how data augmentation affects convergence and accuracy.

---

## Implementation Details

### 1. Baseline Model
#### Architecture
- **Convolutional Layers**: Extract spatial features using 2D kernels with ReLU activation.
- **Max-Pooling Layers**: Downsample feature maps to reduce spatial dimensions and computational complexity.
- **Fully Connected Layers**: Map learned features to class probabilities.
- **Softmax Output**: Converts logits into class probabilities.

#### Training Configuration
- **Optimizer**: Adam Optimizer
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 10
- **Batch Size**: 32

---

### 2. Data-Augmented Model
Real-time transformations were applied during training to improve the model's robustness and generalization. Transformations included:
- **Random Rotations**: Rotated images randomly within a specified angle range.
- **Horizontal Flips**: Mirrored images along the vertical axis.
- **Shifts**: Random translations in both horizontal and vertical directions.
- **Normalization**: Pixel values scaled to the range [0, 1].

---

## Training and Results

### 1. Baseline Model
- **Final Training Accuracy**: 71.55%
- **Final Validation Accuracy**: 69.27%
- **Final Training Loss**: 0.7976
- **Final Validation Loss**: 0.8951
- **Overall Test Accuracy**: 69.12%

**Observations**:
- Rapid accuracy improvement in initial epochs, with diminishing returns in later epochs.
- Precision and recall were higher for simpler classes like "Airplane" and "Automobile" and lower for complex classes like "Cat" and "Bird."

---

### 2. Augmented Model
- **Final Training Accuracy**: 53.01%
- **Final Validation Accuracy**: 61.24%
- **Final Training Loss**: 1.3236
- **Final Validation Loss**: 1.0848
- **Overall Test Accuracy**: 61.24%

**Observations**:
- Slower convergence due to added variability from augmentations.
- Improved generalization, especially for harder-to-classify classes and distorted inputs.
- Increased robustness to edge cases, evident in better recall for challenging classes.

---

## Results Comparison

| Metric                    | Baseline Model | Augmented Model |
|---------------------------|----------------|-----------------|
| Final Training Accuracy   | 71.55%         | 53.01%          |
| Final Validation Accuracy | 69.27%         | 61.24%          |
| Final Training Loss       | 0.7976         | 1.3236          |
| Final Validation Loss     | 0.8951         | 1.0848          |

**Key Insights**:
- The baseline model outperformed the augmented model in terms of accuracy due to faster convergence on simpler features.
- The augmented model, while slower to converge, exhibited improved generalization and robustness to transformations.

---

## How to Run
### Prerequisites
Ensure the following are installed:
- Python 3.7+
- TensorFlow/Keras
- NumPy
- Matplotlib

### Steps
1. Clone this repository:
   ```bash
   git clone 
