# MNIST Digit Classification with PyTorch

A complete handwritten digit classification project built with PyTorch.

The project progresses from a basic Multi-Layer Perceptron (MLP) to a Convolutional Neural Network (CNN), followed by data augmentation, hyperparameter optimization using Optuna, custom image prediction, and error analysis.

---

## Results

| Model | Test Accuracy |
|---|---:|
| MLP | 96.67% |
| CNN | 99.14% |
| CNN + Data Augmentation | **99.26%** |

### Best Model

**CNN + Data Augmentation**

- Test Accuracy: **99.26%**
- Test samples: **10,000**
- Incorrect predictions: **74**
- Correct predictions: **9,926**

The CNN improved substantially over the MLP because convolutional layers are better suited for extracting spatial features from images.

---

# Project Overview

The goal of this project is to classify handwritten digits from the MNIST dataset.

The project was developed incrementally to understand the complete deep learning workflow:

1. Dataset preparation
2. Data preprocessing
3. MLP implementation
4. MLP training
5. MLP evaluation
6. CNN implementation
7. CNN training
8. CNN evaluation
9. Custom image prediction
10. Error analysis
11. Data augmentation
12. Augmented CNN training
13. Model comparison
14. Hyperparameter optimization with Optuna
15. Final model evaluation

---

# Dataset

The project uses the MNIST handwritten digit dataset.

MNIST contains:

- 60,000 training images
- 10,000 test images
- 10 classes
- Image size: 28 × 28 pixels
- Grayscale images

The 60,000 training images are split into:

- 48,000 training samples
- 12,000 validation samples

The test set contains:

- 10,000 samples

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Optuna
- Pillow

---

# Project Structure

```text
MNIST-PyTorch/
│
├── data/
│
├── models/
│   ├── best_mnist_mlp.pth
│   ├── best_mnist_cnn.pth
│   ├── best_mnist_cnn_augmented.pth
│   ├── best_mnist_cnn_optuna.pth
│   ├── best_mnist_cnn_v2.pth
│   ├── cnn_history.pth
│   ├── cnn_augmented_history.pth
│   └── optuna_study.db
│
├── results/
│   ├── cnn_accuracy_curve.png
│   ├── cnn_loss_curve.png
│   └── ...
│
├── src/
│   ├── cnn_model.py
│   ├── compare_all_models.py
│   ├── compare_models.py
│   ├── dataset.py
│   ├── error_analysis.py
│   ├── evaluate.py
│   ├── evaluate_cnn.py
│   ├── evaluate_cnn_augmented.py
│   ├── model.py
│   ├── plot_cnn_history.py
│   ├── predict.py
│   ├── predict_cnn.py
│   ├── show_optuna_results.py
│   ├── train.py
│   ├── train_cnn.py
│   ├── train_cnn_augmented.py
│   ├── visualize_augmentation.py
│   └── ...
│
├── .gitignore
├── requirements.txt
└── README.md