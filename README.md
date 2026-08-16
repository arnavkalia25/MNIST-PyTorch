# MNIST Digit Classification with PyTorch

A complete deep learning project for handwritten digit classification using **PyTorch** and the **MNIST dataset**.

The project progressively builds and compares multiple neural network approaches, starting with a fully connected **MLP**, then a **CNN**, and finally a **CNN with Data Augmentation**.

The best model achieved **99.26% test accuracy** on the MNIST test set.

---

## Project Highlights

- Handwritten digit classification using PyTorch
- MNIST dataset exploration and visualization
- Custom MLP neural network
- CNN architecture using Conv2D and MaxPooling
- Training and validation pipeline
- Training/validation loss and accuracy curves
- Model evaluation using classification reports
- Confusion matrix analysis
- Custom image prediction
- CNN error analysis
- Data augmentation
- MLP vs CNN vs Augmented CNN comparison
- Reproducible environment using `requirements.txt`

---

# Model Performance

Three models were trained and evaluated on the MNIST test dataset.

| Model | Test Accuracy | Improvement |
|---|---:|---:|
| MLP | 96.67% | — |
| CNN | 99.14% | +2.34 percentage points |
| CNN + Data Augmentation | **99.26%** | **+2.59 percentage points** |

### Final Result

**Best Model: CNN + Data Augmentation**

**Test Accuracy: 99.26%**

The augmented CNN improved over the original CNN by:

**+0.25 percentage points**

---

# Dataset

The project uses the **MNIST handwritten digit dataset**.

MNIST contains:

- 70,000 grayscale handwritten digit images
- Image size: 28 × 28 pixels
- 10 classes: digits 0–9

The original MNIST training set contains 60,000 images.

This project splits those images into:

```text
Training:   48,000
Validation: 12,000
Test:       10,000