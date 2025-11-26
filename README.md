# Malaria Cell Detection using Deep Learning

A deep learning project for detecting malaria-infected cells in blood smear images using Convolutional Neural Networks (CNN) and Transfer Learning with ResNet50.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technologies Used](#technologies-used)

## 🔬 Project Overview

This project implements deep learning models to classify blood cell images as either:
- **Parasitized** (infected with malaria)
- **Uninfected** (healthy cells)

The project includes:
1. **Exploratory Data Analysis (EDA)** - Understanding the dataset distribution and characteristics
2. **Baseline CNN Model** - A custom 3-layer convolutional neural network
3. **Transfer Learning with ResNet50** - Fine-tuned pre-trained model for improved performance

## 📊 Dataset

**Source:** Malaria Cell Images Dataset
- **Total Images:** 27,558
- **Training Set:** 19,290 images (70%)
- **Validation Set:** 4,133 images (15%)
- **Test Set:** 4,135 images (15%)

**Classes:**
- Parasitized cells (infected)
- Uninfected cells (healthy)

**Image Specifications:**
- Format: PNG
- Input size: 224×224 pixels (after preprocessing)
- Color: RGB (3 channels)

**Data Preprocessing:**
- Resizing to 224×224 pixels
- Normalization with ImageNet mean and std
- Data augmentation (training only):
  - Random horizontal flip (p=0.5)
  - Random vertical flip (p=0.2)
  - Random rotation (±20°)
  - Color jitter (brightness, contrast, saturation)

## 📁 Project Structure

```
malaria-cell-detection-dl/
├── README.md
├── requirements.txt
├── configs/
│   └── fine_tune_resnet50.yaml      # ResNet50 training configuration
├── data/
│   ├── cell_images/                 # Raw dataset
│   │   ├── Parasitized/
│   │   └── Uninfected/
│   └── preprocessed_data/           # Processed dataset
│       ├── train/
│       ├── val/
│       ├── test/
│       └── dataset_info.pt
├── notebooks/
│   └── EDA.ipynb                    # Exploratory Data Analysis
├── scripts/
│   ├── data.py                      # Data loading and preprocessing
│   ├── baseline_model.py            # Baseline CNN implementation
│   ├── test_baseline_model.py       # Baseline model evaluation
│   └── fine_tune_resnet50.py        # ResNet50 transfer learning
├── outputs/
    ├── models/                      # Saved model weights
    │   ├── baseline_model.pth
    │   ├── resnet50_finetuned.pth
    │   └── checkpoint.pth
    ├── figures/                     # Evaluation plots
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    ├── logs/                        # TensorBoard logs
    └── screenshots/                 # Training progress screenshots

```

## 🤖 Models

### 1. Baseline CNN Model

**Architecture:**
```
Input (224×224×3)
    ↓
Conv2D (32 filters, 3×3) → ReLU → MaxPool2D (2×2)
    ↓
Conv2D (64 filters, 3×3) → ReLU → MaxPool2D (2×2)
    ↓
Conv2D (128 filters, 3×3) → ReLU → MaxPool2D (2×2)
    ↓
Flatten → Linear (128×16×16 → 256) → ReLU → Dropout (0.5)
    ↓
Linear (256 → 1) → Sigmoid
```

**Training Configuration:**
- Optimizer: AdamW (lr=1e-4)
- Loss Function: BCEWithLogitsLoss
- Epochs: 10
- Batch Size: 32
- Device: Apple Silicon (MPS)

**Key Features:**
- Simple 3-layer CNN architecture
- Dropout regularization to prevent overfitting
- Binary classification output

### 2. ResNet50 Transfer Learning

**Architecture:**
- Base: Pre-trained ResNet50 (ImageNet weights)
- Modified final layer for binary classification
- Option to freeze/unfreeze backbone layers

**Training Configuration:**
```yaml
Model:
  - Backbone: ResNet50 (pre-trained)
  - Freeze backbone: True
  - Output classes: 1 (binary)

Training:
  - Epochs: 12
  - Batch Size: 32
  - Learning Rate: 0.0001
  - Weight Decay: 0.0001
  - Gradient Clipping: max_norm=1.0
  - Random Seed: 42
  
Optimization:
  - Optimizer: AdamW
  - Scheduler: ReduceLROnPlateau
    - Factor: 0.5
    - Patience: 3
  - Early Stopping: patience=5

Advanced Features:
  - AMP (Automatic Mixed Precision): Optional
  - Checkpoint Averaging: Top-3 models
  - TensorBoard Logging
  - Gradient Clipping
```

## 📈 Results

### Baseline CNN Model

**Test Set Performance:**
- **Accuracy:** ~95%
- **Precision:** High precision in detecting infected cells
- **Recall:** Good recall rate
- **F1-Score:** Balanced performance
- **ROC-AUC:** Strong discriminative ability

**Training Progress:**
- Stable convergence over 10 epochs
- Validation accuracy closely follows training accuracy
- Minimal overfitting with dropout regularization

### ResNet50 Transfer Learning

**Improvements over Baseline:**
- Better feature extraction using pre-trained weights
- More robust to variations in cell images
- Higher accuracy and generalization
- Advanced training techniques (gradient clipping, learning rate scheduling)

**Features:**
- TensorBoard integration for real-time monitoring
- Checkpoint saving with model averaging
- Early stopping to prevent overfitting
- Support for multiple compute devices (CUDA/MPS/CPU)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd malaria-cell-detection-dl
```

2. **Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Exploratory Data Analysis

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 2. Train Baseline Model

```bash
cd scripts
python baseline_model.py
```

This will:
- Train the baseline CNN for 10 epochs
- Save the model to `outputs/models/baseline_model.pth`
- Display training progress with loss and accuracy

### 3. Evaluate Baseline Model

```bash
cd scripts
python test_baseline_model.py
```

This will:
- Load the trained baseline model
- Evaluate on the test set
- Generate confusion matrix and ROC curve
- Save plots to `outputs/figures/`
- Display metrics (accuracy, precision, recall, F1-score)

### 4. Train ResNet50 Model

```bash
cd scripts
python fine_tune_resnet50.py
```

This will:
- Load configuration from `configs/fine_tune_resnet50.yaml`
- Train ResNet50 with transfer learning
- Save checkpoints and final model
- Log training progress to TensorBoard

### 5. Monitor Training with TensorBoard

```bash
tensorboard --logdir outputs/logs
```

Then open your browser to `http://localhost:6006`

## ⚙️ Configuration

Edit `configs/fine_tune_resnet50.yaml` to customize training:

```yaml
model:
  freeze_backbone: True/False    # Freeze ResNet50 layers
  num_classes: 1                 # Binary classification

training:
  epochs: 12                     # Number of training epochs
  batch_size: 32                 # Batch size
  learning_rate: 0.0001          # Initial learning rate
  early_stopping_patience: 5     # Early stopping patience

amp:
  enabled: False                 # Use mixed precision training

```

## 🔧 Technologies Used

### Deep Learning Framework
- **PyTorch 2.0+** - Deep learning framework
- **torchvision** - Image transformations and pre-trained models

### Data Processing
- **NumPy** - Numerical computations
- **Pillow (PIL)** - Image processing
- **scikit-learn** - Metrics and evaluation

### Visualization
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical visualizations
- **TensorBoard** - Training monitoring

### Development Tools
- **tqdm** - Progress bars
- **PyYAML** - Configuration management
- **Jupyter** - Interactive notebooks

## 📊 Model Evaluation Metrics

The project uses comprehensive metrics:
- **Accuracy** - Overall correctness
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Visualization of predictions vs actual
- **ROC Curve & AUC** - Model discriminative ability

## 🎯 Future Improvements

- [ ] Implement additional architectures (EfficientNet, Vision Transformer)
- [ ] Add data augmentation techniques (Mixup, CutMix)
- [ ] Experiment with different loss functions
- [ ] Deploy model as a web application
- [ ] Add model interpretability (Grad-CAM, attention maps)
- [ ] Ensemble multiple models for better performance

## 📝 License

This project is for educational purposes.
