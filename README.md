# 🌩️ CloudStorm: A Hybrid ML/DL Framework for Cloudburst Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Can machine learning predict nature's fury before it strikes?**

*An intelligent early-warning system that combines classical ML with deep learning to forecast cloudburst events using meteorological data.*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Architecture](#-architecture)

</div>

---

## 🎯 The Mission

Cloudbursts are among nature's most devastating phenomena—sudden, violent, and often deadly. In regions like the Swat Valley, Pakistan, these events trigger catastrophic flash floods with little warning.

**CloudStorm** leverages the power of machine learning to analyze atmospheric conditions and predict cloudburst events *before* they occur, potentially saving lives and enabling timely evacuations.

> *"In the battle against natural disasters, every second of early warning counts. This project is a step toward giving communities that precious time."*

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **Multi-Model Approach** | Compares Logistic Regression, Random Forest, and LSTM neural networks |
| 📊 **Comprehensive Evaluation** | Accuracy, Precision, Recall, F1-Score, and Confusion Matrices |
| 🧠 **Deep Learning Integration** | LSTM architecture for temporal pattern recognition |
| 🎮 **Interactive Prediction** | Real-time cloudburst risk assessment via CLI |
| 📈 **Visual Analytics** | Automated performance comparison charts |
| 🔧 **Production-Ready Pipeline** | Scalable preprocessing with StandardScaler |

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/cloudstorm-prediction.git
cd cloudstorm-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 📖 Usage

### 1. Prepare Your Data

Ensure your CSV file contains the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `Max_Daily_Rainfall_mm` | Maximum daily rainfall | mm |
| `Pressure_hPa` | Atmospheric pressure | hPa |
| `Humidity_%` | Relative humidity | % |
| `Cloud_Cover_%` | Cloud coverage | % |
| `Cloudburst_Event` | Target variable (0/1) | binary |

### 2. Configure the Data Path

```python
# Update this line in the script
df = pd.read_csv(r'path/to/your/swat_cloudburst_data.csv')
```

### 3. Run the Prediction Pipeline

```bash
python cloudstorm.py
```

### 4. Interactive Prediction

Once trained, the system prompts for real-time input:

```
--- Cloudburst Prediction ---
Enter Max Daily Rainfall (mm): 145.5
Enter Pressure (hPa): 1002.3
Enter Humidity (%): 87
Enter Cloud Cover (%): 92

⚠️ CLOUDBURST ALERT! (Probability: 0.89)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│  Raw CSV → Missing Value Imputation → Feature Scaling → Split   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                              │
├──────────────────┬──────────────────┬───────────────────────────┤
│ Logistic         │ Random Forest    │ LSTM Neural Network       │
│ Regression       │ Classifier       │ (TensorFlow/Keras)        │
│                  │                  │                           │
│ • Linear         │ • Ensemble       │ • Sequential              │
│ • Interpretable  │ • Non-linear     │ • Dropout Regularization  │
│ • Baseline       │ • Feature Imp.   │ • Early Stopping          │
└──────────────────┴──────────────────┴───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION & SELECTION                        │
├─────────────────────────────────────────────────────────────────┤
│  Accuracy │ Precision │ Recall │ F1-Score │ Confusion Matrix    │
│                                                                  │
│              🏆 Best Model Selection (by F1-Score)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INTERACTIVE PREDICTION                         │
│         Real-time cloudburst risk assessment interface           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Deep Dive

### Logistic Regression
- **Role**: Interpretable baseline model
- **Strength**: Fast training, clear feature coefficients
- **Use Case**: Understanding feature importance

### Random Forest Classifier
- **Role**: Robust ensemble method
- **Configuration**: 100 estimators
- **Strength**: Handles non-linear relationships, resistant to overfitting

### LSTM Neural Network

```python
model_lstm = Sequential([
    LSTM(32, input_shape=(1, 4), activation='tanh'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

- **Role**: Deep learning approach
- **Strength**: Captures complex patterns, potential for temporal extension

---

## 📊 Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| **LSTM** | **0.XXXX** | **0.XXXX** | **0.XXXX** | **0.XXXX** |

*Run the pipeline to populate with your dataset's results*

### Visualizations Generated

1. **Performance Comparison Bar Chart** - Side-by-side metric comparison
2. **Confusion Matrices** - Detailed classification breakdown per model
3. **Training History** - Loss and accuracy curves for LSTM

---

## 🔬 Technical Highlights

### Data Preprocessing

```python
# Robust missing value handling
df.fillna(df.median(numeric_only=True), inplace=True)

# Feature standardization (critical for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified splitting preserves class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
```

### LSTM Input Reshaping

```python
# LSTM requires 3D input: (samples, timesteps, features)
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
```

### Early Stopping

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

---

## 🛠️ Customization Guide

### Adding New Features

```python
# Extend feature columns
feature_cols = [
    'Max_Daily_Rainfall_mm',
    'Pressure_hPa',
    'Humidity_%',
    'Cloud_Cover_%',
    'Wind_Speed_kmh',      # New feature
    'Temperature_C'        # New feature
]
```

### Tuning LSTM Hyperparameters

```python
model_lstm = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),  # More units
    LSTM(32),
    Dropout(0.3),                                                          # Higher dropout
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)
```

---

## 📁 Project Structure

```
cloudstorm-prediction/
│
├── cloudstorm.py              # Main pipeline script
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── data/
│   └── swat_cloudburst.csv    # Dataset (not included)
│
├── outputs/
│   ├── model_comparison.png   # Generated visualizations
│   └── confusion_matrices.png
│
└── models/
    └── best_model.h5          # Saved model (optional)
```

---

## 🌍 Real-World Impact

This framework can be extended to:

- 🏔️ **Mountainous Regions**: Adapt for Himalayan, Andean, or Alpine cloudburst patterns
- 🌊 **Flood Early Warning**: Integrate with existing disaster management systems
- 📱 **Mobile Alerts**: Deploy as API for SMS-based warning systems
- 🛰️ **Satellite Integration**: Combine with remote sensing data

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Improvement

- [ ] Add XGBoost/LightGBM comparison
- [ ] Implement GRU and Transformer models
- [ ] Add SHAP explainability
- [ ] Create web dashboard (Streamlit/Flask)
- [ ] Implement k-fold cross-validation
- [ ] Add hyperparameter tuning (Optuna)

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- Meteorological data provided by Pakistan Meteorological Department
- Inspired by the need for accessible disaster prediction tools
- Built with open-source tools: TensorFlow, scikit-learn, pandas

---

<div align="center">

**⭐ Star this repo if it helped you!**

*"Predicting the storm, protecting the people."*

[Report Bug](https://github.com/asadkhanjadoonece-cmd/c-loud-burst-prediction) • [Request Feature](https://github.com/yourusername/cloudstorm-prediction/issues)

</div>
