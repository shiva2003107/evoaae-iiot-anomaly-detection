# EvoAAE-Based IIoT Anomaly Detection

A deep-learning-based anomaly detection framework for Industrial Internet of Things (IIoT) systems using an Evolutionary Adversarial Autoencoder (EvoAAE) with Particle Swarm Optimization (PSO).

The project uses the WADI (Water Distribution) dataset to investigate anomaly detection in industrial sensor data. The framework combines data preprocessing, convolutional autoencoder-based representation learning, adversarial training, and PSO-based hyperparameter optimization.

---

## 📌 Project Overview

Industrial IoT environments continuously generate data from sensors and process-control systems. Detecting abnormal behavior in this data is important for identifying potential faults, attacks, and unsafe operating conditions.

This project implements an EvoAAE-based anomaly detection approach that learns representations of industrial sensor data and identifies anomalous behavior using reconstruction-based anomaly scores.

Particle Swarm Optimization (PSO) is used to optimize selected model hyperparameters and improve the configuration of the EvoAAE model.

---

## 🎯 Objectives

- Detect anomalous behavior in Industrial IoT sensor data.
- Use an adversarial autoencoder architecture for representation learning.
- Apply convolutional neural networks to multivariate time-series data.
- Use PSO for hyperparameter optimization.
- Evaluate the trained model using classification metrics and confusion matrices.
- Analyze the model's ability to distinguish normal and attack samples.

---

## 🏗️ Methodology

The implemented workflow consists of the following major stages:

```text
WADI Industrial IoT Dataset
            |
            v
     Data Preprocessing
            |
            v
      Sliding Windows
            |
            v
   EvoAAE Model Architecture
            |
      +-----+-----+
      |           |
      v           v
   Encoder     Decoder
      |           |
      +-----+-----+
            |
            v
    Reconstruction Error
            |
            v
     Anomaly Score
            |
            v
      Thresholding
            |
            v
    Normal / Attack
```
The model uses convolutional layers in the encoder and decoder to process multivariate time-series windows.

The adversarial component uses discriminator networks to support the adversarial autoencoder learning process.

🧠 EvoAAE Model

The EvoAAE implementation contains:

- Convolutional encoder
- Latent representation
- Convolutional decoder
- Reconstruction loss
- Adversarial discriminators
- Hyperparameter optimization using PSO
  
The implemented model operates on multivariate time-series windows and reconstructs the input data. Reconstruction-based scores are subsequently used for anomaly detection.

## 🔍 Data Preprocessing

The project uses the WADI industrial control system dataset.

The preprocessing pipeline includes:

- Loading the industrial sensor data.
- Selecting numerical features.
- Handling missing values.
- Preparing multivariate time-series data.
- Creating fixed-length sliding windows.
- Preparing data for the EvoAAE model.

The final model input uses time-series windows of length 100 with multiple sensor features.

---

## ⚙️ PSO Hyperparameter Optimization

Particle Swarm Optimization (PSO) is used to search for suitable model configurations.

The optimization process evaluates candidate configurations and searches for improved hyperparameter combinations for the EvoAAE model.

The optimization stage is implemented separately in:

```text
src/pso.py
```
## 📊 Evaluation

The trained model is evaluated using anomaly detection metrics including:

- Precision
- Recall
- F1-score
- Accuracy
- ROC-AUC
- Confusion matrix

A threshold-based approach is used to convert anomaly scores into normal and attack predictions.

### Sample Evaluation

The repository includes a sample evaluation showing both a statistical threshold and a best-F1 threshold analysis.

For the best-F1 threshold analysis, the recorded result includes:

```text
Best F1: 0.3506
ROC-AUC: approximately 0.7775
```

The corresponding confusion matrix and detailed evaluation output are available in the `screenshots/` directory.

Performance can vary depending on the threshold and execution environment.

---

## 📷 Project Screenshots

The repository includes selected project outputs:

- EvoAAE architecture
- Confusion matrix
- Evaluation results

---

## 📁 Project Structure

```text
evoaae-iiot-anomaly-detection/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── README.md
│   ├── evoaae_model.py
│   ├── preprocessing.py
│   ├── pso.py
│   ├── training.py
│   └── evaluation.py
│
├── screenshots/
│   ├── README.md
│   ├── evoaae-architecture.png
│   ├── confusion-matrix.png
│   └── evaluation-results.png
│
├── results/
│   └── README.md
│
└── docs/
    └── README.md
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Scientific Computing | SciPy |
| Visualization | Matplotlib |
| Optimization | Particle Swarm Optimization (PSO) |
| Dataset | WADI |
| Development | VS Code / Jupyter Notebook |

---

## 🚀 Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/shiva2003107/evoaae-iiot-anomaly-detection.git
cd evoaae-iiot-anomaly-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Linux/WSL:

```bash
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Source Code

The main implementation files are available in:

```text
src/
```

The project is organized into separate preprocessing, optimization, model, training, and evaluation components.

---

## 📌 Notes

- The WADI dataset is not included in this repository.
- Large datasets, trained model files, generated intermediate data, and other large experimental artifacts are intentionally excluded to keep the repository lightweight.
- The repository focuses on the core implementation, selected evaluation outputs, and project documentation.

---

## 🔮 Future Improvements

Possible future improvements include:

- Evaluation on larger and more diverse industrial datasets.
- Improved detection of minority attack classes.
- More extensive hyperparameter optimization.
- Comparison with additional anomaly detection methods.
- Real-time IIoT anomaly detection.
- Improved threshold-selection strategies.
- Further scalability and computational-performance analysis.

---

## 🎓 Academic Project

**Project Type:** Minor Project  
**Academic Semester:** Third Semester

**Project Title:** Adversarial Deep Learning with Evolutionary Optimization for Unsupervised Anomaly Detection in Industrial IoT

**Degree:** M.Sc. Computer Science  
**Institution:** Central University of Kerala  
**Year:** 2025

## 👨‍💻 Author

**Mangali Shiva Prasad**  
M.Sc. Computer Science

## 📄 License

This project is developed for academic and educational purposes.
