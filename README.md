


# AI-Based Road Traffic Noise Prediction Using Machine Learning and Deep Learning

Road traffic noise is an important environmental and public health issue, especially in urban areas.  
This project investigates the feasibility of predicting roadside traffic noise levels using supervised machine learning and deep learning models based on environmental and traffic-related metadata.

LightGBM, XGBoost, BiLSTM, LSTM, BiGRU, GRU, and Linear Regression models are used.  
The project uses the IDMT Traffic Dataset, which contains 9,362 two-second stereo recordings of passing vehicles collected under different environmental and traffic conditions, as well as 8,144 background noise recordings.

The `.wav` recordings are converted into decibel (dB) values and used as target outputs for the regression models.  
The goal is to predict roadside traffic noise levels from metadata features without directly using acoustic features as model inputs.

Different regression-based machine learning and deep learning models were evaluated, including:

- LightGBM
- XGBoost
- BiLSTM
- LSTM
- BiGRU
- GRU
- Linear Regression

The project uses the IDMT Traffic Dataset, which contains roadside recordings of passing vehicles under different environmental and traffic conditions.

The `.wav` recordings are processed and converted into decibel (dB) values, which are used as prediction targets.  
The study investigates whether traffic noise levels can be estimated solely from metadata features without directly using acoustic features as model inputs.

---

## Repository Structure

```text
Road_Noise_Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── models/
│
├── results/
│
├── figures/
│
├── README.md
├── requirements.txt
└── environment.yml
```

---

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yml
conda activate road_noise_env
```

---

## Dataset

The project uses the publicly available IDMT Traffic Dataset:

- Dataset: https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html

The dataset contains:

- 9,362 two-second traffic recordings
- 8,144 background noise recordings

Recordings were collected from roadside measurement setups under various environmental and traffic conditions.

<p align="center">
  <img src="https://github.com/user-attachments/assets/15229f28-ba14-4735-af51-9cebfb33d751" width="500"><br>
  <em>Figure 1: Example roadside recording setup from the original dataset publication.</em>
</p>

---

## Metadata Features

Each recording contains the following metadata information:

- 4 different locations
- 4 different speed zones
- Daytime information (morning / afternoon)
- Weather conditions (dry / wet)
- Vehicle category:
  - Bus
  - Car
  - Motorcycle
  - Truck
- Vehicle direction
- Microphone type (ME / SE)
- Background or traffic recording information

The objective is to predict the corresponding sound pressure level (dB) from these metadata features.

---

# Data Cleaning

Some recordings were not usable due to:

- unexpected background sounds
- corrupted microphone signals
- sudden amplitude spikes caused by microphone faults

<p align="center">
  <img src="https://github.com/user-attachments/assets/65657881-7116-4745-b484-df14e4a04c20" width="800"><br>
  <em>Figure 2: Examples of unusable recordings.</em>
</p>

Recordings exceeding predefined amplitude thresholds were removed.

The thresholds were determined separately for:

- vehicle category
- microphone type

<p align="center">
  <img src="https://github.com/user-attachments/assets/06053498-99fd-41b6-855d-a513643cb4be" width="600"><br>
  <em>Figure 3: Maximum allowed amplitude thresholds.</em>
</p>

Thresholds were selected based on outlier analysis using boxplots.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7aea884b-edb5-4d40-965e-81210b5bbac5" width="300"><br>
  <em>Figure 4: Example boxplot analysis for bus recordings.</em>
</p>

After cleaning:

- 472 recordings were removed
- 17,034 recordings remained for further analysis

<p align="center">
  <img src="https://github.com/user-attachments/assets/6acd1773-1861-46ec-bfa4-c59bdedf2464" width="400"><br>
  <em>Figure 5: Comparison of dB distributions before and after data cleaning.</em>
</p>

---

# Data Processing

The stereo `.wav` recordings were first converted into mono signals.

Afterwards:

1. An A-weighting filter was applied
2. The sound pressure level was calculated using the reference value:

```math
p_{ref} = 2 \times 10^{-5}
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/c38bb421-7f3c-490f-8673-505ef55ad952" width="400"><br>
  <em>Figure 6: Conversion pipeline from wav signal to dB.</em>
</p>

---

## Octave Band Analysis

In addition to broadband dB estimation, octave-band-based analysis was also performed.

The audio signals were filtered into octave frequency bands before dB conversion.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9de61454-d2bd-465d-8cc1-43a3a9099d1d" width="400"><br>
  <em>Figure 7: Octave-band filtering and dB conversion process.</em>
</p>

---

# Machine Learning and Deep Learning Models

The following supervised learning approaches were evaluated:

| Model | Type |
|---|---|
| LightGBM | Machine Learning |
| XGBoost | Machine Learning |
| Linear Regression | Machine Learning |
| LSTM | Deep Learning |
| BiLSTM | Deep Learning |
| GRU | Deep Learning |
| BiGRU | Deep Learning |

---

# Deep Learning Architecture

The recurrent neural network architectures used the following structure:

- Layer 1:
  - 128 units
  - LeakyReLU (`alpha = 0.1`)
- Layer 2:
  - 64 units
  - LeakyReLU (`alpha = 0.2`)
- Dropout:
  - 0.3
- Fully connected output layer:
  - 1 neuron

---

# Hyperparameter Optimization

For LightGBM and XGBoost:

- Grid Search
- Random Search

were applied for hyperparameter optimization.

The tuning process resulted in only minor performance improvements.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0efbabe2-9b75-4896-8976-75707855975c" width="600"><br>
  <em>Figure 8: Comparison before and after hyperparameter optimization.</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b730da39-f704-4d54-af73-a7ef06e2bc07" width="600"><br>
  <em>Figure 9: Example grid search visualization.</em>
</p>

---

# Validation Strategy

5-fold cross-validation was used for model evaluation.

Performance was evaluated using:

- R² Score
- RMSE
- MAE

Care was taken to minimize potential data leakage between training and testing splits.

---

# Results

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4bc3226-3b2d-449d-9745-6d6bc34cf2ad" width="600"><br>
  <em>Figure 10: Average R² performance comparison of all models.</em>
</p>

The results indicate that predicting roadside traffic noise solely from metadata features remains challenging.

Most models achieved similar performance levels, suggesting that additional acoustic or contextual information may be required for higher prediction accuracy.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7b2bf966-3651-4588-8f2a-1c9f3ca252cc" width="700"><br>
  <em>Figure 11: Example prediction output using LightGBM.</em>
</p>

---

# Future Work

Potential future improvements include:

- Integration of raw acoustic features
- Spectrogram-based deep learning
- Multi-modal sensor fusion
- Transfer learning approaches
- Transformer-based architectures
- Real-world deployment studies
- Environmental sensor integration

---

  # Contributors

- [Dr. Mustafa Demetgül](https://github.com/mdemetgul179)  
  Scientific Supervision, AI Methodology and Research Guidance

- [Marc Heinzelmann](https://github.com/marc-heinzelmann)  
  Student Researcher
