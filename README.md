## Welcome to Workshop – Systems Analysis

Welcome to this work area of the **Systems Analysis** subject. In this repository you will find the respective activities or workshops carried out by our group.

## Group
- **Jan Henrik Sánchez Jerez** – 20231020130  
- **Sebastián Villarreal Castro** – 20221020059  
- **Juan David Quiroga** – 20222020206  

## Index
| # | Workshop | Description |
|---|----------|-------------|
| 1 | Workshop 1 | *Basics of requirements elicitation* |
| 2 | Workshop 2 – System Impact Index (SII) predictive platform | *Architecture & data‑driven design* |

---

This workshop analyses and designs a data‑driven system that predicts the severity of **Problematic Internet Use** in children (SII 0–4). It integrates questionnaire data (PCIAT) with wrist‑worn actigraphy signals to build an ordinal‑aware machine‑learning pipeline.
### Summary
- **Goal:** predict SII class (0 = low, 4 = high) via supervised multi‑class classification.  
- **Data sources:**  
  - `train/test.csv` – demographics & clinical questionnaire  
  - `series_train.parquet` – 5‑second actigraphy windows  
- **Key challenges:** high missing‑value ratio in physiological signals; subjectivity in PCIAT; temporal noise & chaos in wearables.  
- **Evaluation:** **Quadratic Weighted Kappa (QWK)**, chosen for its ordinal sensitivity.

### Design‑driven requirements
| Area | Requirement |
|------|-------------|
| **Prediction quality** | Optimise QWK over raw accuracy. |
| **Missing data** | Median / temporal interpolation imputation. |
| **Noise tolerance** | Use algorithms robust to outliers (e.g. Random Forest). |
| **Ordinal awareness** | Custom QWK‑based loss for deep models. |
| **Probabilistic adaptability** | Bayesian or uncertainty‑aware layers where feasible. |

### Technical stack
| Layer | Technology | Why |
|-------|------------|-----|
| Language | Python 3.11 | Community ML standard |
| Data | **pandas / polars / pyarrow** | Fast tabular & parquet I/O |
| Features | **tsfresh / pyts** | Time‑series feature extraction |
| Modelling | **scikit‑learn** (Random Forest), **PyTorch** (1‑D CNN/RNN) | Handles tabular & sequential data; supports custom QWK loss |
| Environment | **Google Colab** | GPU access, collaboration |

### Development‑process reference
The full development PDF documents:

1. Data ingestion & schema checks  
2. Imputation experiments (median vs interpolation)  
3. Feature engineering for actigraphy (ENMO, Angle‑Z, circadian stats)  
4. Baseline Random Forest with stratified k‑fold CV  
5. Deep ordinal‑CNN prototype with QWK‑loss  
6. Error analysis & chaos‑variable mitigation

Architecture diagrams in Workshop 2 map one‑to‑one onto these steps to ensure traceability between *process* and *architecture*.

---

### References
- Kaggle discussion on **QWK** loss functions  
- Parent‑Child Internet Addiction Test clinical questionnaire  
- WristPy actigraphy feature guides
