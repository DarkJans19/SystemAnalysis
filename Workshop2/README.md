# Child Mind Institute — Problematic Internet Use among Youth  
### Systems Analysis – Workshops Repository

This repository hosts the hands‑on workshops for the Systems Analysis course. Each workshop builds upon the Kaggle challenge **“Problematic Internet Use”** to practise real‑world data analysis, modelling and architecture design.

---

## Group
| Name | ID |
|------|----|
| **Jan Henrik Sánchez Jerez** | 20231020130 |
| **Sebastián Villarreal Castro** | 20221020059 |
| **Juan David Quiroga** | 20222020206 |

---

## Index
| # | Workshop | Folder / Link | Focus |
|---|----------|---------------|-------|
| 1 | Workshop 1 | [`/Workshop1`](./Workshop1) | Exploratory analysis & early modelling |
| 2 | Workshop 2 – *System Impact Index (SII)* predictive platform | [`/Workshop2`](./Workshop2) | Architecture‑driven design & ordinal ML |

---

## Workshop 1  – Exploratory Analysis (recap)

We analysed the data step‑by‑step: defined the **IBS** focus variable, inspected relationships, and designed a basic pipeline to understand the data flow.  
![pipeline](https://github.com/user-attachments/assets/5cd71b34-9b28-48aa-be54-b4ad88bf933d)

Key outcomes:

* Mapped all questionnaires and HBN instruments to spot data gaps and SII mis‑reports.  
* Built an initial model (beta) and iterated with feedback from cross‑validation.

---

## Workshop 2 – System Impact Index (SII) predictive platform

### Scope & Summary
We design a data‑driven system that predicts SII severity (0 – 4) using a fusion of **PCIAT questionnaires** and **wrist‑worn actigraphy**. The ordinal nature of SII demands models and metrics that understand “distance” between classes, so we optimise **Quadratic Weighted Kappa (QWK)** instead of plain accuracy.
### Design requirements (excerpt)
* **Prediction quality first** – maximise QWK.  
* **Robust missing‑value handling** – median or temporal interpolation.  
* **Noise / chaos tolerance** – algorithms resilient to outliers (e.g., Random Forest).  
* **Ordinal‑aware modelling** – custom QWK‑loss for deep learners.  
* **Probabilistic adaptability** – Bayesian layers where feasible. 

### High‑level architecture
Detailed UML, sequence and data‑flow diagrams are stored in the Workshop folder (see `/Workshop2/diagrams`). 

### Technical stack
| Layer | Tool | Rationale |
|-------|------|-----------|
| Language | Python 3.11 | Data‑science standard |
| Data I/O | pandas • polars • pyarrow | Fast tabular + Parquet handling |
| Time‑series feats | tsfresh • pyts | Actigraphy feature extraction |
| Models | scikit‑learn (Random Forest) • PyTorch (1‑D CNN) | Tabular & sequential, supports custom loss |
| Environment | Google Colab | Free GPUs, collaboration |

### Development documentation

1. Data ingestion & schema checks  
2. Imputation experiments  
3. Feature engineering (ENMO, Angle‑Z, circadian stats)  
4. Baseline Random Forest + stratified k‑fold CV  
5. Ordinal CNN prototype with QWK‑loss  
6. Error analysis & chaos‑variable mitigation

Architecture steps in the diagram file map 1‑to‑1 to notebook sections for full traceability between **process** and **architecture**.

---
