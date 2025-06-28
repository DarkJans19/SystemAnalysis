# Workshop 3: Problematic Internet Use Prediction

## Description

This project implements an ordinal prediction system for the Severely Impairment Index (SII), following the workflow proposed in the Kaggle competition ["Child Mind Institute - Problematic Internet Use"](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview).

The goal is to transform raw data from adolescents into ready-to-evaluate predictions using preprocessing techniques, feature engineering, and machine learning models.

---

## Folder Structure

- `data/`: Original data files (`train.csv`, `test.csv`, etc.) and variable dictionary.
- `src/`: Scripts for processing, modeling, and utilities.
- `outputs/`: Results and prediction files (`submission.csv`).
- `informe_workshop3.md`: Formal report for Workshop 3.

---

## Dependency Installation

Make sure you have Python 3.7+ installed.  
Install the dependencies by running:

```
pip install -r requirements.txt
```

---

## Pipeline Execution

1. **Place the data files** (`train.csv`, `test.csv`) in the `data/` folder.
2. **Run the main script** from the project root:

   ```
   python src/main.py
   ```


3. **Result:**  
The prediction file `submission.csv` will be generated in the `outputs/` folder, ready to be submitted to Kaggle.

---

## System Workflow

The system follows this workflow:

1. **Data Loading:**  
Reads the data files and extracts the relevant variables.

2. **Preprocessing:**  
- Missing value imputation (mean).
- Categorical variable encoding (e.g., sex).
- Selection of numerical features present in both sets.

3. **Target Variable Grouping:**  
The `PCIAT-PCIAT_Total` variable is grouped into 4 ordinal categories (0, 1, 2, 3) using quartiles, to match the ordinal nature of the problem.

4. **Training and Validation:**  
A Random Forest model is trained and validated using the Quadratic Weighted Kappa (QWK) metric, the official metric for the competition.

5. **Prediction and Submission Generation:**  
Predictions for the test set are generated and saved in the required format (`id,sii`).

---

## Submission Format

The `outputs/submission.csv` file will have the following format:

---

```
id,sii
00008ff9,3
000fd460,0
...
```

- `id`: Identifier of each test sample.
- `sii`: Ordinal prediction for SII (values from 0 to 3).

---

## Requirements

The `requirements.txt` file includes:

```
pandas
numpy
scikit-learn
jupyter
matplotlib
seaborn
```

---

## References

- [Kaggle Competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview)

---

## Group
- **Jan Henrik Sánchez Jerez** – 20231020130  
- **Juan David Quiroga** – 20222020206
